"""Composition root for the GitHub Action.

Builds the same engines the worker builds, runs the same ``run_review``
pipeline, and turns the outcome into a process exit code. The only things that
differ from the hosted path are where the credential comes from (the runner's
``GITHUB_TOKEN`` rather than an App installation) and where the result is
surfaced (a job summary as well as a PR comment).
"""

import argparse
import os
import sys
from collections.abc import Mapping

import sentinel.workers.background_worker as bw_module
from sentinel.action.context import ActionContext, ActionContextError, read_action_context
from sentinel.application.audit_orchestrator import NEVER_FAILS, AuditOrchestrator
from sentinel.application.risk_engine import RiskEngine
from sentinel.config.settings import Settings, get_settings
from sentinel.domain.services.document_service import DocumentService
from sentinel.domain.services.semantic_service import SemanticService
from sentinel.domain.value_objects.severity_level import SeverityLevel
from sentinel.infrastructure.github.github_client import GitHubClient
from sentinel.infrastructure.llm.llm_service import LLMService
from sentinel.infrastructure.semantic.embedding_engine import EmbeddingEngine
from sentinel.monitoring.logger import get_logger
from sentinel.workers.background_worker import ReviewOutcome, run_review

logger = get_logger(__name__)

EXIT_OK = 0
EXIT_RISK = 1  # the review ran; the PR is at or above the fail-on threshold
EXIT_ERROR = 2  # the review could not run, or the PR could not be fetched

_NEVER_RANK = 99
DEFAULT_FAIL_ON = "high"

FORK_NOTE = (
    "> **Posted here only.** This pull request comes from a fork, so the token is "
    "read-only and Sentinel could not add a PR comment or a check run. The full "
    "review follows.\n\n"
)


def _build_github_client(context: ActionContext) -> GitHubClient:
    """The client for this run.

    Deliberately NOT gated on ``ENABLE_GITHUB``: in an Action, being handed a
    token IS the enablement signal, and this repository's own CI sets
    ``ENABLE_GITHUB=false`` globally — gating here would silently disable the
    very thing under test.
    """
    return GitHubClient.from_access_token(context.token, api_base_url=context.api_base_url)


def _build_llm_service(settings: Settings) -> LLMService:
    """Module-scope seam, mirroring the worker's builder so tests can swap it.

    Inert without a resolved key: no key means static recommendations and no
    network call, which is the normal case for a consumer who has not set a
    secret.
    """
    return bw_module._build_llm_service(settings)


def _build_risk_engine() -> RiskEngine:
    return RiskEngine(semantic_service=SemanticService(EmbeddingEngine()))


def _build_orchestrator(settings: Settings) -> AuditOrchestrator:
    # No queue: nothing is enqueued here, the review runs inline. The attribute
    # is only ever touched by enqueue_pull_request, which this path never calls.
    return AuditOrchestrator(
        None,
        llm_service=_build_llm_service(settings),
        document_service=DocumentService(),
    )


def _build_job(context: ActionContext) -> dict:
    job: dict = {
        "owner": context.owner,
        "repo": context.repo,
        "pr_number": context.pr_number,
    }
    if context.head_sha:
        # Stashed up front so a check run can still attach to the right commit
        # even if the refs call is what fails.
        job["head_sha"] = context.head_sha
    return job


def _fail_on_rank(value: str | None) -> int:
    """Lowest severity that should fail the job, as a rank."""
    normalized = (value or "").strip().lower()
    if normalized in ("never", "none", "off"):
        return _NEVER_RANK
    rank = SeverityLevel.rank_of(normalized)
    if rank is None:
        logger.warning("Unknown fail-on value %r; falling back to %r", value, DEFAULT_FAIL_ON)
        return SeverityLevel.rank_of(DEFAULT_FAIL_ON)
    return rank


def _canonical_fail_on(value: str | None) -> str:
    """The threshold as the check run should see it: a severity name or NEVER.

    Canonicalised here, not there, so the orchestrator never has to know this
    module's input vocabulary ("off", "none", a typo that falls back to high).
    """
    rank = _fail_on_rank(value)
    if rank == _NEVER_RANK:
        return NEVER_FAILS
    for level in SeverityLevel:
        if level.rank == rank:
            return level.value
    return DEFAULT_FAIL_ON.upper()


def _severity_text(severity: object) -> str:
    if isinstance(severity, SeverityLevel):
        return severity.value
    return str(severity).upper() if severity is not None else ""


def _exit_code(outcome: ReviewOutcome, fail_on: int, *, fail_on_error: bool) -> int:
    """A skipped review is an error, not a pass.

    This mirrors the hosted path's rule: a review that could not look at the
    code must never report the same thing as a review that looked and found
    nothing.
    """
    if outcome.skipped:
        return EXIT_ERROR if fail_on_error else EXIT_OK

    rank = SeverityLevel.rank_of(outcome.severity)
    if rank is not None and rank >= fail_on:
        return EXIT_RISK
    return EXIT_OK


def _append_file(path: str | None, text: str) -> None:
    """Append to one of the runner's command files. Never raises.

    Instrumentation must not fail a review that already succeeded — the same
    rule the metrics registry follows.
    """
    if not path or not text:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text)
    except Exception:  # pragma: no cover - defensive, exercised via a bad path
        logger.warning("Could not write to %s", path)


def _write_step_summary(report: str | None, env: Mapping[str, str], *, is_fork: bool) -> None:
    """Publish the report to the workflow summary.

    This is the review's guaranteed delivery channel: a plain file append, so
    it works when the token is read-only and no comment or check run can be
    posted. On a fork PR it is the only channel there is.
    """
    if not report:
        return
    prefix = FORK_NOTE if is_fork else ""
    _append_file(env.get("GITHUB_STEP_SUMMARY"), f"{prefix}{report}\n")


def _write_outputs(outcome: ReviewOutcome, env: Mapping[str, str]) -> None:
    """Expose the result to later workflow steps via $GITHUB_OUTPUT."""
    lines = (
        f"severity={_severity_text(outcome.severity)}\n"
        f"skipped={'true' if outcome.skipped else 'false'}\n"
    )
    _append_file(env.get("GITHUB_OUTPUT"), lines)


def _parse_args(argv: list[str] | None, env: Mapping[str, str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sentinel.action",
        description="Review the current pull request and report the result.",
    )
    parser.add_argument(
        "--fail-on",
        default=env.get("SENTINEL_FAIL_ON", DEFAULT_FAIL_ON),
        help="lowest severity that fails the job: critical|high|medium|low|never",
    )
    parser.add_argument(
        "--no-fail-on-error",
        dest="fail_on_error",
        action="store_false",
        help="exit 0 even when the review could not run",
    )
    parser.set_defaults(fail_on_error=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    """Run one review from the Actions environment. Returns a process exit code."""
    environment: Mapping[str, str] = env if env is not None else os.environ
    args = _parse_args(argv, environment)

    try:
        context = read_action_context(environment)
    except ActionContextError as exc:
        sys.stderr.write(f"Sentinel: {exc}\n")
        # fail-on-error is documented as covering "no PR context, fetch failed".
        # Honouring it only for the fetch half made the flag useless to the
        # workflow that most needs it: one whose triggers do not all carry a
        # pull request.
        return EXIT_ERROR if args.fail_on_error else EXIT_OK

    if context.is_fork:
        logger.warning(
            "PR #%s comes from a fork: the token is read-only, so the review will be "
            "published to the job summary instead of a PR comment.",
            context.pr_number,
        )

    settings = get_settings()
    github_client = _build_github_client(context)
    orchestrator = _build_orchestrator(settings)
    job = _build_job(context)
    # So the check run agrees with the job status. Without it a repo running
    # `fail-on: never` gets a green job beside a red check, and the reader has
    # to work out which one to believe.
    job["check_fail_on"] = _canonical_fail_on(args.fail_on)

    outcome = run_review(job, _build_risk_engine(), orchestrator, github_client)

    _write_step_summary(outcome.report, environment, is_fork=context.is_fork)
    _write_outputs(outcome, environment)

    return _exit_code(outcome, _fail_on_rank(args.fail_on), fail_on_error=args.fail_on_error)
