import asyncio
import contextlib
import sys
import time
from dataclasses import dataclass

from sentinel.application.audit_orchestrator import AuditOrchestrator
from sentinel.application.risk_engine import RiskEngine
from sentinel.config.settings import Settings, get_settings
from sentinel.domain.entities.pull_request import PullRequest
from sentinel.domain.services.ast_security_service import ASTSecurityService
from sentinel.domain.services.document_service import DocumentService
from sentinel.domain.services.semantic_service import SemanticService
from sentinel.domain.value_objects.severity_level import SeverityLevel
from sentinel.infrastructure.github.github_client import GitHubClient
from sentinel.infrastructure.llm.llm_service import LLMService
from sentinel.infrastructure.semantic.embedding_engine import EmbeddingEngine
from sentinel.monitoring.logger import get_logger
from sentinel.monitoring.metrics import metrics
from sentinel.workers.job_queue import JobQueue

logger = get_logger(__name__)

MAX_CODE_LENGTH = 2 * 1024 * 1024
SMALL_PAYLOAD_THRESHOLD = 20_000
TARGET_LATENCY_SECONDS = 0.1
# Upper bound on semantic-corpus entries per job (each is embedded per assessment).
CORPUS_MAX_UNITS = 200


def _build_github_client(
    settings: Settings, installation_id: str | None = None
) -> GitHubClient | None:
    """Build a GitHub client for ONE installation, or None when GitHub is disabled.

    ``installation_id`` arrives on the job, taken from the delivery that queued
    it, so a single deployment can serve every installation of the App. It
    falls back to the configured id, which is what keeps manual triggers and
    the e2e harness — neither of which carries a real delivery — working
    unchanged.

    Kept at module scope so tests can monkeypatch ``bw_module._build_github_client``
    to inject a fake.
    """
    if not settings.ENABLE_GITHUB:
        return None
    return GitHubClient(
        app_id=settings.GITHUB_APP_ID,
        installation_id=installation_id or settings.GITHUB_INSTALLATION_ID,
        private_key=settings.GITHUB_PRIVATE_KEY,
        api_base_url=settings.GITHUB_API_BASE_URL,
    )


def _build_llm_service(settings: Settings) -> LLMService:
    """Build the LLM service for the worker (mirrors webhook_controller.get_llm_service).

    Kept at module scope so tests can monkeypatch ``bw_module._build_llm_service``.
    With no LLM_API_KEY, ``enable_llm`` is False and the service returns fallback
    strings without any network call — so wiring it in is safe with creds absent.
    Base URL / model / key come from env (LLM_BASE_URL / LLM_MODEL / LLM_API_KEY),
    so the provider is swappable without a code change.
    """
    llm_enabled = settings.ENABLE_LLM and bool(settings.LLM_API_KEY)
    return LLMService(
        enable_llm=llm_enabled,
        max_calls=settings.LLM_MAX_CALLS,
        timeout=settings.LLM_TIMEOUT,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        model=settings.LLM_MODEL,
    )


@dataclass(frozen=True)
class ReviewOutcome:
    """What one review produced.

    ``severity`` is None when the review was skipped (the PR data could not be
    fetched), which is deliberately distinct from a LOW verdict: one means
    "looked, found little", the other "could not look". ``report`` is the
    markdown body, kept so a caller that is not posting a comment — the GitHub
    Action writing a job summary — does not have to rebuild it and pay for the
    LLM calls twice.
    """

    severity: SeverityLevel | None = None
    report: str | None = None

    @property
    def skipped(self) -> bool:
        return self.severity is None


def _safe_assessment() -> dict:
    """Assessment-shaped defaults, used when risk analysis fails outright."""
    return {
        "severity": SeverityLevel.LOW,
        "complexity": 1,
        "maintainability": 100.0,
        "security_findings_count": 0,
        "security": {"findings": [], "severity": SeverityLevel.LOW},
        "semantic_findings_count": 0,
        "semantic": {"findings": [], "severity": SeverityLevel.LOW},
    }


class BackgroundWorker:
    # Pause after a failed loop iteration. Only the error path is paced: both
    # queues already block or poll inside dequeue(), so a delay on the success
    # path is pure idle time between jobs. This exists solely to stop a tight
    # spin when dequeue() itself keeps raising (e.g. Redis unreachable).
    ERROR_BACKOFF_SECONDS = 2.0

    # Upper bound on cached per-installation clients. Each one holds an access
    # token and a small corpus cache, so the cap is what stops a widely
    # installed App from growing them without limit.
    MAX_CACHED_CLIENTS = 32

    def __init__(self, queue: JobQueue) -> None:
        self.queue = queue
        # Count of jobs whose processing has fully completed. Deterministic completion
        # signal (used by tests) and basic throughput observability.
        self.processed_count = 0
        # One GitHub client per installation, reused across jobs. Keyed by
        # installation id; "" is the configured-fallback client.
        self._clients: dict[str, GitHubClient] = {}

    def _client_for(self, settings: Settings, installation_id: object) -> GitHubClient | None:
        """Resolve the GitHub client for one installation, reused across jobs.

        Reuse is the point: the installation token is cached on the client, so
        building a fresh one per job would mint a new token for every pull
        request. A job with no installation id (manual trigger, e2e harness)
        maps to the empty key and gets the configured fallback client.
        """
        key = str(installation_id) if installation_id else ""

        cached = self._clients.get(key)
        if cached is not None:
            return cached

        client = _build_github_client(settings, key or None)
        if client is None:
            return None

        self._clients[key] = client
        if len(self._clients) > self.MAX_CACHED_CLIENTS:
            # A plain dict keeps insertion order, so the first key is the oldest.
            del self._clients[next(iter(self._clients))]
        return client

    @staticmethod
    def _format_risk_line(pr_number: object, risk: object) -> str:
        risk_value = risk.value if isinstance(risk, SeverityLevel) else str(risk).upper()
        return f"PR #{pr_number} Risk: {risk_value}"

    @staticmethod
    def _assess(
        job: dict, risk_engine: RiskEngine, extra_security_findings: list | None = None
    ) -> tuple[PullRequest, dict]:
        """Read/coerce/truncate the job's code and run the resilient assessment.

        Returns the ``PullRequest`` identity plus the full assessment dict (safe
        defaults on unexpected failure). Shared by ``process_job`` (one-liner) and
        ``start`` (report) so the code is assessed exactly once per job.
        """
        pull_request = PullRequest(
            repo=job.get("repo", "unknown"),
            pr_number=job.get("pr_number", 0),
        )

        code = job.get("code", "")
        if code is None:
            code = ""
        elif not isinstance(code, str):
            code = str(code)

        if len(code) > MAX_CODE_LENGTH:
            logger.warning(
                "Code payload too large for repo=%s pr_number=%s; truncating from %s chars",
                pull_request.repo,
                pull_request.pr_number,
                len(code),
            )
            code = code[:MAX_CODE_LENGTH]

        # Real corpus fetched from the PR's base ref (M5). With no corpus the
        # semantic engine is skipped outright rather than compared against a
        # stand-in: a tiny placeholder corpus can cross the 0.9 threshold and
        # fabricate a HIGH, which _overall_severity turns into a failing check.
        existing_code_list = job.get("existing_code_list")
        if not isinstance(existing_code_list, list) or not existing_code_list:
            existing_code_list = []
            logger.info(
                "No semantic corpus for repo=%s pr_number=%s; skipping duplicate detection",
                pull_request.repo,
                pull_request.pr_number,
            )

        assessment_start = time.monotonic()
        try:
            assessment = risk_engine.assess_resilient(
                code=code,
                existing_code_list=existing_code_list,
                warn_threshold_seconds=TARGET_LATENCY_SECONDS,
                extra_security_findings=extra_security_findings,
            )
        except Exception:
            logger.exception(
                "Risk assessment failed unexpectedly for repo=%s pr_number=%s; defaulting to LOW",
                pull_request.repo,
                pull_request.pr_number,
            )
            assessment = _safe_assessment()

        assessment_elapsed = time.monotonic() - assessment_start
        if len(code) <= SMALL_PAYLOAD_THRESHOLD and assessment_elapsed > TARGET_LATENCY_SECONDS:
            logger.warning(
                "Small payload PR analysis exceeded target: repo=%s pr=%s elapsed=%.4fs",
                pull_request.repo,
                pull_request.pr_number,
                assessment_elapsed,
            )

        return pull_request, assessment

    @staticmethod
    def _ast_security_findings(job: dict) -> list:
        """AST findings for every changed .py file whose full source parses.

        Restricted to lines this PR actually added: the analyzer sees the whole
        module, so without the filter it would report pre-existing problems the
        author never touched — accurate, but not their PR's business.

        Files handled here are removed from the regex blob by
        ``_strip_ast_covered_lines``, so the two engines never both report the
        same line and no de-duplication guesswork is needed.
        """
        file_contents = job.get("file_contents")
        line_map = job.get("line_map")
        if not isinstance(file_contents, dict) or not isinstance(line_map, list):
            return []

        added_by_file: dict[str, set[int]] = {}
        for entry in line_map:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                path, line_no = entry
                if isinstance(path, str) and isinstance(line_no, int):
                    added_by_file.setdefault(path, set()).add(line_no)

        analyzer = ASTSecurityService()
        findings: list = []
        covered: list[str] = []
        for path, source in file_contents.items():
            if not isinstance(path, str) or not path.lower().endswith(".py"):
                continue
            file_findings = analyzer.analyze_source(source, path)
            if file_findings is None:
                continue  # unparseable — leave it to the regex engine
            covered.append(path)
            touched = added_by_file.get(path, set())
            findings.extend(
                f for f in file_findings if isinstance(f.line, int) and f.line in touched
            )

        if covered:
            job["ast_covered_files"] = covered
            logger.info(
                "AST security analysis covered %s file(s): %s", len(covered), ", ".join(covered)
            )
        return findings

    @staticmethod
    def _strip_ast_covered_lines(job: dict) -> None:
        """Remove AST-analyzed files from the regex blob, in lockstep with line_map.

        The blob and the map are index-parallel by construction, so both are
        filtered by the same predicate to keep them that way.
        """
        covered = set(job.get("ast_covered_files") or [])
        line_map = job.get("line_map")
        code = job.get("code")
        if not covered or not isinstance(line_map, list) or not isinstance(code, str):
            return

        code_lines = code.split("\n")
        if len(code_lines) != len(line_map):
            return  # shapes disagree; leave both alone rather than corrupt them

        kept = [
            (text, entry)
            for text, entry in zip(code_lines, line_map)
            if not (
                isinstance(entry, (list, tuple))
                and len(entry) == 2
                and entry[0] in covered
            )
        ]
        job["code"] = "\n".join(text for text, _ in kept)
        job["line_map"] = [entry for _, entry in kept]

    @staticmethod
    def process_job(job: dict, risk_engine: RiskEngine) -> str:
        pull_request, assessment = BackgroundWorker._assess(job, risk_engine)
        risk = assessment["severity"]
        logger.info(
            "Completed PR analysis for repo=%s pr_number=%s severity=%s",
            pull_request.repo,
            pull_request.pr_number,
            risk.value if isinstance(risk, SeverityLevel) else risk,
        )
        return BackgroundWorker._format_risk_line(pull_request.pr_number, risk)

    @staticmethod
    def _identity(job: dict) -> tuple[object, object, object]:
        """Extract (owner, bare_repo_name, pr_number) for GitHub API calls.

        ``repo`` may arrive as ``"owner/name"`` or a bare ``"name"``; owner is a
        separate job key. Returns the bare repo name so the API URL is well-formed.
        """
        owner = job.get("owner")
        repo_raw = job.get("repo")
        if isinstance(repo_raw, str) and "/" in repo_raw:
            repo_name: object = repo_raw.split("/", 1)[1]
        else:
            repo_name = repo_raw
        return owner, repo_name, job.get("pr_number")

    @staticmethod
    def _fetch_refs(job: dict, github_client: GitHubClient, owner, repo_name, pr_number) -> None:
        """Stash the PR's head/base SHAs on the job. Best-effort, never raises.

        Runs BEFORE the file fetch: ``head_sha`` is what a check run attaches to,
        so it has to be available even when the file fetch is what failed.
        """
        try:
            refs = github_client.get_pull_request_refs(owner, repo_name, pr_number)
        except Exception:
            logger.exception("Failed to fetch PR refs for repo=%s pr=%s", repo_name, pr_number)
            return
        head_sha = refs.get("head_sha")
        if head_sha:
            job["head_sha"] = head_sha  # check runs attach to the head commit
        base_sha = refs.get("base_sha")
        if base_sha:
            job["base_sha"] = base_sha

    @staticmethod
    def _fetch_pr_data(job: dict, github_client: GitHubClient | None) -> bool:
        """Populate ``job['code']``/``files``/``file_contents`` from the PR. Failure-safe.

        Returns whether the review may proceed on trustworthy input: ``False``
        ONLY when a fetch that should have happened raised. "Nothing to fetch"
        (no client, code already on the job, owner-less job) is not a failure —
        neither is a failed corpus build, which stays best-effort.

        Fetched data wins over payload-supplied values when non-empty; payload values
        are kept otherwise.
        """
        if github_client is None or job.get("code"):
            return True
        owner, repo_name, pr_number = BackgroundWorker._identity(job)
        if not owner or not repo_name or pr_number is None:
            return True

        BackgroundWorker._fetch_refs(job, github_client, owner, repo_name, pr_number)

        try:
            fetched = github_client.get_pull_request_data(owner, repo_name, pr_number)
        except Exception:
            logger.exception("Failed to fetch PR data for repo=%s pr=%s", repo_name, pr_number)
            return False

        code = fetched.get("code")
        if isinstance(code, str) and code.strip():
            job["code"] = code
        files = fetched.get("files")
        if isinstance(files, list) and files:
            job["files"] = files
        file_contents = fetched.get("file_contents")
        if isinstance(file_contents, dict) and file_contents:
            job["file_contents"] = file_contents
        line_map = fetched.get("line_map")
        if isinstance(line_map, list) and line_map:
            job["line_map"] = line_map

        logger.info(
            "Fetched %s chars of PR code across %s files for repo=%s pr=%s",
            len(code) if isinstance(code, str) else 0,
            len(files) if isinstance(files, list) else 0,
            repo_name,
            pr_number,
        )

        # Build the semantic corpus from the base branch (what the PR lands on),
        # chunked into top-level units. Best-effort: on any failure the job keeps
        # no corpus and _assess skips duplicate detection entirely.
        base_sha = job.get("base_sha")
        if base_sha:
            try:
                # Rank the corpus toward what this PR actually touched.
                corpus_files = github_client.get_repo_code_corpus(
                    owner, repo_name, base_sha, prefer_paths=job.get("files")
                )
                chunks = [
                    chunk
                    for content in corpus_files
                    for chunk in SemanticService.chunk_code_units(content)
                ]
                if chunks:
                    job["existing_code_list"] = chunks[:CORPUS_MAX_UNITS]
                    logger.info(
                        "Built semantic corpus of %s units from %s files for repo=%s pr=%s",
                        len(job["existing_code_list"]),
                        len(corpus_files),
                        repo_name,
                        pr_number,
                    )
            except Exception:
                logger.exception(
                    "Semantic corpus build failed for repo=%s pr=%s", repo_name, pr_number
                )

        return True

    @staticmethod
    def _post_review(
        job: dict,
        assessment: dict,
        orchestrator: AuditOrchestrator,
        github_client: GitHubClient | None,
    ) -> str | None:
        """Build a structured report from the assessment and post it. Failure-safe.

        Returns the report body when one was built, so a caller can surface it
        somewhere other than a PR comment. Posting may have failed even when a
        report comes back — that is the fork-PR case, where the token is
        read-only but the review itself is perfectly valid.
        """
        if github_client is None:
            return None
        owner, repo_name, pr_number = BackgroundWorker._identity(job)
        if not owner or not repo_name or pr_number is None:
            return None
        try:
            security = assessment.get("security", {})
            findings = security.get("findings", []) if isinstance(security, dict) else []
            all_findings, report = orchestrator.run_full_review(
                code=job.get("code", ""),
                findings=findings,
                risk=assessment["severity"],
                files=job.get("files"),
                file_contents=job.get("file_contents"),
                complexity=assessment.get("complexity"),
                maintainability=assessment.get("maintainability"),
                semantic_findings_count=assessment.get("semantic_findings_count"),
            )
            posted = github_client.upsert_comment(owner, repo_name, pr_number, report)
            metrics.counter_inc(
                "sentinel_github_posts_total",
                {"kind": "comment", "outcome": "ok" if posted else "failed"},
            )
            logger.info(
                "Worker posted review comment=%s repo=%s pr=%s", posted, repo_name, pr_number
            )

            # Native check run alongside the comment: pass/fail in the merge box
            # plus per-line annotations. Needs the app's Checks:write permission;
            # without it create_check_run returns False (skipped, non-fatal).
            head_sha = job.get("head_sha")
            if get_settings().ENABLE_CHECKS and head_sha:
                payload = orchestrator.build_check_payload(
                    all_findings,
                    assessment["severity"],
                    line_map=job.get("line_map"),
                    semantic_findings_count=assessment.get("semantic_findings_count"),
                )
                check_posted = github_client.create_check_run(
                    owner,
                    repo_name,
                    head_sha,
                    conclusion=payload["conclusion"],
                    title=payload["title"],
                    summary=payload["summary"],
                    text=report,
                    annotations=payload["annotations"],
                )
                metrics.counter_inc(
                    "sentinel_github_posts_total",
                    {"kind": "check_run", "outcome": "ok" if check_posted else "failed"},
                )
                logger.info(
                    "Worker posted check run=%s repo=%s pr=%s",
                    check_posted,
                    repo_name,
                    pr_number,
                )
            return report
        except Exception:
            logger.exception("Failed to post PR review for repo=%s pr=%s", repo_name, pr_number)
            return None

    @staticmethod
    def _post_fetch_failure(job: dict, github_client: GitHubClient | None) -> None:
        """Report that the review was skipped, rather than leaving the PR looking clean.

        A neutral check run: visible in the merge box so nobody reads silence as a
        pass, but never blocking a merge on Sentinel's own outage. No comment —
        a broad GitHub failure would otherwise write one on every open PR.
        Failure-safe, like every other GitHub edge.
        """
        if github_client is None:
            return
        owner, repo_name, pr_number = BackgroundWorker._identity(job)
        head_sha = job.get("head_sha")
        if not owner or not repo_name or pr_number is None or not head_sha:
            return
        if not get_settings().ENABLE_CHECKS:
            return
        try:
            posted = github_client.create_check_run(
                owner,
                repo_name,
                head_sha,
                conclusion="neutral",
                title="Review skipped",
                summary=(
                    "Sentinel could not fetch this pull request's contents from the "
                    "GitHub API. No review was performed."
                ),
            )
        except Exception:
            logger.exception(
                "Failed to post skipped-review check for repo=%s pr=%s", repo_name, pr_number
            )
            return
        metrics.counter_inc(
            "sentinel_github_posts_total",
            {"kind": "check_run", "outcome": "skipped" if posted else "failed"},
        )
        logger.info(
            "Worker posted skipped-review check=%s repo=%s pr=%s", posted, repo_name, pr_number
        )

    def _process_one(
        self,
        job: dict,
        risk_engine: RiskEngine,
        orchestrator: AuditOrchestrator,
        github_client: GitHubClient | None,
    ) -> None:
        """Run one job through the review pipeline. See ``run_review``."""
        run_review(job, risk_engine, orchestrator, github_client)

    async def start(self) -> None:
        embedding_engine = EmbeddingEngine()
        semantic_service = SemanticService(embedding_engine)
        risk_engine = RiskEngine(semantic_service=semantic_service)

        settings = get_settings()
        llm_service = _build_llm_service(settings)
        orchestrator = AuditOrchestrator(
            self.queue,
            llm_service=llm_service,
            document_service=DocumentService(),
        )

        # Re-queue any jobs a previous (crashed) worker left mid-processing.
        # No-op on the in-memory queue.
        try:
            await self.queue.recover_pending()
        except Exception:
            logger.exception("Job recovery failed; starting with the queue as-is")

        while True:
            try:
                job = await self.queue.dequeue()
                # Resolved per job, not once per process: each installation
                # needs its own credentials, and the wrong token fetches
                # nothing and posts nothing.
                github_client = self._client_for(settings, job.get("installation_id"))
                # Shielded so a shutdown mid-job lets that job finish and ack
                # instead of being abandoned unacked. The shield alone would
                # return control immediately and orphan the inner task, so the
                # cancellation handler awaits it to completion before re-raising.
                in_flight = asyncio.ensure_future(
                    self._run_and_ack(job, risk_engine, orchestrator, github_client)
                )
                try:
                    await asyncio.shield(in_flight)
                except asyncio.CancelledError:
                    with contextlib.suppress(Exception):
                        await in_flight
                    raise
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Worker failed to process job; continuing")
                await asyncio.sleep(self.ERROR_BACKOFF_SECONDS)

    async def _run_and_ack(
        self,
        job: dict,
        risk_engine: RiskEngine,
        orchestrator: AuditOrchestrator,
        github_client: GitHubClient | None,
    ) -> None:
        """Process one job, then ack it. The unit shutdown is allowed to finish."""
        # Offload the entire blocking pipeline so a slow GitHub GET / LLM call
        # can never stall the shared event loop (health, webhooks, queue intake).
        await asyncio.to_thread(
            self._process_one, job, risk_engine, orchestrator, github_client
        )
        # Only a fully processed job is acked; a crash before this line
        # leaves it in the processing list for recovery on next start.
        await self.queue.ack(job)
        self.processed_count += 1
        metrics.counter_inc("sentinel_jobs_processed_total")


def run_review(
    job: dict,
    risk_engine: RiskEngine,
    orchestrator: AuditOrchestrator,
    github_client: GitHubClient | None,
) -> ReviewOutcome:
    """Run one job's full pipeline: fetch -> assess -> emit -> post.

    Entirely synchronous/blocking (GitHub urllib + sklearn + openai). The worker
    runs it via ``asyncio.to_thread`` so none of it executes on the event loop;
    the GitHub Action calls it directly on the runner. It lives at module scope
    rather than on ``BackgroundWorker`` precisely so the Action can reach it
    without constructing a worker and a queue it will never use.
    """
    start_time = time.monotonic()

    # A failed fetch must never look like a clean PR: assessing the empty
    # code we did not manage to fetch would report "no issues" and a passing
    # check run. Say nothing about the code; say the review was skipped.
    if not BackgroundWorker._fetch_pr_data(job, github_client):
        pr_number = job.get("pr_number")
        skipped_line = f"PR #{pr_number} Risk: SKIPPED (fetch failed)"
        metrics.counter_inc("sentinel_reviews_skipped_total", {"reason": "fetch_failed"})
        logger.warning("%s", skipped_line)
        sys.stdout.write(f"{skipped_line}\n")
        sys.stdout.flush()
        BackgroundWorker._post_fetch_failure(job, github_client)
        metrics.observe("sentinel_job_duration_seconds", time.monotonic() - start_time)
        return ReviewOutcome()

    # Structure-aware analysis first: it claims the files it can parse, and
    # those lines are then withheld from the text-matching engine so a single
    # line is never reported by both.
    ast_findings = BackgroundWorker._ast_security_findings(job)
    BackgroundWorker._strip_ast_covered_lines(job)

    pull_request, assessment = BackgroundWorker._assess(job, risk_engine, ast_findings)
    risk = assessment["severity"]
    report_line = BackgroundWorker._format_risk_line(pull_request.pr_number, risk)
    metrics.counter_inc(
        "sentinel_reviews_total",
        {"severity": risk.value if isinstance(risk, SeverityLevel) else str(risk).upper()},
    )

    logger.info("%s", report_line)
    sys.stdout.write(f"{report_line}\n")
    sys.stdout.flush()

    report = BackgroundWorker._post_review(job, assessment, orchestrator, github_client)

    elapsed = time.monotonic() - start_time
    metrics.observe("sentinel_job_duration_seconds", elapsed)
    logger.info("Processed job in %.4fs", elapsed)
    return ReviewOutcome(severity=risk, report=report)
