"""The GitHub Action entry point: same review, different delivery.

Drives ``runner.main`` with a fake GitHub client injected through the
module-level seam, the same idiom the worker tests use. The environment is
passed in as a dict rather than patched into ``os.environ``, so each case
states exactly the environment it is about.
"""

import json
import pathlib
import shutil
import tempfile

import pytest

import sentinel.action.runner as runner
from sentinel.action.context import ActionContext
from sentinel.domain.value_objects.severity_level import SeverityLevel
from sentinel.workers.background_worker import ReviewOutcome

RISKY_CODE = 'api_key = "sk-abcdefghijklmnopqrst"\npassword = "hunter2"\n'
BENIGN_CODE = "total = 1 + 1\n"


@pytest.fixture
def workdir():
    """Scratch space for the event payload and the runner's command files."""
    path = tempfile.mkdtemp(prefix="sentinel-action-run-")
    try:
        yield pathlib.Path(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)


class _FakeGitHub:
    def __init__(self, code: str = RISKY_CODE, *, writable: bool = True) -> None:
        self._code = code
        self._writable = writable
        self.fetched_with: tuple | None = None
        self.posted: list[tuple] = []
        self.check_runs: list[dict] = []

    def get_pull_request_data(self, owner, repo, pr_number) -> dict:
        self.fetched_with = (owner, repo, pr_number)
        return {"code": self._code, "files": ["app.py"], "file_contents": {}, "line_map": []}

    def get_pull_request_refs(self, owner, repo, pr_number) -> dict:
        return {"head_sha": "head-sha-abc", "base_sha": "base-sha"}

    def get_repo_code_corpus(self, owner, repo, ref, prefer_paths=None) -> list[str]:
        return []

    def upsert_comment(self, owner, repo, pr_number, body) -> bool:
        if not self._writable:
            return False
        self.posted.append((owner, repo, pr_number, body))
        return True

    def create_check_run(self, owner, repo, head_sha, *, conclusion, title, summary,
                         text=None, annotations=None) -> bool:
        if not self._writable:
            return False
        self.check_runs.append({"conclusion": conclusion, "title": title})
        return True


class _UnfetchableGitHub(_FakeGitHub):
    """A PR whose files cannot be read — the 'review skipped' path."""

    def get_pull_request_data(self, owner, repo, pr_number):
        raise RuntimeError("GitHub is unreachable")


class _NoLLM:
    def reset_budget(self) -> None:
        pass

    def generate_pr_audit(self, code, findings):
        return {}


def _event_file(workdir: pathlib.Path, fork: bool = False) -> str:
    head_repo = "contributor/hello" if fork else "octo/hello"
    payload = {
        "action": "opened",
        "pull_request": {
            "number": 7,
            "head": {"sha": "head-sha-abc", "repo": {"full_name": head_repo}},
            "base": {"repo": {"full_name": "octo/hello"}},
        },
        "repository": {"full_name": "octo/hello"},
    }
    path = workdir / "event.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _env(workdir: pathlib.Path, fork: bool = False, **overrides) -> dict:
    env = {
        "GITHUB_TOKEN": "ghs_test",
        "GITHUB_REPOSITORY": "octo/hello",
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_EVENT_PATH": _event_file(workdir, fork=fork),
        "GITHUB_API_URL": "https://api.github.com",
        "GITHUB_STEP_SUMMARY": str(workdir / "summary.md"),
        "GITHUB_OUTPUT": str(workdir / "output.txt"),
    }
    env.update(overrides)
    return env


@pytest.fixture(autouse=True)
def _inert_llm(monkeypatch):
    """No test here may reach a real provider, whatever sits in a local .env."""
    monkeypatch.setenv("ENABLE_LLM", "false")
    monkeypatch.setenv("ENABLE_CHECKS", "true")
    monkeypatch.setattr(runner, "_build_llm_service", lambda settings: _NoLLM())


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


# --- the happy path ----------------------------------------------------------


def test_action_reviews_the_pr_and_posts_a_comment_and_check_run(monkeypatch, workdir):
    fake = _FakeGitHub()
    monkeypatch.setattr(runner, "_build_github_client", lambda context: fake)

    exit_code = runner.main([], _env(workdir))

    assert fake.fetched_with == ("octo", "hello", 7)
    assert len(fake.posted) == 1
    assert len(fake.check_runs) == 1
    # Hardcoded credentials are HIGH, and `high` is the default threshold.
    assert exit_code == runner.EXIT_RISK


def test_a_clean_pull_request_exits_zero(monkeypatch, workdir):
    monkeypatch.setattr(runner, "_build_github_client", lambda context: _FakeGitHub(BENIGN_CODE))
    assert runner.main([], _env(workdir)) == runner.EXIT_OK


def test_the_report_is_written_to_the_step_summary(monkeypatch, workdir):
    monkeypatch.setattr(runner, "_build_github_client", lambda context: _FakeGitHub())

    runner.main([], _env(workdir))

    summary = _read(workdir / "summary.md")
    assert "Risk Score" in summary
    assert runner.FORK_NOTE not in summary


def test_outputs_are_exposed_to_later_steps(monkeypatch, workdir):
    monkeypatch.setattr(runner, "_build_github_client", lambda context: _FakeGitHub())

    runner.main([], _env(workdir))

    output = _read(workdir / "output.txt")
    assert "severity=HIGH" in output
    assert "skipped=false" in output


# --- fork pull requests ------------------------------------------------------


def test_a_fork_pr_still_delivers_the_review_when_posting_is_refused(monkeypatch, workdir):
    """The token is read-only on a fork, so the summary is the only channel.

    The review itself is perfectly valid; only the posting is refused. Losing
    it silently is the failure this guards against.
    """
    fake = _FakeGitHub(writable=False)
    monkeypatch.setattr(runner, "_build_github_client", lambda context: fake)

    exit_code = runner.main([], _env(workdir, fork=True))

    assert fake.posted == []
    summary = _read(workdir / "summary.md")
    assert runner.FORK_NOTE in summary
    assert "Risk Score" in summary
    # Posting failed, but the verdict still stands and still fails the job.
    assert exit_code == runner.EXIT_RISK


# --- failure handling --------------------------------------------------------


def test_a_failed_fetch_exits_two_and_posts_no_review(monkeypatch, workdir):
    fake = _UnfetchableGitHub()
    monkeypatch.setattr(runner, "_build_github_client", lambda context: fake)

    exit_code = runner.main([], _env(workdir))

    assert fake.posted == []
    assert exit_code == runner.EXIT_ERROR


def test_no_fail_on_error_softens_a_failed_fetch(monkeypatch, workdir):
    monkeypatch.setattr(runner, "_build_github_client", lambda context: _UnfetchableGitHub())
    assert runner.main(["--no-fail-on-error"], _env(workdir)) == runner.EXIT_OK


def test_a_missing_pull_request_context_exits_two_without_calling_github(monkeypatch, capsys):
    def _explode(context):  # pragma: no cover - must never run
        raise AssertionError("GitHub must not be touched without a context")

    monkeypatch.setattr(runner, "_build_github_client", _explode)

    assert runner.main([], {}) == runner.EXIT_ERROR
    assert "GITHUB_TOKEN" in capsys.readouterr().err


def test_an_unwritable_summary_never_fails_the_review(monkeypatch, workdir):
    """Instrumentation must not break a review that already succeeded."""
    monkeypatch.setattr(runner, "_build_github_client", lambda context: _FakeGitHub(BENIGN_CODE))
    # A directory, so the append raises.
    env = _env(workdir, GITHUB_STEP_SUMMARY=str(workdir))

    assert runner.main([], env) == runner.EXIT_OK


def test_absent_command_files_are_not_an_error(monkeypatch, workdir):
    monkeypatch.setattr(runner, "_build_github_client", lambda context: _FakeGitHub(BENIGN_CODE))
    env = _env(workdir)
    del env["GITHUB_STEP_SUMMARY"]
    del env["GITHUB_OUTPUT"]

    assert runner.main([], env) == runner.EXIT_OK


# --- the fail-on threshold, as pure functions --------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("critical", 3),
        ("high", 2),
        ("HIGH", 2),
        ("medium", 1),
        ("low", 0),
        ("never", runner._NEVER_RANK),
        ("off", runner._NEVER_RANK),
    ],
)
def test_fail_on_values_map_to_ranks(value, expected):
    assert runner._fail_on_rank(value) == expected


@pytest.mark.parametrize("value", ["", "   ", "banana", None])
def test_an_unknown_fail_on_value_falls_back_to_high(value):
    """A typo must not silently disable the gate."""
    assert runner._fail_on_rank(value) == runner._fail_on_rank("high")


@pytest.mark.parametrize(
    ("severity", "fail_on", "expected"),
    [
        (SeverityLevel.CRITICAL, "high", runner.EXIT_RISK),
        (SeverityLevel.HIGH, "high", runner.EXIT_RISK),
        (SeverityLevel.MEDIUM, "high", runner.EXIT_OK),
        (SeverityLevel.MEDIUM, "medium", runner.EXIT_RISK),
        (SeverityLevel.LOW, "low", runner.EXIT_RISK),
        (SeverityLevel.CRITICAL, "never", runner.EXIT_OK),
    ],
)
def test_severity_decides_the_exit_code(severity, fail_on, expected):
    outcome = ReviewOutcome(severity=severity, report="...")
    assert runner._exit_code(
        outcome, runner._fail_on_rank(fail_on), fail_on_error=True
    ) == expected


def test_a_skipped_review_is_an_error_not_a_pass():
    """'Could not look' must never read the same as 'looked and found nothing'."""
    skipped = ReviewOutcome()
    assert skipped.skipped is True
    assert runner._exit_code(skipped, runner._fail_on_rank("never"), fail_on_error=True) == (
        runner.EXIT_ERROR
    )


# --- the client builder ------------------------------------------------------


def _context(**overrides) -> ActionContext:
    base = {
        "owner": "octo",
        "repo": "hello",
        "pr_number": 7,
        "token": "ghs_test",
        "api_base_url": "https://ghe.corp/api/v3",
        "head_sha": "head-sha-abc",
        "event_name": "pull_request",
        "is_fork": False,
    }
    base.update(overrides)
    return ActionContext(**base)


def test_the_client_carries_the_token_and_api_url_and_no_app_credentials():
    client = runner._build_github_client(_context())

    assert client.access_token == "ghs_test"
    assert client.api_base_url == "https://ghe.corp/api/v3"
    assert (client.app_id, client.installation_id, client.private_key) == (None, None, None)


def test_the_client_ignores_enable_github(monkeypatch):
    """Being handed a token IS the enablement signal on a runner.

    This repository's own CI sets ENABLE_GITHUB=false globally; gating here
    would disable the thing under test and the suite would still pass.
    """
    monkeypatch.setenv("ENABLE_GITHUB", "false")
    assert runner._build_github_client(_context()) is not None


def test_the_job_carries_the_head_sha_from_the_event():
    """So a check run attaches to the PR's commit, not the merge commit."""
    assert runner._build_job(_context())["head_sha"] == "head-sha-abc"


def test_the_job_omits_an_absent_head_sha():
    assert "head_sha" not in runner._build_job(_context(head_sha=None))


# --- the check run must agree with the job -----------------------------------


def test_the_threshold_reaches_the_check_run(monkeypatch, workdir):
    """fail-on: never means a green job, so the check must not be red."""
    fake = _FakeGitHub()
    monkeypatch.setattr(runner, "_build_github_client", lambda context: fake)

    exit_code = runner.main(["--fail-on", "never"], _env(workdir))

    assert exit_code == runner.EXIT_OK
    assert fake.check_runs[0]["conclusion"] == "neutral"


def test_a_met_threshold_still_fails_the_check(monkeypatch, workdir):
    fake = _FakeGitHub()
    monkeypatch.setattr(runner, "_build_github_client", lambda context: fake)

    exit_code = runner.main(["--fail-on", "high"], _env(workdir))

    assert exit_code == runner.EXIT_RISK
    assert fake.check_runs[0]["conclusion"] == "failure"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("never", "NEVER"), ("off", "NEVER"), ("high", "HIGH"),
     ("CRITICAL", "CRITICAL"), ("banana", "HIGH")],
)
def test_the_threshold_is_canonicalised_before_it_leaves_the_action(value, expected):
    """So the orchestrator never has to know this module's input vocabulary."""
    assert runner._canonical_fail_on(value) == expected


def test_no_fail_on_error_also_covers_a_broken_context():
    """action.yml documents fail-on-error as covering "no PR context" too.

    It only honoured the fetch-failed half: a context error returned EXIT_ERROR
    before the flag was ever consulted, so the workflow most in need of the
    escape hatch — one whose triggers do not all carry a pull request — still
    went red.
    """
    env = {"GITHUB_REPOSITORY": "acme/widget"}  # no GITHUB_TOKEN

    assert runner.main(argv=[], env=env) == runner.EXIT_ERROR
    assert runner.main(argv=["--no-fail-on-error"], env=env) == runner.EXIT_OK


def test_no_fail_on_error_covers_every_context_error():
    for env in (
        {"GITHUB_REPOSITORY": "acme/widget"},            # no token
        {"GITHUB_TOKEN": "t"},                            # no repository
        {"GITHUB_TOKEN": "t", "GITHUB_REPOSITORY": "acme/widget"},  # no PR
    ):
        assert runner.main(argv=["--no-fail-on-error"], env=env) == runner.EXIT_OK
        assert runner.main(argv=[], env=env) == runner.EXIT_ERROR
