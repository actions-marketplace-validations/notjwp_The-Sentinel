"""M1: the async worker fetches real PR code from GitHub and posts a review.

Drives a real ``BackgroundWorker.start()`` iteration with a fake GitHub client
injected via ``bw_module._build_github_client`` (the module-level seam), proving
the full loop: fetch diff -> assess -> build_report -> post_comment.
"""

import asyncio

import sentinel.workers.background_worker as bw_module
from sentinel.workers.background_worker import BackgroundWorker
from sentinel.workers.job_queue import JobQueue


class _FakeGitHub:
    def __init__(
        self,
        code: str,
        files: list[str] | None = None,
        file_contents: dict[str, str] | None = None,
        corpus: list[str] | None = None,
        line_map: list | None = None,
    ) -> None:
        self._code = code
        self._files = files or []
        self._file_contents = file_contents or {}
        self._corpus = corpus or []
        self._line_map = line_map or []
        self.fetched_with: tuple | None = None
        self.corpus_ref: str | None = None
        self.posted: list[tuple] = []
        self.check_runs: list[dict] = []

    def get_pull_request_data(self, owner: str, repo: str, pr_number) -> dict:
        self.fetched_with = (owner, repo, pr_number)
        return {
            "code": self._code,
            "files": self._files,
            "file_contents": self._file_contents,
            "line_map": self._line_map,
        }

    def create_check_run(
        self, owner: str, repo: str, head_sha: str, *, conclusion, title, summary,
        text=None, annotations=None,
    ) -> bool:
        self.check_runs.append(
            {
                "owner": owner,
                "repo": repo,
                "head_sha": head_sha,
                "conclusion": conclusion,
                "title": title,
                "summary": summary,
                "text": text,
                "annotations": annotations or [],
            }
        )
        return True

    def get_pull_request_refs(self, owner: str, repo: str, pr_number) -> dict:
        return {"head_sha": "head-sha", "base_sha": "base-sha"}

    def get_repo_code_corpus(self, owner: str, repo: str, ref: str, prefer_paths=None) -> list[str]:
        self.corpus_ref = ref
        self.corpus_prefer_paths = prefer_paths
        return self._corpus

    def post_comment(self, owner: str, repo: str, pr_number, body: str) -> bool:
        self.posted.append((owner, repo, pr_number, body))
        return True

    def upsert_comment(self, owner: str, repo: str, pr_number, body: str) -> bool:
        # The worker posts via the idempotent upsert; record identically to post.
        self.posted.append((owner, repo, pr_number, body))
        return True


class _FakeLLM:
    """A stand-in LLM service that tags every finding with a distinctive explanation.

    Proves the worker actually routes findings through enrich_findings_with_llm before
    building the report (the enrichment matches by id(finding)).
    """

    ENRICHED_MARK = "SENTINEL_ENRICHED_EXPLANATION"

    def reset_budget(self) -> None:
        pass

    def generate_pr_audit(self, code, findings):
        return {
            id(finding): {"explanation": self.ENRICHED_MARK, "fix": "apply the fix"}
            for finding in findings
        }


def _drive_one_job(worker: BackgroundWorker, queue: JobQueue, fake: _FakeGitHub) -> None:
    real_sleep = asyncio.sleep

    async def fast_sleep(_: float) -> None:
        await real_sleep(0)

    async def _run() -> None:
        original_sleep = bw_module.asyncio.sleep
        bw_module.asyncio.sleep = fast_sleep

        task = asyncio.create_task(worker.start())
        for _ in range(500):
            if fake.posted:
                break
            await real_sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        bw_module.asyncio.sleep = original_sleep

    asyncio.run(_run())


def test_worker_fetches_pr_code_and_posts_structured_review(monkeypatch, capsys):
    # This test asserts the enrichment marker reaches the report, and
    # enrich_findings_with_llm gates on ENABLE_LLM — so the flag has to be on for
    # the assertion to mean anything. CI sets ENABLE_LLM=false globally, which is
    # why this needs to be explicit rather than inherited from the environment.
    # No network: _build_llm_service is replaced with a fake below.
    monkeypatch.setenv("ENABLE_LLM", "true")

    # Vulnerable code the worker will only ever see by fetching it from GitHub.
    fake = _FakeGitHub('password = "hunter2"\napi_key = "sk-abcdefghijklmnopqrst"')
    monkeypatch.setattr(bw_module, "_build_github_client", lambda settings, installation_id=None: fake)
    monkeypatch.setattr(bw_module, "_build_llm_service", lambda settings: _FakeLLM())

    async def _seed(queue: JobQueue) -> None:
        await queue.enqueue({"owner": "octo", "repo": "hello", "pr_number": 7})

    queue = JobQueue()
    asyncio.run(_seed(queue))
    worker = BackgroundWorker(queue)

    _drive_one_job(worker, queue, fake)

    # Fetched using the job's identity (bare repo name).
    assert fake.fetched_with == ("octo", "hello", 7)

    # Posted exactly one structured review to the right PR.
    assert len(fake.posted) == 1
    owner, repo, pr_number, body = fake.posted[0]
    assert (owner, repo, pr_number) == ("octo", "hello", 7)
    assert "# Sentinel AI Code Review" in body
    assert "## Risk Score: HIGH" in body  # hardcoded secrets are HIGH severity
    assert "## Security Issues" in body
    # Proof the fetched findings were routed through LLM enrichment before the report
    # was built (the '## Explanation' section now carries the enriched text).
    assert _FakeLLM.ENRICHED_MARK in body

    # The one-liner still prints, now reflecting the fetched code (not empty -> LOW).
    assert "PR #7 Risk: HIGH" in capsys.readouterr().out


def test_worker_splits_owner_repo_from_full_name(monkeypatch, capsys):
    """When repo arrives as 'owner/name', the worker calls GitHub with the bare name."""
    fake = _FakeGitHub("x = 1")
    monkeypatch.setattr(bw_module, "_build_github_client", lambda settings, installation_id=None: fake)

    async def _seed(queue: JobQueue) -> None:
        # owner separate, repo carries the full "owner/name" form.
        await queue.enqueue({"owner": "octo", "repo": "octo/hello", "pr_number": 3})

    queue = JobQueue()
    asyncio.run(_seed(queue))
    worker = BackgroundWorker(queue)

    _drive_one_job(worker, queue, fake)

    assert fake.fetched_with == ("octo", "hello", 3)
    assert fake.posted and fake.posted[0][:3] == ("octo", "hello", 3)
    assert "PR #3 Risk:" in capsys.readouterr().out


def test_worker_skips_github_when_no_owner(monkeypatch, capsys):
    """Owner-less (flat/manual) jobs neither fetch nor post — no regression."""
    fake = _FakeGitHub('password = "leak"')
    monkeypatch.setattr(bw_module, "_build_github_client", lambda settings, installation_id=None: fake)

    real_sleep = asyncio.sleep

    async def fast_sleep(_: float) -> None:
        await real_sleep(0)

    async def _run() -> None:
        queue = JobQueue()
        await queue.enqueue({"repo": "hello", "pr_number": 9})  # no owner
        worker = BackgroundWorker(queue)

        original_sleep = bw_module.asyncio.sleep
        bw_module.asyncio.sleep = fast_sleep
        task = asyncio.create_task(worker.start())
        for _ in range(500):
            if worker.processed_count >= 1:
                break
            await real_sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        bw_module.asyncio.sleep = original_sleep

    asyncio.run(_run())

    assert fake.fetched_with is None
    assert fake.posted == []
    # Still analyzed (empty code -> LOW) and printed the one-liner.
    assert "PR #9 Risk: LOW" in capsys.readouterr().out


def test_worker_review_includes_documentation_findings(monkeypatch, capsys):
    """M3 parity: async reviews now carry doc findings, like the sync path always has."""
    fake = _FakeGitHub(
        "x = 1",
        files=["README.md"],
        file_contents={"README.md": "notes only"},  # lacks install + usage guidance
    )
    monkeypatch.setattr(bw_module, "_build_github_client", lambda settings, installation_id=None: fake)
    monkeypatch.setattr(bw_module, "_build_llm_service", lambda settings: _FakeLLM())

    async def _seed(queue: JobQueue) -> None:
        await queue.enqueue({"owner": "octo", "repo": "hello", "pr_number": 11})

    queue = JobQueue()
    asyncio.run(_seed(queue))
    worker = BackgroundWorker(queue)

    _drive_one_job(worker, queue, fake)

    assert len(fake.posted) == 1
    body = fake.posted[0][3]
    assert "## Documentation Issues" in body
    assert "missing installation instructions" in body
    assert "missing usage guidance" in body
    assert "PR #11 Risk:" in capsys.readouterr().out


def test_worker_semantic_corpus_flags_duplicate_code(monkeypatch, capsys):
    """M5: the worker builds a real corpus from the base ref and flags duplication.

    The PR code is benign (no secrets), so the HIGH risk can only come from the
    semantic engine matching it against a corpus chunk.
    """
    pr_code = (
        "def compute_total(values):\n"
        "    total = 0\n"
        "    for value in values:\n"
        "        total = total + value\n"
        "    return total"
    )
    corpus_file = (
        "import math\n"
        "\n" + pr_code + "\n\n"
        "def unrelated_parser(text, sep, limit, flags):\n"
        "    return text.split(sep, limit)\n"
    )
    fake = _FakeGitHub(pr_code, corpus=[corpus_file])
    monkeypatch.setattr(bw_module, "_build_github_client", lambda settings, installation_id=None: fake)
    monkeypatch.setattr(bw_module, "_build_llm_service", lambda settings: _FakeLLM())

    async def _seed(queue: JobQueue) -> None:
        await queue.enqueue({"owner": "octo", "repo": "hello", "pr_number": 12})

    queue = JobQueue()
    asyncio.run(_seed(queue))
    worker = BackgroundWorker(queue)

    _drive_one_job(worker, queue, fake)

    assert fake.corpus_ref == "base-sha"  # corpus is built from the PR's BASE ref
    assert len(fake.posted) == 1
    body = fake.posted[0][3]
    assert "Similar findings detected: 1" in body
    assert "- No security issues detected." in body
    assert "## Risk Score: HIGH" in body  # driven purely by the semantic duplicate
    assert "PR #12 Risk: HIGH" in capsys.readouterr().out


def test_worker_posts_check_run_with_line_mapped_annotations(monkeypatch, capsys):
    """M6: alongside the comment, a check run lands with head-line annotations.

    The line map says the two added lines live at app.py lines 14-15 (a mid-file
    patch), so annotations must point THERE, not at code-blob lines 1-2.
    """
    code = 'password = "hunter2"\napi_key = "sk-abcdefghijklmnopqrst"'
    fake = _FakeGitHub(code, line_map=[("app.py", 14), ("app.py", 15)])
    monkeypatch.setattr(bw_module, "_build_github_client", lambda settings, installation_id=None: fake)
    monkeypatch.setattr(bw_module, "_build_llm_service", lambda settings: _FakeLLM())

    async def _seed(queue: JobQueue) -> None:
        await queue.enqueue({"owner": "octo", "repo": "hello", "pr_number": 13})

    queue = JobQueue()
    asyncio.run(_seed(queue))
    worker = BackgroundWorker(queue)

    _drive_one_job(worker, queue, fake)

    assert len(fake.posted) == 1  # the comment still lands
    assert len(fake.check_runs) == 1
    run = fake.check_runs[0]
    assert (run["owner"], run["repo"], run["head_sha"]) == ("octo", "hello", "head-sha")
    assert run["conclusion"] == "failure"  # HIGH risk -> failure
    assert "Risk: HIGH" in run["title"]
    assert "# Sentinel AI Code Review" in run["text"]  # full report attached

    security_annotations = [
        a for a in run["annotations"] if a["annotation_level"] == "failure"
    ]
    lines_hit = {(a["path"], a["start_line"]) for a in security_annotations}
    assert lines_hit == {("app.py", 14), ("app.py", 15)}
    assert all(a["start_line"] == a["end_line"] for a in security_annotations)
    assert "PR #13 Risk: HIGH" in capsys.readouterr().out


def test_worker_records_metrics(monkeypatch, capsys):
    """M7: one processed job populates the whole counter/summary set."""
    from sentinel.monitoring.metrics import metrics

    metrics.reset()
    fake = _FakeGitHub('password = "hunter2"')
    monkeypatch.setattr(bw_module, "_build_github_client", lambda settings, installation_id=None: fake)
    monkeypatch.setattr(bw_module, "_build_llm_service", lambda settings: _FakeLLM())

    real_sleep = asyncio.sleep

    async def fast_sleep(_: float) -> None:
        await real_sleep(0)

    async def _run() -> None:
        queue = JobQueue()
        await queue.enqueue({"owner": "octo", "repo": "hello", "pr_number": 15})
        worker = BackgroundWorker(queue)
        original_sleep = bw_module.asyncio.sleep
        bw_module.asyncio.sleep = fast_sleep
        task = asyncio.create_task(worker.start())
        for _ in range(500):
            if worker.processed_count >= 1:
                break
            await real_sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        bw_module.asyncio.sleep = original_sleep

    asyncio.run(_run())

    snap = metrics.snapshot()
    assert snap["counters"]["sentinel_jobs_processed_total"] == 1
    assert snap["counters"]['sentinel_reviews_total{severity="HIGH"}'] == 1
    assert snap["counters"]['sentinel_github_posts_total{kind="comment",outcome="ok"}'] == 1
    assert snap["counters"]['sentinel_github_posts_total{kind="check_run",outcome="ok"}'] == 1
    assert snap["summaries"]["sentinel_job_duration_seconds"]["count"] == 1
    capsys.readouterr()


def test_worker_skips_check_run_when_flag_off(monkeypatch, capsys):
    """ENABLE_CHECKS=false: the comment posts, no check run is created."""
    monkeypatch.setenv("ENABLE_CHECKS", "false")
    fake = _FakeGitHub('password = "leak"')
    monkeypatch.setattr(bw_module, "_build_github_client", lambda settings, installation_id=None: fake)
    monkeypatch.setattr(bw_module, "_build_llm_service", lambda settings: _FakeLLM())

    async def _seed(queue: JobQueue) -> None:
        await queue.enqueue({"owner": "octo", "repo": "hello", "pr_number": 14})

    queue = JobQueue()
    asyncio.run(_seed(queue))
    worker = BackgroundWorker(queue)

    _drive_one_job(worker, queue, fake)

    assert len(fake.posted) == 1
    assert fake.check_runs == []
    assert "PR #14 Risk:" in capsys.readouterr().out


def test_assess_skips_semantic_when_the_job_carries_no_corpus():
    """M8: no corpus -> no duplicate detection, instead of a placeholder comparison.

    This code is verbatim the old two-function fallback corpus, so under the
    previous behavior it scored a perfect self-match: semantic HIGH, and a
    failing check run on a PR that duplicated nothing real.
    """
    from sentinel.application.risk_engine import RiskEngine
    from sentinel.domain.services.semantic_service import SemanticService
    from sentinel.domain.value_objects.severity_level import SeverityLevel
    from sentinel.infrastructure.semantic.embedding_engine import EmbeddingEngine

    engine = RiskEngine(semantic_service=SemanticService(EmbeddingEngine()))
    job = {"repo": "hello", "pr_number": 1, "code": "def add(a, b): return a + b"}

    _, assessment = BackgroundWorker._assess(job, engine)

    assert assessment["semantic_findings_count"] == 0
    assert assessment["severity"] == SeverityLevel.LOW


def test_assess_still_detects_duplicates_against_a_real_corpus():
    """The corpus path is untouched — only the placeholder fallback is gone."""
    from sentinel.application.risk_engine import RiskEngine
    from sentinel.domain.services.semantic_service import SemanticService
    from sentinel.domain.value_objects.severity_level import SeverityLevel
    from sentinel.infrastructure.semantic.embedding_engine import EmbeddingEngine

    duplicated = "def add(a, b): return a + b"
    engine = RiskEngine(semantic_service=SemanticService(EmbeddingEngine()))
    job = {
        "repo": "hello",
        "pr_number": 2,
        "code": duplicated,
        "existing_code_list": [duplicated],
    }

    _, assessment = BackgroundWorker._assess(job, engine)

    assert assessment["semantic_findings_count"] == 1
    assert assessment["severity"] == SeverityLevel.HIGH


class _FailingGitHub(_FakeGitHub):
    """A client whose PR-file fetch is down; refs (and so head_sha) still work."""

    def get_pull_request_data(self, owner: str, repo: str, pr_number) -> dict:
        raise RuntimeError("GitHub API unavailable")


def _drive_until(worker: BackgroundWorker, predicate) -> None:
    real_sleep = asyncio.sleep

    async def fast_sleep(_: float) -> None:
        await real_sleep(0)

    async def _run() -> None:
        original_sleep = bw_module.asyncio.sleep
        bw_module.asyncio.sleep = fast_sleep

        task = asyncio.create_task(worker.start())
        for _ in range(500):
            if predicate():
                break
            await real_sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        bw_module.asyncio.sleep = original_sleep

    asyncio.run(_run())


def test_failed_fetch_posts_a_neutral_check_and_no_review(monkeypatch, capsys):
    """M8: a GitHub outage must not read as a clean PR.

    Before this, the worker assessed the empty code it failed to fetch and
    posted 'No security issues detected' plus a passing check run.
    """
    from sentinel.monitoring.metrics import metrics

    metrics.reset()
    fake = _FailingGitHub("never returned")
    monkeypatch.setattr(bw_module, "_build_github_client", lambda settings, installation_id=None: fake)
    monkeypatch.setattr(bw_module, "_build_llm_service", lambda settings: _FakeLLM())

    async def _seed(queue: JobQueue) -> None:
        await queue.enqueue({"owner": "octo", "repo": "hello", "pr_number": 20})

    queue = JobQueue()
    asyncio.run(_seed(queue))
    worker = BackgroundWorker(queue)

    _drive_until(worker, lambda: worker.processed_count >= 1)

    # No review comment at all — a broad outage must not comment on every PR.
    assert fake.posted == []

    # One neutral check run, carrying the head SHA the refs call stashed BEFORE
    # the files call failed.
    assert len(fake.check_runs) == 1
    run = fake.check_runs[0]
    assert run["conclusion"] == "neutral"  # visible, but never blocks a merge
    assert run["head_sha"] == "head-sha"
    assert run["title"] == "Review skipped"
    assert "could not fetch" in run["summary"]

    counters = metrics.snapshot()["counters"]
    assert counters['sentinel_reviews_skipped_total{reason="fetch_failed"}'] == 1
    assert counters['sentinel_github_posts_total{kind="check_run",outcome="skipped"}'] == 1
    # Crucially: no review was recorded at any severity.
    assert not any(key.startswith("sentinel_reviews_total") for key in counters)

    assert "PR #20 Risk: SKIPPED (fetch failed)" in capsys.readouterr().out


def test_docs_only_pr_is_still_reviewed(monkeypatch, capsys):
    """Empty code is NOT a fetch failure: a docs-only PR still gets a full review."""
    from sentinel.monitoring.metrics import metrics

    metrics.reset()
    fake = _FakeGitHub(
        "",  # no added code lines at all — only a markdown file changed
        files=["README.md"],
        file_contents={"README.md": "notes only"},
    )
    monkeypatch.setattr(bw_module, "_build_github_client", lambda settings, installation_id=None: fake)
    monkeypatch.setattr(bw_module, "_build_llm_service", lambda settings: _FakeLLM())

    async def _seed(queue: JobQueue) -> None:
        await queue.enqueue({"owner": "octo", "repo": "hello", "pr_number": 21})

    queue = JobQueue()
    asyncio.run(_seed(queue))
    worker = BackgroundWorker(queue)

    _drive_one_job(worker, queue, fake)

    assert len(fake.posted) == 1
    body = fake.posted[0][3]
    assert "## Documentation Issues" in body

    assert len(fake.check_runs) == 1
    assert fake.check_runs[0]["conclusion"] == "success"  # a real result, not "skipped"

    counters = metrics.snapshot()["counters"]
    assert counters['sentinel_reviews_total{severity="LOW"}'] == 1
    assert not any(key.startswith("sentinel_reviews_skipped_total") for key in counters)

    assert "PR #21 Risk: LOW" in capsys.readouterr().out


# --- M11: AST security analysis takes over the files it can parse ---


_MODULE_WITH_SECRET = (
    '"""Service module."""\n'          # 1
    "import os\n"                      # 2
    "\n"                               # 3
    "\n"                               # 4
    "def connect(host):\n"             # 5
    '    password = "hunter2"\n'       # 6  <- planted, and added by the PR
    "    return (host, password)\n"    # 7
)


def test_ast_findings_carry_the_real_file_and_line():
    """The whole point: a finding located in the file, not in a diff blob."""
    job = {
        "file_contents": {"svc.py": _MODULE_WITH_SECRET},
        "line_map": [("svc.py", 6)],
        "code": '    password = "hunter2"',
    }
    findings = BackgroundWorker._ast_security_findings(job)

    assert [f.rule for f in findings] == ["hardcoded_secret"]
    assert findings[0].file == "svc.py"
    assert findings[0].line == 6
    assert job["ast_covered_files"] == ["svc.py"]


def test_ast_ignores_problems_the_pr_did_not_touch():
    """The analyzer sees the whole module; the review is only about this PR."""
    job = {
        "file_contents": {"svc.py": _MODULE_WITH_SECRET},
        "line_map": [("svc.py", 7)],  # the PR added line 7, not the secret on line 6
        "code": "    return (host, password)",
    }
    assert BackgroundWorker._ast_security_findings(job) == []


def test_ast_covered_files_are_withheld_from_the_regex_blob():
    """Two engines, one line each — never both on the same line."""
    job = {
        "file_contents": {"svc.py": _MODULE_WITH_SECRET},
        "line_map": [("svc.py", 6), ("other.txt", 3)],
        "code": '    password = "hunter2"\ntoken = "abc"',
        "ast_covered_files": ["svc.py"],
    }
    BackgroundWorker._strip_ast_covered_lines(job)

    assert job["code"] == 'token = "abc"'
    assert job["line_map"] == [("other.txt", 3)]


def test_unparseable_python_falls_back_to_the_regex_engine():
    """A file that does not parse must not be silently skipped by both engines."""
    job = {
        "file_contents": {"broken.py": "def oops(:\n    pass\n"},
        "line_map": [("broken.py", 1)],
        "code": "def oops(:",
    }
    assert BackgroundWorker._ast_security_findings(job) == []
    assert "ast_covered_files" not in job

    BackgroundWorker._strip_ast_covered_lines(job)
    assert job["code"] == "def oops(:", "the blob must be left for regex to handle"


def test_strip_is_a_no_op_when_blob_and_map_disagree():
    """Corrupting index-parallel structures is worse than leaving them alone."""
    job = {
        "code": "a\nb\nc",
        "line_map": [("x.py", 1)],
        "ast_covered_files": ["x.py"],
    }
    BackgroundWorker._strip_ast_covered_lines(job)
    assert job["code"] == "a\nb\nc"
    assert job["line_map"] == [("x.py", 1)]
