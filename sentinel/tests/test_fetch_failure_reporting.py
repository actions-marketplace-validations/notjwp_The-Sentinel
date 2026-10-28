"""A review that could not read the pull request must not report a clean one.

The gap these cover: ``GitHubClient`` is failure-safe and turns HTTP errors
into empty results rather than exceptions, so the worker's "fetch failed"
branch — which only triggered on an exception — never fired for the most
common real failure. A 500 on the files endpoint produced "Risk: LOW, no
security issues detected" and a passing check run.
"""

import sentinel.workers.background_worker as bw_module
from sentinel.application.audit_orchestrator import AuditOrchestrator
from sentinel.application.risk_engine import RiskEngine
from sentinel.domain.services.document_service import DocumentService
from sentinel.workers.background_worker import BackgroundWorker, run_review


class _ClientThatCannotList:
    """What the real client does when GitHub answers 500: empty, not raising."""

    def __init__(self):
        self.checks = []
        self.comments = []

    def get_pull_request_refs(self, owner, repo, pr_number):
        return {"head_sha": "a" * 40, "base_sha": "b" * 40}

    def get_pull_request_data(self, owner, repo, pr_number):
        return None  # could not list the files

    def get_repo_code_corpus(self, *a, **k):
        return []

    def upsert_comment(self, owner, repo, pr_number, body):
        self.comments.append(body)
        return True

    def create_check_run(self, owner, repo, head_sha, **kwargs):
        self.checks.append(kwargs)
        return True


class _ClientWithAnEmptyPR(_ClientThatCannotList):
    """A real pull request that happens to change nothing — still reviewable."""

    def get_pull_request_data(self, owner, repo, pr_number):
        return {"code": "", "files": [], "file_contents": {}, "line_map": []}


def _job():
    return {"owner": "acme", "repo": "widget", "pr_number": 42}


def _orchestrator():
    return AuditOrchestrator(None, document_service=DocumentService())


def test_fetch_data_returns_false_when_the_listing_failed():
    job = _job()
    assert BackgroundWorker._fetch_pr_data(job, _ClientThatCannotList()) is False


def test_fetch_data_returns_true_for_a_genuinely_empty_pr():
    job = _job()
    assert BackgroundWorker._fetch_pr_data(job, _ClientWithAnEmptyPR()) is True


def test_failed_listing_skips_the_review_and_posts_no_clean_comment(monkeypatch):
    monkeypatch.setattr(bw_module, "_build_llm_service", lambda settings: None)
    client = _ClientThatCannotList()
    outcome = run_review(_job(), RiskEngine(), _orchestrator(), client)

    assert outcome.skipped is True
    assert outcome.severity is None
    assert client.comments == [], "a failed fetch must not comment at all"
    assert len(client.checks) == 1
    assert client.checks[0]["conclusion"] == "neutral"
    assert "skipped" in client.checks[0]["title"].lower()


def test_empty_pr_still_gets_a_real_review(monkeypatch):
    monkeypatch.setattr(bw_module, "_build_llm_service", lambda settings: None)
    client = _ClientWithAnEmptyPR()
    outcome = run_review(_job(), RiskEngine(), _orchestrator(), client)

    assert outcome.skipped is False
    assert outcome.severity is not None
    assert client.comments, "a docs-only PR is a real review and should be posted"
