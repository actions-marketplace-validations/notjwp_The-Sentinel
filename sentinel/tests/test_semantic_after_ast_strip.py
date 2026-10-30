"""The AST security pass must not starve the duplicate detector.

``_strip_ast_covered_lines`` removes every ``.py`` file the AST engine could
read from the regex blob — and that blob was also what ``_assess`` handed the
semantic engine. So on a Python pull request, which is the only language the
corpus is built from, duplicate detection silently never ran: the identical
copied function scored LOW as ``auth.py`` and HIGH as ``auth.js``.

The fix threads the pre-strip blob separately, following the same rule as
``extra_security_findings`` and ``extra_debt_complexity`` — only the caller
knows which input each engine should see.
"""

from sentinel.application.risk_engine import RiskEngine
from sentinel.domain.services.semantic_service import SemanticService
from sentinel.domain.value_objects.severity_level import SeverityLevel
from sentinel.infrastructure.semantic.embedding_engine import EmbeddingEngine
from sentinel.workers.background_worker import BackgroundWorker

CORPUS_UNIT = """def check_account_credentials(username, password, store):
    if not username or not password:
        return False
    record = store.get(username)
    if record is None:
        return False
    return record.get("password") == password
"""

# The same function, copied into the pull request under a new name. Nothing
# else in it: no secret, no eval, nothing the security engines would report.
COPIED = CORPUS_UNIT.replace("check_account_credentials", "validate_user_credentials")


def _engine() -> RiskEngine:
    return RiskEngine(semantic_service=SemanticService(EmbeddingEngine()))


def _job(path: str) -> dict:
    """The job ``_fetch_pr_data`` builds for a PR that adds one file whole."""
    lines = COPIED.split("\n")
    return {
        "repo": "test-repo",
        "pr_number": 1,
        "code": COPIED,
        "line_map": [(path, n) for n in range(1, len(lines) + 1)],
        "files": [path],
        "file_contents": {path: COPIED},
        "existing_code_list": [CORPUS_UNIT],
    }


def _run_pipeline(job: dict) -> dict:
    """The exact order ``run_review`` uses, minus the network."""
    ast_findings = BackgroundWorker._ast_security_findings(job)
    ast_complexity = BackgroundWorker._ast_complexity(job)
    job["semantic_code"] = job["code"]
    BackgroundWorker._strip_ast_covered_lines(job)
    _, assessment = BackgroundWorker._assess(job, _engine(), ast_findings, ast_complexity)
    return assessment


def test_duplicate_in_a_python_file_survives_the_ast_strip():
    """The regression. A .py file is exactly what the AST engine claims."""
    job = _job("auth.py")

    assessment = _run_pipeline(job)

    assert job["ast_covered_files"] == ["auth.py"], "precondition: the strip must apply"
    assert job["code"].strip() == "", "precondition: the blob must be emptied"
    assert assessment["semantic_findings_count"] == 1
    assert assessment["severity"] == SeverityLevel.HIGH


def test_duplicate_in_a_non_python_file_is_unaffected():
    """The case that always worked; it must keep working."""
    assessment = _run_pipeline(_job("auth.js"))

    assert assessment["semantic_findings_count"] == 1
    assert assessment["severity"] == SeverityLevel.HIGH


def test_the_two_languages_now_agree():
    """The asymmetry itself is the bug, so pin its absence directly."""
    python = _run_pipeline(_job("auth.py"))
    javascript = _run_pipeline(_job("auth.js"))

    assert python["semantic_findings_count"] == javascript["semantic_findings_count"]
    assert python["severity"] == javascript["severity"]


def test_the_strip_still_removes_the_file_from_the_regex_blob():
    """Guard against 'fixing' this by simply not stripping.

    The strip is what stops one line being reported by both security engines.
    Feeding the semantic engine must not cost that.
    """
    job = _job("auth.py")

    _run_pipeline(job)

    assert job["code"].strip() == ""
    assert job["line_map"] == []
    assert job["semantic_code"].strip() != ""


def test_semantic_code_feeds_only_the_duplicate_pass():
    """``code`` still drives security and debt; only the semantic engine differs."""
    engine = _engine()

    assessment = engine.assess_resilient(
        code='api_key = "literal-value-here"',
        existing_code_list=[CORPUS_UNIT],
        semantic_code=COPIED,
    )

    # The duplicate came from semantic_code...
    assert assessment["semantic_findings_count"] == 1
    # ...while the secret came from code, which the semantic engine never saw.
    assert any(f.rule == "api_key_assignment" for f in assessment["security"]["findings"])


def test_semantic_code_defaults_to_code_when_absent():
    """Every existing caller passes nothing and must behave exactly as before."""
    engine = _engine()

    assessment = engine.assess_resilient(code=COPIED, existing_code_list=[CORPUS_UNIT])

    assert assessment["semantic_findings_count"] == 1
