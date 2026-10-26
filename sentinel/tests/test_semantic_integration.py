from sentinel.application.risk_engine import RiskEngine
from sentinel.domain.entities.finding import Finding
from sentinel.domain.services.semantic_service import SemanticService
from sentinel.domain.value_objects.severity_level import SeverityLevel
from sentinel.infrastructure.semantic.embedding_engine import EmbeddingEngine


def _engine_with_semantic() -> RiskEngine:
    embedding_engine = EmbeddingEngine()
    semantic_service = SemanticService(embedding_engine)
    return RiskEngine(semantic_service=semantic_service)


def _engine_without_semantic() -> RiskEngine:
    return RiskEngine()


SIMPLE_CODE = "def simple():\n    return 1"
DUPLICATE_CODE = "def add(a, b): return a + b"
DUPLICATE_EXISTING = ["def add(a, b): return a + b"]
NON_DUPLICATE_EXISTING = [
    "def sort_list(lst):\n    for i in range(len(lst)):\n        for j in range(i+1, len(lst)):\n            if lst[i] > lst[j]:\n                lst[i], lst[j] = lst[j], lst[i]\n    return lst"
]

HIGH_COMPLEXITY_CODE = "\n".join(
    ["def extreme(value):"] + ["    if value > 0:"] * 15 + ["        return value"]
)

SECURITY_HIGH_CODE = 'def f():\n    api_key = "abc"\n    return 1'


# --- Semantic HIGH Overrides LOW Debt ---


def test_semantic_high_overrides_low_debt():
    engine = _engine_with_semantic()
    result = engine.assess(code=DUPLICATE_CODE, existing_code_list=DUPLICATE_EXISTING)
    assert result["severity"] == SeverityLevel.HIGH
    assert result["semantic"]["severity"] == SeverityLevel.HIGH
    assert result["semantic_findings_count"] >= 1


def test_semantic_high_overrides_medium_debt():
    medium_code = "def add(a, b):\n" + "    if a > 0:\n        a += 1\n" * 8 + "    return a + b"
    engine = _engine_with_semantic()
    existing = [medium_code]
    result = engine.assess(code=medium_code, existing_code_list=existing)
    assert result["severity"] == SeverityLevel.HIGH
    assert result["semantic"]["severity"] == SeverityLevel.HIGH


# --- Semantic LOW Does NOT Escalate ---


def test_semantic_low_does_not_escalate_low_debt():
    engine = _engine_with_semantic()
    result = engine.assess(code=SIMPLE_CODE, existing_code_list=NON_DUPLICATE_EXISTING)
    assert result["severity"] == SeverityLevel.LOW
    assert result["semantic"]["severity"] == SeverityLevel.LOW
    assert result["semantic_findings_count"] == 0


def test_semantic_low_preserves_medium_debt():
    medium_code = "def med(v):\n" + "    if v > 0:\n        v += 1\n" * 8 + "    return v"
    engine = _engine_with_semantic()
    result = engine.assess(code=medium_code, existing_code_list=NON_DUPLICATE_EXISTING)
    assert result["severity"] == SeverityLevel.MEDIUM
    assert result["semantic"]["severity"] == SeverityLevel.LOW


def test_semantic_low_leaves_high_debt_capped_at_medium():
    """Debt is capped on the way into the overall score; semantic LOW adds nothing."""
    engine = _engine_with_semantic()
    result = engine.assess(code=HIGH_COMPLEXITY_CODE, existing_code_list=NON_DUPLICATE_EXISTING)
    assert result["severity"] == SeverityLevel.MEDIUM
    assert result["semantic"]["severity"] == SeverityLevel.LOW


# --- No Semantic Duplicates ---


def test_no_semantic_duplicates_produces_no_findings():
    engine = _engine_with_semantic()
    result = engine.assess(code=SIMPLE_CODE, existing_code_list=NON_DUPLICATE_EXISTING)
    assert result["semantic"]["findings"] == []
    assert result["semantic_findings_count"] == 0


def test_no_semantic_service_produces_no_findings():
    engine = _engine_without_semantic()
    result = engine.assess(code=DUPLICATE_CODE)
    assert result["semantic"]["findings"] == []
    assert result["semantic_findings_count"] == 0
    assert result["semantic"]["severity"] == SeverityLevel.LOW


# --- Multiple Duplicates ---


def test_duplicate_count_is_per_pr_unit_not_per_corpus_match():
    """The count answers "how many things did this PR duplicate", not "how many
    corpus entries resemble it" — one copied function is one finding."""
    engine = _engine_with_semantic()
    existing = [
        "def add(a, b): return a + b",
        "def add(x, y): return x + y",
    ]
    result = engine.assess(code=DUPLICATE_CODE, existing_code_list=existing)
    assert result["semantic_findings_count"] == 1
    assert all(isinstance(f, Finding) for f in result["semantic"]["findings"])
    assert all(f.finding_type == "semantic" for f in result["semantic"]["findings"])


# --- Combined Aggregation ---


def test_security_high_still_forces_high_when_semantic_low():
    engine = _engine_with_semantic()
    result = engine.assess(code=SECURITY_HIGH_CODE, existing_code_list=NON_DUPLICATE_EXISTING)
    assert result["severity"] == SeverityLevel.HIGH
    assert result["security"]["severity"] == SeverityLevel.HIGH
    assert result["semantic"]["severity"] == SeverityLevel.LOW


def test_all_engines_low_produces_overall_low():
    engine = _engine_with_semantic()
    result = engine.assess(code=SIMPLE_CODE, existing_code_list=NON_DUPLICATE_EXISTING)
    assert result["severity"] == SeverityLevel.LOW


def test_semantic_and_security_both_high():
    code = 'def add(a, b):\n    api_key = "secret"\n    return a + b'
    engine = _engine_with_semantic()
    existing = [code]
    result = engine.assess(code=code, existing_code_list=existing)
    assert result["severity"] == SeverityLevel.HIGH
    assert result["security"]["severity"] == SeverityLevel.HIGH
    assert result["semantic"]["severity"] == SeverityLevel.HIGH


# --- calculate_risk Backward Compatibility ---


def test_calculate_risk_without_existing_code_list():
    engine = _engine_with_semantic()
    risk = engine.calculate_risk(pr_number=1, code=SIMPLE_CODE)
    assert risk == SeverityLevel.LOW


def test_calculate_risk_with_existing_code_list_and_duplicate():
    engine = _engine_with_semantic()
    risk = engine.calculate_risk(
        pr_number=1,
        code=DUPLICATE_CODE,
        existing_code_list=DUPLICATE_EXISTING,
    )
    assert risk == SeverityLevel.HIGH


def test_calculate_risk_without_code_and_without_semantic():
    engine = _engine_without_semantic()
    risk = engine.calculate_risk(pr_number=5)
    assert risk == SeverityLevel.LOW


# --- Empty Code With Semantic Service ---


def test_empty_code_with_semantic_service_produces_no_findings():
    engine = _engine_with_semantic()
    result = engine.assess(code="", existing_code_list=DUPLICATE_EXISTING)
    assert result["semantic"]["findings"] == []
    assert result["semantic_findings_count"] == 0


# --- Contract Shape With Semantic ---


def test_assess_output_always_contains_semantic_keys():
    engine = _engine_without_semantic()
    result = engine.assess(code=SIMPLE_CODE)
    assert "semantic" in result
    assert "semantic_findings_count" in result
    assert isinstance(result["semantic"], dict)
    assert "findings" in result["semantic"]
    assert "severity" in result["semantic"]


def test_assess_output_with_semantic_service_contains_semantic_keys():
    engine = _engine_with_semantic()
    result = engine.assess(code=DUPLICATE_CODE, existing_code_list=DUPLICATE_EXISTING)
    assert "semantic" in result
    assert "semantic_findings_count" in result
    assert result["semantic_findings_count"] == len(result["semantic"]["findings"])


# --- Determinism ---


def test_semantic_integration_deterministic_over_20_runs():
    engine = _engine_with_semantic()
    first = engine.assess(code=DUPLICATE_CODE, existing_code_list=DUPLICATE_EXISTING)
    for _ in range(20):
        result = engine.assess(code=DUPLICATE_CODE, existing_code_list=DUPLICATE_EXISTING)
        assert result["severity"] == first["severity"]
        assert result["semantic_findings_count"] == first["semantic_findings_count"]
        assert result["semantic"]["severity"] == first["semantic"]["severity"]


# --- M10: chunk-vs-chunk finds a copied function inside a large PR ---


def _large_pr_containing(copied: str) -> str:
    """A realistic PR: one copied function surrounded by unrelated new code."""
    filler = "\n\n".join(
        f"def unrelated_{i}(alpha, beta, gamma):\n"
        f"    result = alpha * {i} + beta\n"
        f"    for step in range(gamma):\n"
        f"        result += step\n"
        f"    return result"
        for i in range(6)
    )
    return filler + "\n\n" + copied + "\n\n" + filler


def test_single_copied_function_inside_a_large_pr_is_detected():
    """The case whole-PR-vs-unit could never catch.

    Embedding the entire PR as one vector diluted a single duplicated function
    below the threshold — the more unrelated code around it, the lower the score.
    Chunking the PR is what makes this detectable.
    """
    copied = (
        "def compute_total(values):\n"
        "    total = 0\n"
        "    for value in values:\n"
        "        total = total + value\n"
        "    return total"
    )
    engine = _engine_with_semantic()

    result = engine.assess(code=_large_pr_containing(copied), existing_code_list=[copied])

    assert result["semantic_findings_count"] == 1
    assert result["severity"] == SeverityLevel.HIGH


def test_large_pr_with_no_duplication_stays_clean():
    """The other half of the claim: chunking must not manufacture matches."""
    engine = _engine_with_semantic()

    result = engine.assess(
        code=_large_pr_containing("def unique_helper(q):\n    return q.strip().lower()"),
        existing_code_list=["def compute_total(values):\n    return sum(values)"],
    )

    assert result["semantic_findings_count"] == 0
