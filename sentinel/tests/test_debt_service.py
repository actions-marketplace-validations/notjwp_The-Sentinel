import random
from pathlib import Path

import pytest

from sentinel.domain.services.debt_service import DebtService
from sentinel.domain.value_objects.severity_level import SeverityLevel

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


def _code_for_decisions(decision_points: int) -> str:
    lines = ["def boundary_fn(x):"]
    for index in range(decision_points):
        lines.append(f"    if x > {index}:")
        lines.append("        x += 1")
    lines.append("    return x")
    return "\n".join(lines)


def _assert_valid_result(result: dict) -> None:
    assert result["complexity"] >= 1
    assert 0 <= result["maintainability"] <= 100
    assert result["severity"] in {SeverityLevel.LOW, SeverityLevel.MEDIUM, SeverityLevel.HIGH}


def test_simple_function_low_risk():
    result = DebtService().evaluate_debt(_load_fixture("simple_function.py"))
    assert result["severity"] == SeverityLevel.LOW


def test_medium_complexity_function_medium_risk():
    result = DebtService().evaluate_debt(_load_fixture("medium_function.py"))
    assert result["severity"] == SeverityLevel.MEDIUM


def test_highly_nested_function_high_risk():
    result = DebtService().evaluate_debt(_load_fixture("complex_function.py"))
    assert result["severity"] == SeverityLevel.HIGH


def test_empty_string_input_is_safe():
    result = DebtService().evaluate_debt("")
    _assert_valid_result(result)


def test_whitespace_only_input_is_safe():
    result = DebtService().evaluate_debt("   \n \t\n  ")
    _assert_valid_result(result)


def test_comment_only_input_is_safe():
    result = DebtService().evaluate_debt("# header\n# details\n# summary")
    _assert_valid_result(result)


def test_very_long_code_string_10000_lines_is_bounded_and_safe():
    long_code = "\n".join("if" for _ in range(10000))
    result = DebtService().evaluate_debt(long_code)
    _assert_valid_result(result)
    assert result["severity"] == SeverityLevel.HIGH


def test_pathological_keyword_spam_is_safe():
    spam = ("if if if if " * 5000).strip()
    result = DebtService().evaluate_debt(spam)
    _assert_valid_result(result)
    # Was 20001. The raw count is a diff-size proxy, not a complexity, and
    # printing it beside a real per-function score overstated its precision.
    assert result["complexity"] == DebtService.MAX_FALLBACK_COMPLEXITY
    assert result["severity"] == SeverityLevel.HIGH


def test_keywords_inside_quotes_are_counted_deterministically():
    code = 'def quoted():\n    return "if elif for while except and or case"'
    result = DebtService().evaluate_debt(code)
    _assert_valid_result(result)
    assert result["complexity"] == 9


def test_null_byte_content_is_safe():
    code = "def nul():\n    text = \"abc\\x00def\"\n    if text:\n        return 1"
    result = DebtService().evaluate_debt(code)
    _assert_valid_result(result)


def test_unicode_heavy_content_is_safe():
    code = "def unicode_payload():\n    value = \"λ🚀漢字✨\"\n    if value:\n        return value"
    result = DebtService().evaluate_debt(code)
    _assert_valid_result(result)


@pytest.mark.parametrize(
    ("decision_points", "expected_complexity", "expected_severity"),
    [
        (6, 7, SeverityLevel.LOW),
        (7, 8, SeverityLevel.MEDIUM),
        (14, 15, SeverityLevel.MEDIUM),
        (15, 16, SeverityLevel.HIGH),
    ],
)
def test_boundary_complexity_thresholds(
    decision_points: int,
    expected_complexity: int,
    expected_severity: SeverityLevel,
):
    code = _code_for_decisions(decision_points)
    result = DebtService().evaluate_debt(code)
    assert result["complexity"] == expected_complexity
    assert result["severity"] == expected_severity
    _assert_valid_result(result)


def test_maintainability_always_clamped_between_zero_and_hundred():
    service = DebtService()
    low_input = service.calculate_maintainability(complexity=1, loc=1)
    high_input = service.calculate_maintainability(complexity=1000, loc=5000)
    assert 0 <= low_input <= 100
    assert 0 <= high_input <= 100


def test_maintainability_formula_exact_values():
    service = DebtService()
    assert service.calculate_maintainability(complexity=1, loc=1) == 94.9
    assert service.calculate_maintainability(complexity=10, loc=100) == 40.0
    assert service.calculate_maintainability(complexity=0, loc=0) == 100.0


def test_maintainability_clamps_hard_bounds_exactly():
    service = DebtService()
    assert service.calculate_maintainability(complexity=10_000, loc=10_000) == 0.0
    assert service.calculate_maintainability(complexity=-100, loc=-100) == 100.0


def test_calculate_complexity_counts_all_decision_keywords_exactly():
    code = "if elif for while except and or case"
    complexity = DebtService().calculate_complexity(code)
    assert complexity == 9


def test_evaluate_debt_deterministic_for_100_runs():
    service = DebtService()
    code = _load_fixture("medium_function.py")
    first = service.evaluate_debt(code)
    for _ in range(99):
        assert service.evaluate_debt(code) == first


def test_fuzz_synthetic_strings_deterministic_no_crash():
    seed = 1337
    rng = random.Random(seed)
    tokens = ["if", "elif", "for", "while", "except", "and", "or", "case", "value", "(", ")", ":"]
    service = DebtService()

    for _ in range(100):
        line_count = rng.randint(1, 30)
        lines = []
        for _ in range(line_count):
            token_count = rng.randint(3, 15)
            line = " ".join(rng.choice(tokens) for _ in range(token_count))
            lines.append(line)
        code = "\n".join(lines)
        result = service.evaluate_debt(code)
        _assert_valid_result(result)


# --- the AST override ---------------------------------------------------------


def test_a_supplied_complexity_replaces_the_keyword_count():
    """The caller that parsed the real files knows better than the regex.

    `code` here would count 21 decisions; the supplied 2 is what a per-function
    analyzer saw. The supplied value must drive BOTH the severity band and the
    maintainability figure, or the report contradicts itself.
    """
    service = DebtService()
    code = "\n\n".join(
        f"def f{i}(x):\n    if x:\n        return 1\n    return 0" for i in range(20)
    )

    counted = service.evaluate_debt(code)
    supplied = service.evaluate_debt(code, complexity=2)

    assert counted["complexity"] == 21
    assert counted["severity"] == SeverityLevel.HIGH
    assert supplied["complexity"] == 2
    assert supplied["severity"] == SeverityLevel.LOW
    assert supplied["maintainability"] > counted["maintainability"]


def test_the_override_is_keyword_only_and_optional():
    """Positional would silently swallow a second argument meant for something else."""
    service = DebtService()
    code = "def f(x):\n    if x:\n        return 1\n"

    assert service.evaluate_debt(code) == service.evaluate_debt(code, complexity=None)
    with pytest.raises(TypeError):
        service.evaluate_debt(code, 2)


def test_an_overridden_result_keeps_the_contract_shape():
    result = DebtService().evaluate_debt("def f():\n    return 1\n", complexity=9)
    assert set(result.keys()) == {"severity", "complexity", "maintainability"}
    assert type(result["complexity"]) is int
    assert result["severity"] == SeverityLevel.MEDIUM


def test_keyword_fallback_is_clamped_to_a_reportable_ceiling():
    """The fallback counts keywords across the WHOLE blob, so it tracks diff size.

    A large diff used to report a six-figure "complexity" beside a real
    per-function score, as though the two meant the same thing. Clamping does
    not make the fallback meaningful — ASTDebtService does that — it stops the
    report claiming a precision it does not have.
    """
    service = DebtService()
    huge = "if x and y or z:\n    pass\n" * 50000

    complexity = service.calculate_complexity(huge)
    assert complexity == DebtService.MAX_FALLBACK_COMPLEXITY
    assert isinstance(complexity, int)

    result = service.evaluate_debt(huge)
    assert result["complexity"] == DebtService.MAX_FALLBACK_COMPLEXITY
    assert set(result) == {"complexity", "maintainability", "severity"}


def test_the_ceiling_never_touches_a_realistic_score():
    service = DebtService()
    ordinary = "if a:\n    pass\n" * 20  # 21, far under the ceiling
    assert service.calculate_complexity(ordinary) == 21
    assert service.calculate_complexity("x = 1") == 1
