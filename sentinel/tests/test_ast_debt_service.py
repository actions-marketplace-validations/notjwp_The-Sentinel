"""Per-function complexity: what structure sees that a keyword count cannot.

Several tests deliberately assert the CONTRAST with the regex debt engine.
That contrast is the reason this analyzer exists — the regex sums every branch
in the diff, so its number grows with the size of a pull request rather than
the tangle of anything in it. Pinning the difference means a change that
reintroduces the summing fails here rather than in someone's review.
"""

from pathlib import Path

from sentinel.domain.services.ast_debt_service import ASTDebtService
from sentinel.domain.services.debt_service import DebtService
from sentinel.domain.value_objects.severity_level import SeverityLevel

ast_service = ASTDebtService()
regex_service = DebtService()

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def score(source: str) -> int:
    result = ast_service.analyze_source(source, "app.py")
    assert result is not None, "source unexpectedly failed to parse"
    return result


# --- unparseable input is reported, not guessed at ---------------------------


def test_unparseable_source_returns_none_not_a_score():
    """None ("could not look") must not be confused with 1 ("looked, found nothing").

    The caller falls back to the regex engine only in the first case.
    """
    assert ast_service.analyze_source("    indented fragment = 1", "a.py") is None
    assert ast_service.analyze_source("def broken(:", "a.py") is None
    assert ast_service.analyze_source("", "a.py") is None
    assert ast_service.analyze_source(None, "a.py") is None

    # Parsed, genuinely unbranching: a real score, not an absence.
    assert ast_service.analyze_source("x = 1\n", "a.py") == 1


def test_can_analyze_matches_analyze_source():
    assert ast_service.can_analyze("x = 1") is True
    assert ast_service.can_analyze("   nope = (") is False
    assert ast_service.can_analyze("") is False


# --- the headline fix: a sum is not a complexity -----------------------------


def test_many_simple_functions_score_like_one_simple_function():
    """The defect this class exists to fix.

    Twenty tidy functions are not a complex program, but a keyword count over
    the whole blob says they are. With the old engine a PR of this shape scored
    21 and tripped the HIGH threshold of 15 having found nothing wrong.
    """
    source = "\n\n".join(
        f"def f{i}(x):\n    if x:\n        return 1\n    return 0" for i in range(20)
    )

    assert score(source) == 2
    assert regex_service.calculate_complexity(source) == 21, (
        "the regex engine is expected to sum here — that's the point"
    )


def test_one_tangled_function_still_scores_high():
    """The other half of the guard: the fix must not simply deflate everything."""
    source = "def gnarly(v):\n" + "".join(
        f"    if v > {i}:\n        v += 1\n" for i in range(20)
    ) + "    return v"

    assert score(source) == 21
    assert regex_service.evaluate_debt(source, complexity=score(source))["severity"] == (
        SeverityLevel.HIGH
    )


def test_the_worst_function_wins_not_the_last_or_the_first():
    source = (
        "def tidy():\n    return 1\n\n"
        "def messy(v):\n" + "".join(f"    if v > {i}:\n        v += 1\n" for i in range(9)) +
        "    return v\n\n"
        "def also_tidy():\n    return 2\n"
    )
    assert score(source) == 10


# --- the fixtures must keep their current severities -------------------------


def test_the_debt_fixtures_score_the_same_as_they_always_have():
    """Each fixture is a single function, so per-function == whole-file.

    This is what let the thresholds survive the change untouched. If someone
    adds a second function to a fixture, the two engines diverge and the
    severity tests in test_debt_service.py start depending on which one ran —
    this test is here to fail first and say why.
    """
    expected = {
        "simple_function.py": 1,
        "medium_function.py": 8,
        "complex_function.py": 16,
    }
    for name, want in expected.items():
        source = (FIXTURES_DIR / name).read_text(encoding="utf-8")
        assert score(source) == want, name
        assert regex_service.calculate_complexity(source) == want, (
            f"{name}: engines disagree, so the severity tests are now ambiguous"
        )


# --- what counts as a decision -----------------------------------------------


def test_each_branching_construct_adds_exactly_one():
    assert score("def f(x):\n    if x:\n        pass\n") == 2
    assert score("def f(xs):\n    for x in xs:\n        pass\n") == 2
    assert score("def f(x):\n    while x:\n        pass\n") == 2
    assert score("def f():\n    try:\n        pass\n    except ValueError:\n        pass\n") == 2
    assert score("def f(x):\n    return 1 if x else 2\n") == 2
    assert score("def f(x):\n    assert x\n") == 2


def test_elif_counts_once_and_bare_else_not_at_all():
    """An `elif` is an `If` in the parent's orelse; `else` adds no new decision."""
    assert score("def f(x):\n    if x:\n        pass\n    else:\n        pass\n") == 2
    assert score(
        "def f(x):\n    if x:\n        pass\n    elif x > 1:\n        pass\n"
        "    else:\n        pass\n"
    ) == 3


def test_boolean_operators_count_per_extra_operand():
    assert score("def f(a, b):\n    return a and b\n") == 2
    assert score("def f(a, b, c):\n    return a and b and c\n") == 3
    assert score("def f(a, b, c):\n    if a or b or c:\n        pass\n") == 4  # 1 + if + 2


def test_comprehension_counts_its_loop_and_each_filter():
    assert score("def f(xs):\n    return [x for x in xs]\n") == 2
    assert score("def f(xs):\n    return [x for x in xs if x]\n") == 3
    assert score("def f(xs):\n    return [x for x in xs if x if x > 1]\n") == 4


def test_match_statement_counts_per_case():
    source = (
        "def f(x):\n"
        "    match x:\n"
        "        case 1:\n            pass\n"
        "        case 2:\n            pass\n"
        "        case _:\n            pass\n"
    )
    assert score(source) == 4  # 1 + three cases


def test_async_functions_and_loops_are_scored_too():
    source = "async def f(xs):\n    async for x in xs:\n        if x:\n            pass\n"
    assert score(source) == 3


# --- scoping ------------------------------------------------------------------


def test_a_nested_helper_is_scored_separately_not_folded_into_its_parent():
    """Otherwise an outer function looks complex merely for containing one."""
    source = (
        "def outer(v):\n"
        "    def inner(x):\n" + "".join(f"        if x > {i}:\n            pass\n" for i in range(9)) +
        "    return inner(v)\n"
    )
    assert score(source) == 10  # inner's own score, not outer's 1 + inner's 9


def test_methods_are_scored_per_method_not_per_class():
    source = (
        "class C:\n"
        "    def a(self, x):\n        if x:\n            pass\n\n"
        "    def b(self, x):\n        if x:\n            pass\n"
    )
    assert score(source) == 2


def test_module_level_branching_is_scored_as_its_own_unit():
    """A script with no functions still has a complexity."""
    assert score("import os\nif os.name:\n    x = 1\n") == 2


# --- text that merely looks like code ----------------------------------------


def test_keywords_in_strings_and_comments_do_not_count():
    source = (
        'def f():\n'
        '    """Explain if and for and while to the reader."""\n'
        '    # if for while and or\n'
        '    note = "if for while and or except"\n'
        '    return note\n'
    )
    assert score(source) == 1
    assert regex_service.calculate_complexity(source) > 1, (
        "the regex engine is expected to fire on the prose — that's the contrast"
    )


# --- only the functions this pull request edited -----------------------------


_TWO_FUNCTIONS = (
    "def inherited(v):\n"                                              # 1
    + "".join(f"    if v > {i}:\n        v += 1\n" for i in range(9))  # 2..19
    + "    return v\n"                                                 # 20
    "\n"                                                               # 21
    "def mine(x):\n"                                                   # 22
    "    if x:\n"                                                      # 23
    "        return 1\n"                                               # 24
    "    return 0\n"                                                   # 25
)


def test_only_functions_the_pr_touched_are_scored():
    """Somebody else's complex function is accurate to report and not your business.

    The same restriction ASTSecurityService applies to its findings. Without it,
    editing one line of a file reports whatever the worst function in it happens
    to be — which is the false positive this engine exists to remove, quieter.
    """
    whole_file = ast_service.analyze_source(_TWO_FUNCTIONS, "a.py")
    just_mine = ast_service.analyze_source(_TWO_FUNCTIONS, "a.py", touched_lines={23})

    assert whole_file == 10, "unfiltered, the inherited function dominates"
    assert just_mine == 2, "filtered, only the edited function counts"


def test_editing_the_complex_function_does_report_it():
    """The other half: inherit the number when you are the one in there."""
    assert ast_service.analyze_source(_TWO_FUNCTIONS, "a.py", touched_lines={5}) == 10


def test_a_line_anywhere_inside_the_function_counts_as_touching_it():
    """Span-based, not signature-based — editing the body is editing the function."""
    for line in (2, 10, 19):
        assert ast_service.analyze_source(_TWO_FUNCTIONS, "a.py", touched_lines={line}) == 10


def test_touching_no_function_scores_one_rather_than_the_file_s_worst():
    """A changed import is not a complexity problem you introduced."""
    assert ast_service.analyze_source(_TWO_FUNCTIONS, "a.py", touched_lines={21}) == 1
    assert ast_service.analyze_source(_TWO_FUNCTIONS, "a.py", touched_lines=set()) == 1


def test_an_empty_filter_is_distinct_from_no_filter():
    """set() means 'touched nothing here'; None means 'no diff information'."""
    assert ast_service.analyze_source(_TWO_FUNCTIONS, "a.py", touched_lines=set()) == 1
    assert ast_service.analyze_source(_TWO_FUNCTIONS, "a.py", touched_lines=None) == 10
