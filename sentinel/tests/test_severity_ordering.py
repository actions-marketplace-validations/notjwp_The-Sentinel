"""Severity ordering lives in the domain because two callers must agree on it.

The Action's exit code and the check run's conclusion are computed in different
layers from the same question — "is this bad enough to fail?" — and if they
answer differently the user sees a passing job beside a failing check.
"""

from sentinel.domain.value_objects.severity_level import SeverityLevel


def test_the_order_is_low_to_critical():
    ranks = [level.rank for level in (
        SeverityLevel.LOW, SeverityLevel.MEDIUM, SeverityLevel.HIGH, SeverityLevel.CRITICAL
    )]
    assert ranks == sorted(ranks) == [0, 1, 2, 3]


def test_rank_of_accepts_both_shapes():
    assert SeverityLevel.rank_of(SeverityLevel.HIGH) == SeverityLevel.rank_of("high") == 2
    assert SeverityLevel.rank_of("  Critical  ") == 3


def test_rank_of_rejects_what_is_not_a_severity():
    """None, not zero: 'unknown' must be distinguishable from 'lowest'."""
    for value in (None, "", "banana", 3, object()):
        assert SeverityLevel.rank_of(value) is None
