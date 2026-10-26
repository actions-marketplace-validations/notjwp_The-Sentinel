"""The check run must not contradict the job it sits beside.

A red check next to a green job is a contradiction the reader has to resolve,
and they resolve it by distrusting the check. When a repository has said which
severity should fail its build, the check run honours that too.
"""

import pytest

from sentinel.application.audit_orchestrator import NEVER_FAILS, AuditOrchestrator
from sentinel.domain.value_objects.severity_level import SeverityLevel


def _conclusion(risk, fail_on=None) -> str:
    payload = AuditOrchestrator(queue=None).build_check_payload([], risk, fail_on=fail_on)
    return payload["conclusion"]


def test_without_a_threshold_nothing_changes():
    """The hosted path passes no threshold: it has no build to fail."""
    assert _conclusion(SeverityLevel.CRITICAL) == "failure"
    assert _conclusion(SeverityLevel.HIGH) == "failure"
    assert _conclusion(SeverityLevel.MEDIUM) == "neutral"
    assert _conclusion(SeverityLevel.LOW) == "success"


@pytest.mark.parametrize(
    ("risk", "fail_on", "expected"),
    [
        (SeverityLevel.HIGH, "CRITICAL", "neutral"),   # below the bar: do not fail the check
        (SeverityLevel.CRITICAL, "CRITICAL", "failure"),
        (SeverityLevel.HIGH, "HIGH", "failure"),
        (SeverityLevel.HIGH, "LOW", "failure"),        # a lower bar must not downgrade
        (SeverityLevel.CRITICAL, NEVER_FAILS, "neutral"),
        (SeverityLevel.HIGH, NEVER_FAILS, "neutral"),
    ],
)
def test_the_threshold_is_honoured(risk, fail_on, expected):
    assert _conclusion(risk, fail_on) == expected


def test_the_threshold_only_ever_downgrades():
    """A strict threshold must not turn a clean review red."""
    assert _conclusion(SeverityLevel.LOW, "LOW") == "success"
    assert _conclusion(SeverityLevel.MEDIUM, "LOW") == "neutral"


def test_an_unrecognised_threshold_is_ignored_rather_than_guessed():
    """The Action canonicalises before this point; anything else leaves it alone."""
    assert _conclusion(SeverityLevel.HIGH, "banana") == "failure"
    assert _conclusion(SeverityLevel.HIGH, "") == "failure"
