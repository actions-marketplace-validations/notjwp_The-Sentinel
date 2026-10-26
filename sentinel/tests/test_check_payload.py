"""M6: AuditOrchestrator.build_check_payload — findings -> Checks API shape."""

from sentinel.application.audit_orchestrator import AuditOrchestrator
from sentinel.domain.entities.finding import Finding
from sentinel.domain.value_objects.severity_level import SeverityLevel


def _orchestrator() -> AuditOrchestrator:
    return AuditOrchestrator(queue=None)


def _security(line: int, severity: SeverityLevel = SeverityLevel.HIGH) -> Finding:
    return Finding(
        rule="password_assignment",
        match="x",
        severity=severity,
        description="Hardcoded password detected.",
        line=line,
        recommendation="Load it from the environment.",
    )


def _doc(file_name: str, severity: SeverityLevel = SeverityLevel.MEDIUM) -> Finding:
    return Finding(
        rule="missing_installation",
        match=file_name,
        severity=severity,
        finding_type="documentation",
        description="Docs lack installation instructions.",
        file=file_name,
        line=1,
    )


def test_conclusion_mapping():
    orchestrator = _orchestrator()
    assert orchestrator.build_check_payload([], SeverityLevel.CRITICAL)["conclusion"] == "failure"
    assert orchestrator.build_check_payload([], SeverityLevel.HIGH)["conclusion"] == "failure"
    assert orchestrator.build_check_payload([], SeverityLevel.MEDIUM)["conclusion"] == "neutral"
    assert orchestrator.build_check_payload([], SeverityLevel.LOW)["conclusion"] == "success"
    assert orchestrator.build_check_payload([], "high")["conclusion"] == "failure"  # str risk


def test_security_annotation_resolved_via_line_map():
    # Lists (not tuples) on purpose: that's the shape after a Redis JSON round-trip.
    line_map = [["app.py", 14], ["app.py", 15]]
    payload = _orchestrator().build_check_payload(
        [_security(line=2)], SeverityLevel.HIGH, line_map=line_map
    )

    (annotation,) = payload["annotations"]
    assert annotation["path"] == "app.py"
    assert annotation["start_line"] == 15
    assert annotation["end_line"] == 15
    assert annotation["annotation_level"] == "failure"
    assert "Hardcoded password" in annotation["message"]
    assert "Load it from the environment." in annotation["message"]
    assert annotation["title"] == "password_assignment"


def test_doc_annotation_uses_its_own_location_and_level():
    payload = _orchestrator().build_check_payload([_doc("README.md")], SeverityLevel.MEDIUM)
    (annotation,) = payload["annotations"]
    assert annotation["path"] == "README.md"
    assert annotation["start_line"] == 1
    assert annotation["annotation_level"] == "warning"  # MEDIUM


def test_inline_code_doc_findings_are_not_annotated():
    """analyze_code labels findings file='inline' — a sentinel, not a repo path."""
    inline_doc = Finding(
        rule="missing_module_docstring",
        match="inline",
        severity=SeverityLevel.LOW,
        finding_type="documentation",
        description="Code is missing a module-level docstring.",
        file="inline",
        line=1,
    )
    payload = _orchestrator().build_check_payload([inline_doc], SeverityLevel.LOW)
    assert payload["annotations"] == []


def test_unmappable_findings_are_skipped_not_fatal():
    semantic = Finding(
        rule="semantic_duplicate",
        match="dup",
        severity=SeverityLevel.HIGH,
        finding_type="semantic",
        similarity_score=0.95,
    )
    unmapped_security = _security(line=7)  # no line_map supplied
    out_of_range = _security(line=99)

    payload = _orchestrator().build_check_payload(
        [semantic, unmapped_security, out_of_range],
        SeverityLevel.HIGH,
        line_map=[["app.py", 3]],
    )
    assert payload["annotations"] == []
    assert payload["conclusion"] == "failure"  # conclusion unaffected


def test_annotation_cap_and_overflow_note():
    findings = [_doc(f"file_{i}.md") for i in range(55)]
    payload = _orchestrator().build_check_payload(findings, SeverityLevel.MEDIUM)
    assert len(payload["annotations"]) == AuditOrchestrator.MAX_CHECK_ANNOTATIONS
    assert "(+5 more" in payload["summary"]


def test_title_and_summary_counts():
    findings = [_security(1), _security(2), _doc("README.md")]
    payload = _orchestrator().build_check_payload(
        findings,
        SeverityLevel.HIGH,
        line_map=[["a.py", 1], ["a.py", 2]],
        semantic_findings_count=3,
    )
    assert payload["title"] == "Risk: HIGH — 2 security, 1 documentation issue(s)"
    assert "**Risk score:** HIGH" in payload["summary"]
    assert "Semantic duplicates: 3" in payload["summary"]


# --- The title must not contradict the score ---------------------------------
#
# "Risk: MEDIUM — 0 security, 0 documentation issue(s)" is what the tool used
# to post on an ordinary pull request. A reader resolves that contradiction by
# distrusting the check, so the title now names whatever raised the score.


def test_an_elevated_score_with_no_findings_names_the_complexity_that_raised_it():
    payload = _orchestrator().build_check_payload(
        [], SeverityLevel.MEDIUM, complexity=47, semantic_findings_count=0
    )
    assert payload["title"] == "Risk: MEDIUM — no issues found; code complexity 47"
    assert "- Code complexity: 47" in payload["summary"]
    assert "No defect was found" in payload["summary"]


def test_a_clean_review_says_so_instead_of_printing_two_zeros():
    payload = _orchestrator().build_check_payload([], SeverityLevel.LOW, complexity=3)
    assert payload["title"] == "Risk: LOW — no issues found"
    # LOW needs no excuse: the complexity is in the summary, not the title.
    assert "complexity" not in payload["title"]
    assert "No defect was found" not in payload["summary"]


def test_near_duplicates_are_named_when_they_are_what_raised_the_score():
    payload = _orchestrator().build_check_payload(
        [], SeverityLevel.HIGH, semantic_findings_count=2, complexity=99
    )
    assert payload["title"] == (
        "Risk: HIGH — no security or documentation issues; "
        "2 near-duplicate(s) of existing code"
    )
    # Semantic duplicates ARE a finding, so the diff-shape caveat is wrong here.
    assert "No defect was found" not in payload["summary"]


def test_counts_still_lead_the_title_whenever_there_is_something_to_count():
    """The explanatory forms are for the empty case only; they must not displace counts."""
    payload = _orchestrator().build_check_payload(
        [_security(1)], SeverityLevel.HIGH, line_map=[["a.py", 1]], complexity=99
    )
    assert payload["title"] == "Risk: HIGH — 1 security, 0 documentation issue(s)"


def test_complexity_is_optional_and_its_absence_is_not_a_crash():
    payload = _orchestrator().build_check_payload([], SeverityLevel.MEDIUM)
    assert payload["title"] == "Risk: MEDIUM — no issues found"
    assert "Code complexity" not in payload["summary"]
