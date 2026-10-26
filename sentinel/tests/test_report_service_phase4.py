from dataclasses import replace

from sentinel.application.audit_orchestrator import AuditOrchestrator
from sentinel.domain.entities.finding import Finding
from sentinel.domain.value_objects.severity_level import SeverityLevel


def _security_finding(*, explanation: str | None = "explain", fix: str | None = "fix") -> Finding:
    return Finding(
        rule="sql_injection",
        match="SELECT",
        severity=SeverityLevel.HIGH,
        category="Injection",
        owasp_category="A03: Injection",
        description="SQL injection detected",
        recommendation="Use parameterized queries",
        explanation=explanation,
        fix_suggestion=fix,
    )


def _doc_finding() -> Finding:
    return Finding(
        rule="missing_usage",
        match="README.md",
        severity=SeverityLevel.MEDIUM,
        finding_type="documentation",
        category="Documentation",
        owasp_category="N/A",
        description="Documentation missing usage",
        file="README.md",
        line=1,
        recommendation="Add usage section",
    )


def test_format_report_handles_no_findings():
    orchestrator = AuditOrchestrator(queue=None)
    report = orchestrator.build_report([], SeverityLevel.LOW)

    assert "# Sentinel AI Code Review" in report
    assert "## Risk Score: LOW" in report
    assert "No security issues detected." in report
    assert "No AI explanation available." in report
    assert "No fix suggestion available." in report


def test_format_report_handles_multiple_findings_and_formatting():
    findings = [_security_finding(), _security_finding(explanation="another", fix="fix2"), _doc_finding()]

    orchestrator = AuditOrchestrator(queue=None)
    report = orchestrator.build_report(
        findings,
        SeverityLevel.HIGH,
        complexity=10,
        maintainability=87.25,
        semantic_findings_count=3,
    )

    assert "## Security Issues" in report
    assert "SQL injection detected" in report
    assert report.count("```python") == 2
    assert "## Documentation Issues" in report
    assert "Documentation missing usage" in report
    assert "- Complexity: 10" in report
    assert "- Maintainability: 87.25" in report
    assert "- Similar findings detected: 3" in report


def test_format_report_handles_missing_explanation_and_fix_branches():
    """A finding with nothing to say: the only case that prints the placeholders.

    The recommendation is cleared deliberately. With one present the report
    uses it in place of the missing AI fix, which is the behaviour the next
    test pins — so leaving it set here would quietly stop exercising the empty
    branch at all.
    """
    finding = replace(_security_finding(explanation=None, fix=None), recommendation=None)

    orchestrator = AuditOrchestrator(queue=None)
    report = orchestrator.build_report([finding], "medium", complexity=5)

    assert "## Risk Score: MEDIUM" in report
    assert "No AI explanation available." in report
    assert "No fix suggestion available." in report
    assert "- Complexity: 5" in report
    assert "Maintainability" not in report


def test_the_rule_s_own_recommendation_stands_in_for_a_missing_ai_fix():
    """Without a key there is no AI fix, and an empty section helps nobody.

    The engine already knows what to do about its own finding; printing that
    beats "no fix suggestion available", and beats a generic string that may
    be wrong for the rule.
    """
    findings = [_security_finding(explanation=None, fix=None)]

    report = AuditOrchestrator(queue=None).build_report(findings, "high")

    assert "Use parameterized queries" in report
    assert "No fix suggestion available." not in report



