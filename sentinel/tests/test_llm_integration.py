from sentinel.application.audit_orchestrator import AuditOrchestrator
from sentinel.domain.entities.finding import Finding
from sentinel.domain.value_objects.severity_level import SeverityLevel
from sentinel.infrastructure.llm.base import LLMProvider
from sentinel.infrastructure.llm.llm_service import LLMService
from sentinel.workers.job_queue import JobQueue


class FakeLLMProvider(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    def generate_pr_audit(self, code: str, findings_summary: str) -> str:
        self.calls += 1
        return "Issue:\nExplanation: explanation\nFix: fixed_code"


class FailingLLMProvider(LLMProvider):
    def generate_pr_audit(self, code: str, findings_summary: str) -> str:
        raise RuntimeError("failed")


def test_high_severity_triggers_llm_and_attaches_fields(monkeypatch):
    monkeypatch.setenv("ENABLE_LLM", "true")

    provider = FakeLLMProvider()
    service = LLMService(provider=provider, enable_llm=True, max_calls=5)
    orchestrator = AuditOrchestrator(JobQueue(), llm_service=service)

    findings = [
        Finding(
            rule="sql_injection",
            match="SELECT",
            severity=SeverityLevel.HIGH,
            description="SQL injection detected",
        )
    ]

    enriched = orchestrator.enrich_findings_with_llm("query = user_input", findings)

    assert enriched[0].explanation == "explanation"
    assert enriched[0].fix_suggestion == "fixed_code"
    assert provider.calls == 1


def test_low_severity_does_not_trigger_llm(monkeypatch):
    monkeypatch.setenv("ENABLE_LLM", "true")

    provider = FakeLLMProvider()
    service = LLMService(provider=provider, enable_llm=True, max_calls=5)
    orchestrator = AuditOrchestrator(JobQueue(), llm_service=service)

    findings = [
        Finding(
            rule="eval_call",
            match="eval(",
            severity=SeverityLevel.MEDIUM,
            description="Use of eval",
        )
    ]

    enriched = orchestrator.enrich_findings_with_llm("x = eval('2+2')", findings)

    assert enriched[0].explanation is None
    assert enriched[0].fix_suggestion is None
    assert provider.calls == 0


def test_fallback_values_are_used_on_provider_failure():
    service = LLMService(provider=FailingLLMProvider(), enable_llm=True, max_calls=5)
    
    findings = [
        Finding(rule="r1", match="m", severity=SeverityLevel.HIGH, description="issue")
    ]
    
    result = service.generate_pr_audit("code", findings)
    
    assert result[id(findings[0])]["fix"] == LLMService.FALLBACK_FIX
    assert result[id(findings[0])]["explanation"] == "issue"  # the finding's own description, not the generic string


def test_llm_disabled_mode_skips_processing(monkeypatch):
    monkeypatch.setenv("ENABLE_LLM", "false")

    provider = FakeLLMProvider()
    service = LLMService(provider=provider, enable_llm=False, max_calls=5)
    orchestrator = AuditOrchestrator(JobQueue(), llm_service=service)

    findings = [
        Finding(
            rule="sql_injection",
            match="SELECT",
            severity=SeverityLevel.CRITICAL,
            description="SQL injection detected",
        )
    ]

    enriched = orchestrator.enrich_findings_with_llm("query = user_input", findings)

    assert enriched[0].explanation is None
    assert enriched[0].fix_suggestion is None
    assert provider.calls == 0





# --- Prompt content: findings carry their matched code (M9) ---


class _CapturingProvider:
    """Records the findings_summary the service builds."""

    def __init__(self) -> None:
        self.summary: str | None = None

    def generate_pr_audit(self, code: str, findings_summary: str) -> str:
        self.summary = findings_summary
        return "Issue: x\nExplanation: e\nFix: f"


def _finding(**kw):
    from sentinel.domain.entities.finding import Finding
    from sentinel.domain.value_objects.severity_level import SeverityLevel

    defaults = dict(
        rule="password_assignment",
        match='password = "hunter2"',
        severity=SeverityLevel.HIGH,
        description="Possible hardcoded password assignment detected.",
        line=12,
        recommendation="Store passwords outside source code.",
    )
    defaults.update(kw)
    return Finding(**defaults)


def test_prompt_carries_matched_code_line_and_description():
    """Without these the model reasons about the rule slug, not the offending code."""
    provider = _CapturingProvider()
    service = LLMService(provider=provider, enable_llm=True)

    service.generate_pr_audit("code", [_finding()])

    summary = provider.summary
    assert 'Matched code: password = "hunter2"' in summary
    assert "Line: 12" in summary
    assert "Possible hardcoded password assignment detected." in summary
    assert "[password_assignment]" in summary  # rule kept, but as a labelled id
    assert "Severity: HIGH" in summary


def test_prompt_collapses_multiline_matches_to_one_line():
    """A multi-line match would otherwise break the numbered-list shape."""
    provider = _CapturingProvider()
    service = LLMService(provider=provider, enable_llm=True)

    service.generate_pr_audit("code", [_finding(match="query = (\n   'SELECT *'\n)")])

    line = [x for x in provider.summary.split("\n") if "Matched code:" in x][0]
    assert line.strip() == "Matched code: query = ( 'SELECT *' )"


def test_prompt_truncates_a_pathological_match():
    provider = _CapturingProvider()
    service = LLMService(provider=provider, enable_llm=True)

    service.generate_pr_audit("code", [_finding(match="x" * 5000)])

    line = [x for x in provider.summary.split("\n") if "Matched code:" in x][0]
    assert "...(truncated)" in line
    assert len(line) < 200


def test_prompt_survives_a_finding_with_no_match_or_line():
    provider = _CapturingProvider()
    service = LLMService(provider=provider, enable_llm=True)

    service.generate_pr_audit("code", [_finding(match="", line=None)])

    assert "Matched code:" not in provider.summary
    assert "Line:" not in provider.summary
    assert "[password_assignment]" in provider.summary
