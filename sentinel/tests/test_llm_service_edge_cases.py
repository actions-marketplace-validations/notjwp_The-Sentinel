from sentinel.domain.entities.finding import Finding
from sentinel.domain.value_objects.severity_level import SeverityLevel
from sentinel.infrastructure.llm.llm_service import LLMService


class _SuccessProvider:
    def __init__(self) -> None:
        self.calls = 0

    def generate_pr_audit(self, code: str, findings_summary: str):
        self.calls += 1
        return "Issue:\nExplanation: explanation\nFix: fixed_code"


class _EmptyProvider:
    def generate_pr_audit(self, code: str, summary: str):
        _ = code
        _ = summary
        return ""


class _FailingProvider:
    def generate_pr_audit(self, code: str, summary: str):
        _ = code
        _ = summary
        raise RuntimeError("failure")


def _finding(severity=SeverityLevel.HIGH, rule="r1", finding_type="security") -> Finding:
    return Finding(rule=rule, match="m", severity=severity, finding_type=finding_type, description="d")


def test_llm_disabled_returns_fallback_and_no_calls():
    provider = _SuccessProvider()
    service = LLMService(provider=provider, enable_llm=False, max_calls=5)

    f = _finding()
    result = service.generate_pr_audit("code", [f])

    assert result[id(f)]["fix"] == LLMService.FALLBACK_FIX
    assert result[id(f)]["explanation"] == "d"  # the finding's own description, not the generic string
    assert service.call_count == 0
    assert provider.calls == 0


def test_invalid_severity_string_does_not_invoke_provider():
    provider = _SuccessProvider()
    service = LLMService(provider=provider, enable_llm=True, max_calls=5)

    f = _finding(severity="LOW")
    result = service.generate_pr_audit("code", [f])

    assert id(f) not in result
    assert service.call_count == 0


def test_success_results_are_stripped_and_non_empty():
    provider = _SuccessProvider()
    service = LLMService(provider=provider, enable_llm=True, max_calls=5)

    f = _finding(severity=SeverityLevel.CRITICAL)
    result = service.generate_pr_audit("", [f])

    assert result[id(f)]["fix"] == "fixed_code"
    assert result[id(f)]["explanation"] == "explanation"


def test_provider_failure_and_empty_results_use_fallbacks():
    f = _finding()
    
    failing = LLMService(provider=_FailingProvider(), enable_llm=True, max_calls=5)
    result_fail = failing.generate_pr_audit("code", [f])
    assert result_fail[id(f)]["fix"] == LLMService.FALLBACK_FIX
    assert result_fail[id(f)]["explanation"] == "d"  # the finding's own description, not the generic string

    empty = LLMService(provider=_EmptyProvider(), enable_llm=True, max_calls=5)
    result_empty = empty.generate_pr_audit("code", [f])
    assert result_empty[id(f)]["fix"] == LLMService.FALLBACK_FIX
    assert result_empty[id(f)]["explanation"] == "d"  # the finding's own description, not the generic string


def test_call_limit_enforced_across_multiple_calls():
    provider = _SuccessProvider()
    service = LLMService(provider=provider, enable_llm=True, max_calls=2)

    f1 = _finding(rule="1")
    f2 = _finding(rule="2")
    f3 = _finding(rule="3")

    first = service.generate_pr_audit("code", [f1])
    second = service.generate_pr_audit("code", [f2])
    third = service.generate_pr_audit("code", [f3])

    assert first[id(f1)]["fix"] == "fixed_code"
    assert second[id(f2)]["explanation"] == "explanation"
    assert third[id(f3)]["fix"] == LLMService.FALLBACK_FIX
    assert service.call_count == 2
    assert service.calls_made == 2


def test_reset_budget_allows_subsequent_calls():
    provider = _SuccessProvider()
    service = LLMService(provider=provider, enable_llm=True, max_calls=1)

    f = _finding()
    assert service.generate_pr_audit("code", [f])[id(f)]["fix"] == "fixed_code"
    assert service.generate_pr_audit("code", [f])[id(f)]["fix"] == LLMService.FALLBACK_FIX

    service.reset_budget()

    assert service.call_count == 0
    assert service.generate_pr_audit("code", [f])[id(f)]["fix"] == "fixed_code"


def test_analyze_issue_safe_parses_explanation_and_fix():
    provider = _SuccessProvider()
    service = LLMService(provider=provider, enable_llm=True, max_calls=5)

    f = _finding()
    result = service.generate_pr_audit("code", [f])

    assert result[id(f)]["explanation"] == "explanation"
    assert result[id(f)]["fix"] == "fixed_code"
    assert service.call_count == 1
    assert provider.calls == 1


def test_llm_service_threads_base_url_and_model_to_provider():
    # When no explicit provider is given, base_url/model flow through to NIMProvider,
    # so switching providers is a config change (client construction makes no network call).
    service = LLMService(enable_llm=True, api_key="k", base_url="http://x/v1", model="custom-model")
    assert service.provider is not None
    assert service.provider.base_url == "http://x/v1"
    assert service.provider.model == "custom-model"


def test_llm_service_uses_provider_defaults_when_base_url_model_omitted():
    # Omitting them preserves NIMProvider's NVIDIA class defaults (back-compat).
    service = LLMService(enable_llm=True, api_key="k")
    assert service.provider.base_url == "https://integrate.api.nvidia.com/v1"
    assert service.provider.model == "nvidia/nemotron-3.5-lightning-30b-a3b"


class _MarkdownProvider:
    """A model that wraps labels in markdown bold, adds a numbered header, and fences code."""

    def generate_pr_audit(self, code: str, summary: str):
        _ = code
        _ = summary
        return (
            "**Issue 1: hardcoded_secret**\n\n"
            "**Issue:** Hardcoded secret password\n"
            "**Explanation:** The password is hardcoded, which is a real security risk.\n"
            "**Fix:**\n```python\npassword = os.environ['PW']\n```\n"
        )


def test_parser_handles_markdown_preamble_and_code_fences():
    # Real-world failure mode: model answers correctly but with markdown + a numbered
    # header before the first label — the old index-based parser dropped it to fallback.
    service = LLMService(provider=_MarkdownProvider(), enable_llm=True, max_calls=5)
    f = _finding()
    result = service.generate_pr_audit("code", [f])[id(f)]

    assert result["explanation"] != LLMService.FALLBACK_EXPLANATION
    assert "hardcoded" in result["explanation"].lower()
    assert result["fix"] != LLMService.FALLBACK_FIX
    assert "os.environ" in result["fix"]
    assert "```" not in result["fix"]  # wrapping code fence stripped
    assert "**" not in result["explanation"]  # markdown emphasis stripped


class _MultiIssueProvider:
    def generate_pr_audit(self, code: str, summary: str):
        _ = code
        _ = summary
        return (
            "Issue: one\nExplanation: first explanation\nFix: fix_one\n"
            "Issue: two\nExplanation: second explanation\nFix: fix_two\n"
        )


def test_parser_maps_multiple_issues_in_order():
    service = LLMService(provider=_MultiIssueProvider(), enable_llm=True, max_calls=5)
    f1 = _finding(rule="r1")
    f2 = _finding(rule="r2")
    out = service.generate_pr_audit("code", [f1, f2])

    assert out[id(f1)] == {"explanation": "first explanation", "fix": "fix_one"}
    assert out[id(f2)] == {"explanation": "second explanation", "fix": "fix_two"}


def test_parser_falls_back_per_field_when_labels_missing():
    # No parseable labels at all -> both fields fall back, no crash.
    class _GarbleProvider:
        def generate_pr_audit(self, code: str, summary: str):
            _ = code
            _ = summary
            return "here is some prose with no structured labels whatsoever"

    service = LLMService(provider=_GarbleProvider(), enable_llm=True, max_calls=5)
    f = _finding()
    result = service.generate_pr_audit("code", [f])[id(f)]
    assert result["explanation"] == LLMService.FALLBACK_EXPLANATION
    assert result["fix"] == LLMService.FALLBACK_FIX
