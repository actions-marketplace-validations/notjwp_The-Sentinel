import re

from sentinel.domain.value_objects.severity_level import SeverityLevel
from sentinel.infrastructure.llm.base import LLMProvider
from sentinel.infrastructure.llm.nim_provider import NIMProvider
from sentinel.monitoring.logger import get_logger
from sentinel.monitoring.metrics import metrics


class LLMService:
    # Last-resort strings, used only for a finding that carries nothing better.
    # They are generic on purpose, which is also why they must not be the first
    # choice: "use parameterized queries" is actively wrong advice for a
    # hardcoded credential, and wrong advice reads worse than none.
    FALLBACK_FIX = "Review this finding and apply the fix appropriate to the rule."
    FALLBACK_EXPLANATION = "Potential security issue detected. Review code manually."

    # Upper bound on the matched snippet sent per finding: long enough to identify
    # the offending code, short enough that one huge match can't dominate the prompt.
    MAX_MATCH_CHARS = 120

    # Matches an "Issue:/Explanation:/Fix:" label at the start of a line, tolerating
    # markdown emphasis/bullets/headings around it (e.g. "**Explanation:**", "### Fix:",
    # "- Issue:"). A trailing colon is required so code lines like `def fix():` don't match.
    _LABEL_RE = re.compile(
        r"(?im)^[ \t]*[>#*\-_ \t]*(explanation|fix|issue)[ \t]*[*_]*[ \t]*:[ \t]*"
    )

    def __init__(
        self,
        provider: LLMProvider | None = None,
        *,
        enable_llm: bool = False,
        max_calls: int = 1,
        timeout: float = 5.0,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.logger = get_logger(__name__)
        self.provider = provider
        if self.provider is None and enable_llm:
            provider_kwargs: dict[str, object] = {"api_key": api_key, "timeout": timeout}
            if base_url is not None:
                provider_kwargs["base_url"] = base_url
            if model is not None:
                provider_kwargs["model"] = model
            self.provider = NIMProvider(**provider_kwargs)
        self.enable_llm = enable_llm
        self.max_calls = max(0, max_calls)
        self.call_count = 0

    @property
    def calls_made(self) -> int:
        return self.call_count

    def reset_budget(self) -> None:
        self.call_count = 0

    @staticmethod
    def _severity_name(severity: SeverityLevel | str | None) -> str:
        if isinstance(severity, SeverityLevel):
            return severity.value
        if isinstance(severity, str):
            return severity.upper()
        return "LOW"

    @classmethod
    def _summarize_finding(cls, index: int, finding) -> str:
        """One prompt entry: what the rule found, where, and the text that matched.

        The matched snippet is the point. Sent only a rule slug, the model reasons
        about the slug itself — observed in production, it read the rule name
        ``openai_key`` as a variable and "fixed" it with ``openai_key = None``
        instead of addressing the hardcoded key on the line. The snippet is
        whitespace-collapsed to one line (a multi-line match would otherwise break
        the numbered-list shape the response parser relies on) and truncated, so a
        pathological match can't crowd out the other findings.

        Nothing new is disclosed: ``generate_pr_audit`` already sends the full PR
        code in the same request, and the match is a substring of it.
        """
        parts = [f"{index}. [{finding.rule}] {finding.description or finding.rule}".rstrip()]
        parts.append(f"   Severity: {cls._severity_name(finding.severity)}")
        if isinstance(finding.line, int):
            parts.append(f"   Line: {finding.line}")
        match = getattr(finding, "match", None)
        if isinstance(match, str) and match.strip():
            snippet = " ".join(match.split())
            if len(snippet) > cls.MAX_MATCH_CHARS:
                snippet = snippet[: cls.MAX_MATCH_CHARS] + " ...(truncated)"
            parts.append(f"   Matched code: {snippet}")
        if finding.recommendation:
            parts.append(f"   Suggested direction: {finding.recommendation}")
        return "\n".join(parts)

    def _can_invoke(self) -> bool:
        if not self.enable_llm:
            return False
        if self.provider is None:
            return False
        if self.call_count >= self.max_calls:
            return False
        return True

    @staticmethod
    def _strip_markup(text: str) -> str:
        """Trim surrounding whitespace, markdown emphasis, and a wrapping code fence."""
        cleaned = text.strip().strip("*_").strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].lstrip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        return cleaned

    def _parse_pr_audit_response(self, content: str, expected_ids: list[int]) -> dict[int, dict[str, str]]:
        """Map an LLM audit response to {finding_id: {explanation, fix}}.

        Robust to markdown (``**Explanation:**``), code fences, numbered/preamble text
        before the first issue, and models that omit ``Issue:`` between findings. Each
        ``Explanation:`` starts a new record; the following ``Fix:`` attaches to it; the
        records are mapped to ``expected_ids`` positionally, with per-field fallbacks.
        """
        if not expected_ids:
            return {}

        matches = list(self._LABEL_RE.finditer(content))
        records: list[dict[str, str]] = []
        current: dict[str, str] | None = None

        for index, match in enumerate(matches):
            kind = match.group(1).lower()
            segment_start = match.end()
            segment_end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            text = self._strip_markup(content[segment_start:segment_end])

            if kind == "explanation":
                current = {"explanation": text, "fix": ""}
                records.append(current)
            elif kind == "fix":
                if current is None:
                    current = {"explanation": "", "fix": text}
                    records.append(current)
                else:
                    current["fix"] = text
            else:  # "issue" is only a delimiter — the next Explanation opens a new record
                current = None

        result: dict[int, dict[str, str]] = {}
        for idx, fid in enumerate(expected_ids):
            record = records[idx] if idx < len(records) else {}
            explanation = record.get("explanation") or self.FALLBACK_EXPLANATION
            fix = record.get("fix") or self.FALLBACK_FIX
            result[fid] = {"explanation": explanation, "fix": fix}
        return result

    def _static_enrichment(self, finding) -> dict[str, str]:
        """What to say about a finding when the model said nothing.

        Prefers what the engine already knows about THIS finding over the
        generic strings. A hardcoded-credential rule carries "rotate it and
        load the value from a secure runtime source", which is correct and
        specific; the generic fix talks about parameterised queries, which for
        a credential is simply the wrong advice.
        """
        return {
            "explanation": getattr(finding, "description", "") or self.FALLBACK_EXPLANATION,
            "fix": getattr(finding, "recommendation", None) or self.FALLBACK_FIX,
        }

    def explain_issue_safe(
        self,
        code: str,
        issue: str,
        *,
        severity: SeverityLevel | str | None = None,
    ) -> str:
        """One-shot LLM text task (report translation, doc review). Never raises.

        ``issue`` carries the instruction, ``code`` the content it applies to
        (append_translations passes the report + a translate instruction).
        Returns FALLBACK_EXPLANATION when the LLM is disabled, the LLM_MAX_CALLS
        budget is spent, the provider lacks generate_text, or the call fails —
        callers (append_translations) treat that value as "skip the section".
        """
        _ = severity  # part of the DocumentLLMReviewer protocol; unused here
        if not self._can_invoke() or not hasattr(self.provider, "generate_text"):
            self.logger.info("LLM text task skipped; using fallback response.")
            return self.FALLBACK_EXPLANATION

        try:
            self.call_count += 1  # shares the per-request budget with enrichment
            content = self.provider.generate_text(f"{issue}\n\n---\n\n{code}")
        except Exception:
            self.logger.exception("LLM text task failed; using fallback response.")
            metrics.counter_inc("sentinel_llm_calls_total", {"outcome": "fallback"})
            return self.FALLBACK_EXPLANATION

        if not content or not str(content).strip():
            self.logger.warning("LLM text task returned empty content; using fallback.")
            metrics.counter_inc("sentinel_llm_calls_total", {"outcome": "fallback"})
            return self.FALLBACK_EXPLANATION

        metrics.counter_inc("sentinel_llm_calls_total", {"outcome": "success"})
        return str(content).strip()

    def generate_pr_audit(self, code: str, findings: list) -> dict[int, dict[str, str]]:
        enrichable = []
        for finding in findings:
            severity_name = self._severity_name(finding.severity)
            is_security = finding.type == "security"
            is_meaningful_medium = severity_name != "MEDIUM" or bool(finding.recommendation)
            if is_security and severity_name != "LOW" and is_meaningful_medium:
                enrichable.append(finding)

        if not enrichable or not self._can_invoke():
            self.logger.info("LLM skipped; using fallback response.")
            if enrichable:  # a real fallback; nothing-to-enrich is not one
                metrics.counter_inc("sentinel_llm_calls_total", {"outcome": "fallback"})
            return {id(f): self._static_enrichment(f) for f in enrichable}

        findings_summary = "\n\n".join(
            self._summarize_finding(i, f) for i, f in enumerate(enrichable, 1)
        )

        self.logger.info("LLM PR audit started")
        try:
            self.call_count += 1
            content = self.provider.generate_pr_audit(code, findings_summary)
        except Exception:
            self.logger.exception("LLM request failed; using fallback response.")
            self.logger.info("Fallback triggered")
            metrics.counter_inc("sentinel_llm_calls_total", {"outcome": "fallback"})
            return {id(f): self._static_enrichment(f) for f in enrichable}

        if not content or not str(content).strip():
            self.logger.warning("LLM returned empty content; using fallback response.")
            self.logger.info("Fallback triggered")
            metrics.counter_inc("sentinel_llm_calls_total", {"outcome": "fallback"})
            return {id(f): self._static_enrichment(f) for f in enrichable}

        self.logger.info("LLM response received")
        metrics.counter_inc("sentinel_llm_calls_total", {"outcome": "success"})
        parsed = self._parse_pr_audit_response(str(content), [id(f) for f in enrichable])

        self.logger.info(f"Parsed {len(parsed)} issue explanations")
        self.logger.info(f"Total findings enriched: {len(parsed)}")

        return parsed
