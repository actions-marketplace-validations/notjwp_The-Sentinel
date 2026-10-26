from dataclasses import replace
from typing import Protocol

from sentinel.config.settings import get_settings
from sentinel.domain.entities.finding import Finding
from sentinel.domain.value_objects.severity_level import SeverityLevel
from sentinel.monitoring.logger import get_logger

logger = get_logger(__name__)

# The threshold value meaning "no severity should fail the build". Lives here
# because the check-run conclusion is what honours it; the Action canonicalises
# its own input down to this or a severity name.
NEVER_FAILS = "NEVER"


class JobQueuePort(Protocol):
    """What the orchestrator needs from a queue: enqueue only.

    Satisfied by both the in-memory JobQueue and the durable RedisJobQueue;
    which one is wired in is the composition root's choice (REDIS_URL).
    """

    async def enqueue(self, job: dict) -> None: ...


class LLMServicePort(Protocol):
    def reset_budget(self) -> None: ...

    def generate_pr_audit(self, code: str, findings: list) -> dict[int, dict[str, str]]: ...


class DocumentServicePort(Protocol):
    def analyze(
        self,
        files: list[str],
        file_contents: dict[str, str] | None = None,
        *,
        enable_llm_review: bool = False,
        llm_reviewer: object | None = None,
    ) -> list[Finding]: ...

    def analyze_code(
        self,
        code: str,
        *,
        source_label: str = "inline",
    ) -> list[Finding]: ...


class AuditOrchestrator:
    def __init__(
        self,
        queue: JobQueuePort,
        llm_service: LLMServicePort | None = None,
        document_service: DocumentServicePort | None = None,
    ) -> None:
        self.queue = queue
        self.llm_service = llm_service
        self.document_service = document_service

    async def enqueue_pull_request(self, payload: dict) -> None:
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dictionary")

        repo = payload.get("repo")
        pr_number = payload.get("pr_number")

        job = {
            "repo": repo,
            "pr_number": pr_number,
        }

        if "owner" in payload:
            job["owner"] = payload["owner"]
        # Which installation's credentials the worker must use. Carried on the
        # job rather than read from the environment, because one deployment
        # serves many installations.
        if "installation_id" in payload:
            job["installation_id"] = payload["installation_id"]
        if "author" in payload:
            job["author"] = payload["author"]
        if "files" in payload:
            job["files"] = payload["files"]
        if "code" in payload:
            job["code"] = payload["code"]

        await self.queue.enqueue(job)
        logger.info("Queued PR audit job for repo=%s pr_number=%s", repo, pr_number)

    def enrich_findings_with_llm(
        self, code: str, findings: list[Finding], *, allow_llm: bool = True
    ) -> list[Finding]:
        settings = get_settings()
        if not findings:
            return []
        # allow_llm is the per-review budget decision (see RedisLLMQuota); the
        # settings flag is the deployment-wide one. Either can veto.
        if not allow_llm or not settings.ENABLE_LLM or self.llm_service is None:
            return findings

        self.llm_service.reset_budget()
        logger.info("LLM enrichment enabled for %s findings", len(findings))

        audit_response = self.llm_service.generate_pr_audit(code=code, findings=findings)

        enriched: list[Finding] = []
        for finding in findings:
            fid = id(finding)
            if fid in audit_response:
                enriched.append(
                    replace(
                        finding,
                        explanation=audit_response[fid]["explanation"],
                        fix_suggestion=audit_response[fid]["fix"],
                    )
                )
            else:
                enriched.append(finding)

        logger.info("LLM calls made in request: %s", getattr(self.llm_service, "call_count", "unknown"))
        return enriched

    def collect_document_findings(
        self,
        files: list[str] | None,
        *,
        file_contents: dict[str, str] | None = None,
        code: str | None = None,
    ) -> list[Finding]:
        settings = get_settings()
        if not settings.ENABLE_DOC_REVIEW:
            return []
        if self.document_service is None:
            return []

        findings: list[Finding] = []

        if files:
            try:
                findings.extend(
                    self.document_service.analyze(
                        files,
                        file_contents=file_contents,
                        enable_llm_review=False,
                        llm_reviewer=None,
                    )
                )
            except Exception:
                logger.exception("Document file review failed; continuing")

        if code and isinstance(code, str) and code.strip():
            try:
                findings.extend(
                    self.document_service.analyze_code(code)
                )
            except Exception:
                logger.exception("Document code review failed; continuing")

        logger.info("Document review produced %s findings", len(findings))
        return findings

    def build_report(
        self,
        findings: list[Finding],
        risk: SeverityLevel | str,
        *,
        complexity: int | None = None,
        maintainability: float | None = None,
        semantic_findings_count: int | None = None,
    ) -> str:
        risk_value = risk.value if isinstance(risk, SeverityLevel) else str(risk).upper()
        try:
            lines: list[str] = [
                "# Sentinel AI Code Review",
                "",
                f"## Risk Score: {risk_value}",
                "",
                "## Security Issues",
            ]
            security_findings = [finding for finding in findings if finding.type == "security"]
            if security_findings:
                for finding in security_findings:
                    lines.append(
                        f"- {finding.description or finding.rule} (Severity: {self._severity_value(finding.severity)})"
                    )
            else:
                lines.append("- No security issues detected.")

            lines.extend(["", "## Explanation"])
            explanations = [finding.explanation for finding in security_findings if finding.explanation]
            if explanations:
                for explanation in explanations:
                    lines.append(f"- {explanation}")
            else:
                lines.append("- No AI explanation available.")

            lines.extend(["", "## Fix Suggestion"])
            # The engine's own recommendation stands in when the model was
            # never consulted (no key, enrichment disabled, budget spent). It
            # is rule-specific and correct, where "no fix suggestion
            # available" is just an empty section.
            fixes = [
                finding.fix_suggestion or finding.recommendation
                for finding in security_findings
                if finding.fix_suggestion or finding.recommendation
            ]
            if fixes:
                for fix in fixes:
                    lines.extend(["```python", fix, "```"])
            else:
                lines.append("- No fix suggestion available.")

            documentation_findings = [finding for finding in findings if finding.type == "documentation"]
            if documentation_findings:
                lines.extend(["", "## Documentation Issues"])
                for finding in documentation_findings:
                    lines.append(
                        f"- {finding.description or finding.rule} (Severity: {self._severity_value(finding.severity)})"
                    )

            lines.extend(["", "## Technical Debt"])
            if complexity is None and maintainability is None:
                lines.append("- Technical debt metrics unavailable.")
            else:
                if complexity is not None:
                    lines.append(f"- Complexity: {complexity}")
                if maintainability is not None:
                    lines.append(f"- Maintainability: {maintainability:.2f}")

            lines.extend(["", "## Semantic Similarity"])
            if semantic_findings_count is None:
                lines.append("- Semantic similarity metrics unavailable.")
            else:
                lines.append(f"- Similar findings detected: {semantic_findings_count}")

            return "\n".join(lines).strip()
        except Exception:
            logger.exception("Report formatting failed; returning safe fallback")
            return f"Sentinel AI Code Review\n\nRisk Score: {risk_value}"

    # Checks API mappings: only CRITICAL/HIGH block a protected merge; MEDIUM is
    # informational (neutral). Annotation levels follow the same shape.
    _CHECK_CONCLUSIONS = {"CRITICAL": "failure", "HIGH": "failure", "MEDIUM": "neutral"}
    _ANNOTATION_LEVELS = {"CRITICAL": "failure", "HIGH": "failure", "MEDIUM": "warning"}
    MAX_CHECK_ANNOTATIONS = 50

    @staticmethod
    def _severity_value(severity: SeverityLevel | str | None) -> str:
        if isinstance(severity, SeverityLevel):
            return severity.value
        return str(severity).upper() if severity else "LOW"

    def build_check_payload(
        self,
        findings: list[Finding],
        risk: SeverityLevel | str,
        *,
        line_map: list | None = None,
        semantic_findings_count: int | None = None,
        fail_on: SeverityLevel | str | None = None,
    ) -> dict:
        """Shape a review into a Checks API payload: conclusion, title, summary, annotations.

        Security findings carry line numbers relative to the assembled added-lines
        code; ``line_map`` (one ``(path, head_line)`` pair per code line, lists after
        a Redis round-trip) resolves them to real file locations. Documentation
        findings carry their own file/line. Semantic findings have no location and
        stay summary-only. Failure-safe: annotation building never raises.
        """
        risk_value = self._severity_value(risk)
        conclusion = self._CHECK_CONCLUSIONS.get(risk_value, "success")

        # When the caller has declared what severity should fail the build, a
        # finding below it must not post a red check: a failing check beside a
        # passing job is a contradiction the reader has to resolve, and they
        # resolve it by distrusting the check. Only ever downgrades.
        if conclusion == "failure":
            if isinstance(fail_on, str) and fail_on.strip().upper() == NEVER_FAILS:
                conclusion = "neutral"
            else:
                threshold = SeverityLevel.rank_of(fail_on)
                reached = SeverityLevel.rank_of(risk_value)
                if threshold is not None and reached is not None and reached < threshold:
                    conclusion = "neutral"

        security = [f for f in findings if getattr(f, "type", None) == "security"]
        documentation = [f for f in findings if getattr(f, "type", None) == "documentation"]
        title = (
            f"Risk: {risk_value} — {len(security)} security, "
            f"{len(documentation)} documentation issue(s)"
        )
        summary_lines = [
            f"**Risk score:** {risk_value}",
            f"- Security findings: {len(security)}",
            f"- Documentation findings: {len(documentation)}",
        ]
        if semantic_findings_count is not None:
            summary_lines.append(f"- Semantic duplicates: {semantic_findings_count}")
        summary = "\n".join(summary_lines)

        annotations: list[dict] = []
        try:
            for finding in findings:
                level = self._ANNOTATION_LEVELS.get(
                    self._severity_value(finding.severity), "notice"
                )
                path: str | None = None
                line: int | None = None
                # A finding that already knows its own file needs no translation.
                # Documentation findings always have; AST security findings now do
                # too, because they are produced per-file against full source. Only
                # regex security findings carry a line relative to the assembled
                # added-lines blob, and those still resolve through line_map.
                if (
                    finding.file
                    and finding.file != "inline"  # analyze_code's label, not a real file
                    and isinstance(finding.line, int)
                ):
                    path, line = finding.file, finding.line
                elif (
                    finding.type == "security"
                    and isinstance(finding.line, int)
                    and line_map
                    and 1 <= finding.line <= len(line_map)
                ):
                    path, line = line_map[finding.line - 1]
                if not isinstance(path, str) or not path or not isinstance(line, int):
                    continue
                message = finding.description or finding.rule
                if finding.recommendation:
                    message = f"{message}\n\n{finding.recommendation}"
                annotations.append(
                    {
                        "path": path,
                        "start_line": line,
                        "end_line": line,
                        "annotation_level": level,
                        "message": message,
                        "title": finding.rule,
                    }
                )
        except Exception:
            logger.exception("Check annotation building failed; posting without annotations")
            annotations = []

        overflow = len(annotations) - self.MAX_CHECK_ANNOTATIONS
        if overflow > 0:
            annotations = annotations[: self.MAX_CHECK_ANNOTATIONS]
            summary += f"\n\n(+{overflow} more annotation(s) truncated — API limit is 50.)"

        return {
            "conclusion": conclusion,
            "title": title,
            "summary": summary,
            "annotations": annotations,
        }

    def append_translations(self, report: str, languages: list[str] | None = None) -> str:
        settings = get_settings()
        if not settings.ENABLE_TRANSLATION or self.llm_service is None:
            return report

        SUPPORTED_LANGUAGES = {
            "hindi": "Hindi",
            "kannada": "Kannada",
            "tamil": "Tamil",
            "telugu": "Telugu",
        }

        selected_languages = languages or ["Hindi", "Kannada"]

        # Translation shares LLM_MAX_CALLS with enrichment, which runs first and
        # spends the budget. At the default of 1 that means translations never
        # run — and used to do so in complete silence, looking like a broken
        # feature rather than a configuration consequence. Say so instead, and
        # say exactly what value would fix it.
        # Only a service that actually reports a budget can be judged against
        # one; a stand-in without these attributes proceeds as before.
        spent = getattr(self.llm_service, "call_count", None)
        budget = getattr(self.llm_service, "max_calls", None)
        if isinstance(spent, int) and isinstance(budget, int) and budget - spent < 1:
            logger.warning(
                "ENABLE_TRANSLATION is on but the LLM budget is spent "
                "(LLM_MAX_CALLS=%s, already used=%s); skipping all translations. "
                "Set LLM_MAX_CALLS to at least %s — 1 for enrichment plus 1 per language.",
                budget,
                spent,
                spent + len(selected_languages),
            )
            return report

        translated_sections: list[str] = []

        for language in selected_languages:
            normalized_language = language.strip().lower()
            target_language = SUPPORTED_LANGUAGES.get(normalized_language)
            if target_language is None:
                continue

            try:
                instruction = (
                    "Translate the provided markdown report into "
                    f"{target_language}. Preserve headings, bullets, and code blocks."
                )
                if hasattr(self.llm_service, "explain_issue_safe"):
                    translated = self.llm_service.explain_issue_safe(
                        report,
                        instruction,
                        severity="HIGH",
                    )
                else:
                    return report
            except Exception:
                logger.exception("Translation failed for language=%s; skipping", language)
                continue

            fallback = getattr(
                self.llm_service,
                "FALLBACK_EXPLANATION",
                "Potential security issue detected. Review code manually.",
            )
            if not translated or translated == fallback or translated.strip() == "":
                continue

            translated_sections.append(f"## {target_language} Version\n\n{translated.strip()}")

        if not translated_sections:
            return report

        return report + "\n\n" + "\n\n".join(translated_sections)

    def run_full_review(
        self,
        *,
        code: str,
        findings: list[Finding],
        risk: SeverityLevel | str,
        files: list[str] | None = None,
        file_contents: dict[str, str] | None = None,
        complexity: int | None = None,
        maintainability: float | None = None,
        semantic_findings_count: int | None = None,
        allow_llm: bool = True,
    ) -> tuple[list[Finding], str]:
        security_findings = self.enrich_findings_with_llm(code, findings, allow_llm=allow_llm)
        document_findings = self.collect_document_findings(files, file_contents=file_contents, code=code)
        all_findings = [*security_findings, *document_findings]

        report = self.build_report(
            all_findings,
            risk,
            complexity=complexity,
            maintainability=maintainability,
            semantic_findings_count=semantic_findings_count,
        )
        report = self.append_translations(report)
        return all_findings, report
