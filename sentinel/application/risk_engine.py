import time

from sentinel.domain.services.debt_service import DebtService
from sentinel.domain.services.security_service import SecurityService
from sentinel.domain.services.semantic_service import SemanticService
from sentinel.domain.value_objects.severity_level import SeverityLevel
from sentinel.monitoring.logger import get_logger

logger = get_logger(__name__)


class RiskEngine:
    def __init__(
        self,
        debt_service: DebtService | None = None,
        security_service: SecurityService | None = None,
        semantic_service: SemanticService | None = None,
    ) -> None:
        self.debt_service = debt_service or DebtService()
        self.security_service = security_service or SecurityService()
        self.semantic_service = semantic_service

    @staticmethod
    def _normalize_severity(value: object) -> SeverityLevel:
        if isinstance(value, SeverityLevel):
            return value
        return SeverityLevel.LOW

    def _security_severity_from_result(self, security_result: dict) -> SeverityLevel:
        findings = security_result.get("findings", [])
        finding_severities = [
            self._normalize_severity(getattr(finding, "severity", None)) for finding in findings
        ]

        if any(severity == SeverityLevel.CRITICAL for severity in finding_severities):
            return SeverityLevel.CRITICAL
        if any(severity == SeverityLevel.HIGH for severity in finding_severities):
            return SeverityLevel.HIGH
        if any(severity == SeverityLevel.MEDIUM for severity in finding_severities):
            return SeverityLevel.MEDIUM
        return self._normalize_severity(security_result.get("severity", SeverityLevel.LOW))

    # Debt is capped before it reaches the overall score. Complexity is counted
    # over the whole added-lines blob, so ~20 added lines of ordinary code
    # already passes DebtService's HIGH threshold; letting that alone return
    # HIGH turns the check run red and makes the Action exit non-zero on a
    # review that found nothing, which is the one failure mode guaranteed to
    # get the tool uninstalled. A measurement of the diff may inform a reviewer;
    # it must not block a merge on its own. The uncapped number is unchanged in
    # `complexity`/`maintainability` and still printed in the report.
    DEBT_SEVERITY_CAP = SeverityLevel.MEDIUM

    @classmethod
    def _overall_severity(
        cls,
        debt_severity: SeverityLevel,
        security_severity: SeverityLevel,
        semantic_severity: SeverityLevel,
    ) -> SeverityLevel:
        if debt_severity == SeverityLevel.HIGH:
            debt_severity = cls.DEBT_SEVERITY_CAP
        if security_severity == SeverityLevel.CRITICAL:
            return SeverityLevel.CRITICAL
        if semantic_severity == SeverityLevel.HIGH:
            return SeverityLevel.HIGH
        if security_severity == SeverityLevel.HIGH:
            return SeverityLevel.HIGH
        if security_severity == SeverityLevel.MEDIUM or debt_severity == SeverityLevel.MEDIUM:
            return SeverityLevel.MEDIUM
        return SeverityLevel.LOW

    # Per-engine safe defaults, used only by the resilient variant.
    _DEBT_DEFAULT = {"complexity": 1, "maintainability": 100.0, "severity": SeverityLevel.LOW}
    _SECURITY_DEFAULT = {"findings": [], "severity": SeverityLevel.LOW}

    @staticmethod
    def _run(label: str, call, default, *, resilient: bool):
        """Run one engine. Strict mode propagates; resilient mode logs and degrades."""
        if not resilient:
            return call()
        try:
            return call()
        except Exception:
            logger.exception("%s engine failed; continuing with safe defaults", label)
            return default() if callable(default) else default

    def _assess(
        self,
        code: str,
        existing_code_list: list[str] | None,
        *,
        resilient: bool,
        warn_threshold_seconds: float | None = None,
        extra_security_findings: list | None = None,
        extra_debt_complexity: int | None = None,
    ) -> dict:
        """The single assessment implementation behind assess/assess_resilient.

        The two differ only in whether an engine failure propagates or degrades
        to a safe default, so they share everything else — aggregation, the
        result shape, and the latency warning.
        """
        start_time = time.monotonic()

        # extra_debt_complexity is the AST engine's per-function score, supplied
        # by the caller that fetched the real files — the same arrangement as
        # extra_security_findings below, and for the same reason: only the
        # caller knows which files had full source that parsed.
        #
        # Passed only when there is one. `debt_service` is an injection point, so
        # sending an unconditional keyword would break every implementation
        # written against the old one-argument signature for no gain — there is
        # nothing to say when no file parsed.
        def evaluate_debt() -> dict:
            if extra_debt_complexity is None:
                return self.debt_service.evaluate_debt(code)
            return self.debt_service.evaluate_debt(code, complexity=extra_debt_complexity)

        debt_result = self._run(
            "Debt", evaluate_debt,
            lambda: dict(self._DEBT_DEFAULT), resilient=resilient,
        )
        security_result = self._run(
            "Security", lambda: self.security_service.analyze(code),
            lambda: {"findings": [], "severity": SeverityLevel.LOW}, resilient=resilient,
        )
        # AST findings are produced per-file by the caller (which alone knows
        # which files had parseable full source) and merged here so severity is
        # still aggregated in exactly one place.
        if extra_security_findings:
            security_result = {
                **security_result,
                "findings": [*security_result.get("findings", []), *extra_security_findings],
            }

        semantic_findings: list = []
        if self.semantic_service and code.strip():
            semantic_findings = self._run(
                "Semantic",
                lambda: self.semantic_service.detect_duplicates(code, existing_code_list or []),
                list, resilient=resilient,
            )
        semantic_severity = (
            SeverityLevel.HIGH
            if any(f.severity == SeverityLevel.HIGH for f in semantic_findings)
            else SeverityLevel.LOW
        )

        overall_severity = self._overall_severity(
            debt_severity=self._normalize_severity(debt_result["severity"]),
            security_severity=self._security_severity_from_result(security_result),
            semantic_severity=semantic_severity,
        )

        if warn_threshold_seconds is not None:
            elapsed = time.monotonic() - start_time
            if elapsed > warn_threshold_seconds:
                logger.warning(
                    "Risk assessment exceeded latency target: %.4fs (threshold %.4fs)",
                    elapsed,
                    warn_threshold_seconds,
                )

        return {
            "severity": overall_severity,
            "complexity": debt_result["complexity"],
            "maintainability": debt_result["maintainability"],
            "security_findings_count": len(security_result["findings"]),
            "security": security_result,
            "semantic_findings_count": len(semantic_findings),
            "semantic": {"findings": semantic_findings, "severity": semantic_severity},
        }

    def assess_resilient(
        self,
        code: str = "",
        existing_code_list: list[str] | None = None,
        *,
        warn_threshold_seconds: float = 0.1,
        extra_security_findings: list | None = None,
        extra_debt_complexity: int | None = None,
    ) -> dict:
        """Engine failures degrade to safe defaults. Use for background work."""
        return self._assess(
            code, existing_code_list,
            resilient=True, warn_threshold_seconds=warn_threshold_seconds,
            extra_security_findings=extra_security_findings,
            extra_debt_complexity=extra_debt_complexity,
        )

    def assess(self, code: str = "", existing_code_list: list[str] | None = None) -> dict:
        """Engine failures propagate. Use where the caller must see the error."""
        return self._assess(code, existing_code_list, resilient=False)

    def calculate_risk(
        self,
        pr_number: int,
        code: str = "",
        existing_code_list: list[str] | None = None,
    ) -> SeverityLevel:
        _ = pr_number
        return self.assess(code, existing_code_list)["severity"]