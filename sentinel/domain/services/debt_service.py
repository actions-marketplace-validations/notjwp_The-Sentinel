import re

from sentinel.domain.value_objects.severity_level import SeverityLevel


class DebtService:
    def calculate_complexity(self, code: str) -> int:
        keywords = ["if", "elif", "for", "while", "except", "and", "or", "case"]
        decision_points = 0
        for keyword in keywords:
            decision_points += len(re.findall(rf"\b{keyword}\b", code))
        return max(1, 1 + decision_points)

    def calculate_maintainability(self, complexity: int, loc: int) -> float:
        """A rough composite of how branchy the code is and how much of it there is.

        Note the two terms can come from different scopes: when the caller
        supplies an AST-derived ``complexity`` it describes the worst single
        function, while ``loc`` still counts the whole diff. The formula was
        always a heuristic rather than a measurement, and mixing the two beats
        inventing a second number nobody asked for.
        """
        score = 100 - (complexity * 5) - (loc * 0.1)
        clamped_score = max(0.0, min(100.0, float(score)))
        return round(clamped_score, 2)

    def evaluate_debt(self, code: str, *, complexity: int | None = None) -> dict:
        """Debt metrics for a blob of code.

        ``complexity`` lets a caller that parsed the real files supply a
        per-function cyclomatic score (see ``ASTDebtService``) instead of the
        keyword count, which sums every branch in the diff and so tracks its
        size. Only the caller knows whether any file parsed, which is why the
        decision lives there and not here.
        """
        if complexity is None:
            complexity = self.calculate_complexity(code)
        loc = sum(1 for line in code.splitlines() if line.strip())
        maintainability = self.calculate_maintainability(complexity, loc)

        if complexity > 15:
            severity = SeverityLevel.HIGH
        elif 8 <= complexity <= 15:
            severity = SeverityLevel.MEDIUM
        else:
            severity = SeverityLevel.LOW

        return {
            "complexity": complexity,
            "maintainability": maintainability,
            "severity": severity,
        }
