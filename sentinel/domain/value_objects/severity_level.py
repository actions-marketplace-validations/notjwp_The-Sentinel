from enum import Enum


class SeverityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        """Position in the ordering, lowest first.

        Enum member order is not an ordering anyone should rely on, and the
        comparison is needed in two places that must agree — the check run's
        conclusion and the Action's exit code. Disagreement there means a job
        that passes next to a check that fails.
        """
        return _ORDER[self.value]

    @classmethod
    def rank_of(cls, value: "SeverityLevel | str | None") -> int | None:
        """Rank of a severity given as an enum, a string, or neither.

        Returns None for anything unrecognised, so a caller can tell "below the
        threshold" from "not a severity at all" instead of guessing zero.
        """
        if isinstance(value, SeverityLevel):
            return value.rank
        if isinstance(value, str):
            return _ORDER.get(value.strip().upper())
        return None


_ORDER = {
    SeverityLevel.LOW.value: 0,
    SeverityLevel.MEDIUM.value: 1,
    SeverityLevel.HIGH.value: 2,
    SeverityLevel.CRITICAL.value: 3,
}
