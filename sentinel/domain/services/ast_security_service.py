"""AST-based security analysis — precise where the regex engine is blind.

The regex engine (``security_service``) matches raw text, so it fires inside
comments, docstrings, and ordinary strings that merely mention a rule's words,
and it cannot tell ``os.system(user_input)`` from ``os.system("ls")``. It earns
its place by tolerating input that isn't valid Python, which is exactly what a
diff's added lines usually are.

This analyzer is the other half of that trade: it needs a file that parses, and
in exchange it sees structure. A match inside a comment is invisible to it
because comments aren't in the tree at all; a hardcoded secret is distinguished
from a value read out of the environment because one is a Constant and the other
is a Call.

Pure domain code: stdlib ``ast`` only, no I/O, no infrastructure. The caller
decides which files it can be applied to — see ``analyze_source``, which reports
unparseable input rather than guessing.
"""

import ast
import re

from sentinel.domain.entities.finding import Finding
from sentinel.domain.services.security_service import SecurityService
from sentinel.domain.value_objects.severity_level import SeverityLevel

# Assignment targets whose name implies the value is a credential. Matched as a
# whole word or an underscore-separated part, so `password` and `db_password`
# both hit while `password_field_label` does not.
_SECRET_NAME_PARTS = frozenset(
    {"password", "passwd", "pwd", "secret", "token", "apikey", "api_key", "credential"}
)

# Credentials recognizable by their SHAPE, whatever the variable is called.
# Necessary because the name heuristic alone cannot cover everything: `aws_key`
# is a credential but `key` is too generic a word to treat as one (sort_key,
# cache_key), so without this an AWS literal would slip through — and since a
# file handled here is withheld from the regex engine entirely, slipping through
# means detected by nothing at all.
_CREDENTIAL_SHAPES: tuple[tuple[str, str, str], ...] = (
    ("aws_access_key", r"AKIA[0-9A-Z]{16}", "AWS access key ID"),
    ("openai_key", r"sk-[A-Za-z0-9]{20,}", "OpenAI-style API key"),
    ("github_token", r"gh[pousr]_[A-Za-z0-9]{20,}", "GitHub token"),
    ("slack_token", r"xox[baprs]-[A-Za-z0-9-]{10,}", "Slack token"),
    ("private_key_block", r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", "private key"),
)
_CREDENTIAL_PATTERNS = tuple(
    (rule, re.compile(pattern), label) for rule, pattern, label in _CREDENTIAL_SHAPES
)

# Words that make a string look like SQL rather than prose.
_SQL_VERBS = frozenset({"select", "insert", "update", "delete", "merge"})
_SQL_STRUCTURE = frozenset({"from", "where", "into", "set", "values", "join"})


def _looks_like_sql(text: str) -> bool:
    lowered = text.lower()
    return any(v in lowered for v in _SQL_VERBS) and any(s in lowered for s in _SQL_STRUCTURE)


def _is_secret_name(name: str) -> bool:
    """True when the name denotes a credential rather than merely mentioning one.

    Position carries the meaning. In `db_password` the secret word is the noun,
    so it names a credential; in `password_field_label` it is a modifier and the
    noun is `label`, which names a UI string. Matching the word anywhere would
    flag both, so only the whole name or its trailing segment(s) count.
    """
    lowered = name.lower()
    if lowered in _SECRET_NAME_PARTS or lowered.replace("_", "") in _SECRET_NAME_PARTS:
        return True
    parts = lowered.split("_")
    if parts[-1] in _SECRET_NAME_PARTS:
        return True
    # Two-word credentials written with a separator: api_key, access_token.
    return len(parts) >= 2 and "_".join(parts[-2:]) in _SECRET_NAME_PARTS


def _target_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


class _SecurityVisitor(ast.NodeVisitor):
    """Walks a module and records what the structure — not the text — reveals."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.findings: list[Finding] = []
        # Lines already reported, so one credential is one finding. Without this
        # `api_key = "sk-..."` would be caught twice: once by its name and once
        # by its shape — the very double-reporting this engine exists to avoid.
        self._reported_lines: set[int] = set()

    # --- helpers ---------------------------------------------------------

    def _add(
        self,
        node: ast.AST,
        rule: str,
        severity: SeverityLevel,
        vulnerability_type: str,
        description: str,
        recommendation: str,
        match: str,
    ) -> None:
        category, owasp = SecurityService.VULNERABILITY_CLASSIFICATION.get(
            vulnerability_type, ("Unknown", "Unknown")
        )
        line_no = getattr(node, "lineno", None)
        if isinstance(line_no, int):
            self._reported_lines.add(line_no)
        self.findings.append(
            Finding(
                rule=rule,
                match=match[:200],
                severity=severity,
                category=category,
                owasp_category=owasp,
                description=description,
                file=self.filename,
                line=getattr(node, "lineno", None),
                recommendation=recommendation,
            )
        )

    @staticmethod
    def _dotted(node: ast.AST) -> str:
        """Best-effort dotted name for a call target (os.system, subprocess.run)."""
        parts: list[str] = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))

    # --- hardcoded credentials -------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            name = _target_name(target)
            if not name or not _is_secret_name(name):
                continue
            # A Constant string is a literal in the source. Anything else — a
            # call to os.environ.get, a name, an f-string — is a value fetched
            # at run time, which is the correct pattern rather than a finding.
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                if not node.value.value.strip():
                    continue  # empty placeholder, e.g. `password = ""`
                self._add(
                    node,
                    rule="hardcoded_secret",
                    severity=SeverityLevel.HIGH,
                    vulnerability_type="hardcoded_secret",
                    description=f"Credential '{name}' is assigned a literal value in source.",
                    recommendation="Load it from the environment or a secrets manager.",
                    match=f"{name} = <literal>",
                )
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        """A literal that LOOKS like a credential, wherever it appears.

        Catches what the name heuristic cannot: a key passed straight to a
        function, stored in a dict, or bound to an innocuously-named variable.
        Skips lines already reported so a named credential is not counted twice.
        """
        if not isinstance(node.value, str) or len(node.value) < 12:
            self.generic_visit(node)
            return
        if getattr(node, "lineno", None) in self._reported_lines:
            self.generic_visit(node)
            return

        for rule, pattern, label in _CREDENTIAL_PATTERNS:
            if pattern.search(node.value):
                self._add(
                    node,
                    rule=rule,
                    severity=SeverityLevel.HIGH,
                    vulnerability_type="hardcoded_secret",
                    description=f"Hardcoded {label} found in source.",
                    recommendation="Rotate it and load the value from a secure runtime source.",
                    match=f"<{label}>",
                )
                break
        self.generic_visit(node)

    # --- SQL built by interpolation --------------------------------------

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        """An f-string is only a risk when it BOTH looks like SQL and interpolates."""
        literal = "".join(
            part.value for part in node.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
        interpolates = any(isinstance(part, ast.FormattedValue) for part in node.values)
        if interpolates and _looks_like_sql(literal):
            self._add(
                node,
                rule="sql_injection",
                severity=SeverityLevel.CRITICAL,
                vulnerability_type="sql_injection",
                description="SQL query built by f-string interpolation.",
                recommendation="Use parameterized queries with bound parameters.",
                match=literal.strip()[:120],
            )
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """`"SELECT ... " + var` and `"SELECT ... %s" % var` are the same hazard."""
        if isinstance(node.op, (ast.Add, ast.Mod)):
            left = node.left
            if isinstance(left, ast.Constant) and isinstance(left.value, str):
                if _looks_like_sql(left.value) and not isinstance(node.right, ast.Constant):
                    self._add(
                        node,
                        rule="sql_injection",
                        severity=SeverityLevel.CRITICAL,
                        vulnerability_type="sql_injection",
                        description="SQL query assembled from a non-literal value.",
                        recommendation="Use parameterized queries with bound parameters.",
                        match=left.value.strip()[:120],
                    )
        self.generic_visit(node)

    # --- dynamic execution ------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        dotted = self._dotted(node.func)

        if dotted in ("eval", "exec"):
            # A literal argument is not attacker-controlled; a name or call is.
            dynamic = node.args and not isinstance(node.args[0], ast.Constant)
            self._add(
                node,
                rule=f"{dotted}_call",
                severity=SeverityLevel.HIGH if dynamic else SeverityLevel.MEDIUM,
                vulnerability_type="dangerous_code_execution",
                description=(
                    f"{dotted}() called with a non-literal argument."
                    if dynamic else f"Use of {dotted}() executes code at run time."
                ),
                recommendation="Replace with a safe parser or an explicit dispatch table.",
                match=f"{dotted}(...)",
            )

        elif dotted == "os.system":
            dynamic = node.args and not isinstance(node.args[0], ast.Constant)
            self._add(
                node,
                rule="os_system_call",
                severity=SeverityLevel.CRITICAL if dynamic else SeverityLevel.MEDIUM,
                vulnerability_type=(
                    "command_injection" if dynamic else "security_misconfiguration"
                ),
                description=(
                    "os.system() invoked with a non-literal command."
                    if dynamic else "os.system() runs a shell."
                ),
                recommendation="Use subprocess.run([...]) with an argument list and shell=False.",
                match="os.system(...)",
            )

        elif dotted.startswith("subprocess."):
            shell_true = any(
                kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True
                for kw in node.keywords
            )
            if shell_true:
                dynamic = node.args and not isinstance(node.args[0], ast.Constant)
                self._add(
                    node,
                    rule="subprocess_shell_true",
                    severity=SeverityLevel.CRITICAL if dynamic else SeverityLevel.MEDIUM,
                    vulnerability_type=(
                        "command_injection" if dynamic else "security_misconfiguration"
                    ),
                    description=(
                        f"{dotted}() with shell=True and a non-literal command."
                        if dynamic else f"{dotted}() with shell=True."
                    ),
                    recommendation="Pass an argument list and leave shell=False.",
                    match=f"{dotted}(..., shell=True)",
                )

        self.generic_visit(node)


class ASTSecurityService:
    """Structure-aware security analysis for a single, parseable Python file."""

    @staticmethod
    def can_analyze(source: str) -> bool:
        if not isinstance(source, str) or not source.strip():
            return False
        try:
            ast.parse(source)
        except (SyntaxError, ValueError, RecursionError):
            return False
        return True

    def analyze_source(self, source: str, filename: str) -> list[Finding] | None:
        """Findings for one file, or None when the source does not parse.

        None is deliberately distinct from an empty list: "I could not look" and
        "I looked and found nothing" must not be confused, because the caller
        falls back to the regex engine only in the first case.
        """
        if not isinstance(source, str) or not source.strip():
            return None
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError, RecursionError):
            return None

        visitor = _SecurityVisitor(filename)
        visitor.visit(tree)
        return visitor.findings
