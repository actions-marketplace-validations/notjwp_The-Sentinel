"""AST-based complexity — what the regex debt engine cannot see.

``DebtService.calculate_complexity`` counts decision keywords across whatever
text it is handed. Over a PR's added lines that means summing every branch in
every changed function, so the number grows with the size of the diff rather
than the tangle of any part of it. Measured on this repository: a 1200-line
diff scored 201, and roughly 20 added lines of ordinary code already cleared
the HIGH threshold of 15. "Complexity: 201" told a reader nothing except that
the pull request was large, which they could already see.

Cyclomatic complexity is a property of ONE unit of code. This analyzer scores
each function separately and reports the worst, so a PR that adds twenty simple
functions scores like a simple function, and one that adds a single twenty-branch
monster scores like a monster. That is the number people think they are reading.

The split mirrors ``ast_security_service``: structure where the source parses,
regex where it does not. Added lines on their own rarely parse — an indented
method body is not a module — so the regex engine remains the fallback and is
not going away.

Pure domain code: stdlib ``ast`` only, no I/O, no infrastructure.
"""

import ast

# Nodes that each introduce one independent path through a unit of code.
# `else` and `elif` are deliberately absent: an `elif` is an `If` inside the
# parent's orelse and is counted there, and a bare `else` adds no new decision —
# the branch it belongs to was already counted by its `if`.
_BRANCH_NODES: tuple[type[ast.AST], ...] = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ExceptHandler,
    ast.IfExp,
    ast.Assert,
    ast.match_case,
)

# A unit of code scored on its own rather than folded into its parent.
_SCOPE_NODES: tuple[type[ast.AST], ...] = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
)


class ASTDebtService:
    """Worst-function cyclomatic complexity for a single Python source file."""

    def analyze_source(
        self,
        source: str | None,
        filename: str = "",
        touched_lines: set[int] | None = None,
    ) -> int | None:
        """The highest complexity of any function, or None when it does not parse.

        None is deliberately distinct from 1: "I could not look" and "I looked
        and found nothing branching" send the caller different ways — the first
        falls back to the regex engine, the second is a real, low score.

        ``touched_lines`` narrows the answer to functions the pull request
        actually edited, the same restriction ``ASTSecurityService`` applies to
        its findings and for the same reason: a function somebody else made
        complex is accurate to report and none of this author's business. Pass
        None to score the whole file.

        ``filename`` is accepted for symmetry with ``ASTSecurityService`` and for
        the caller's logging; the score does not depend on it.
        """
        _ = filename
        if not isinstance(source, str) or not source.strip():
            return None
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError, RecursionError):
            return None

        # The module itself counts as a unit, so a file of straight-line
        # top-level code still scores 1 rather than coming back empty.
        worst = 1 if touched_lines is not None else self._score_unit(tree)
        for node in ast.walk(tree):
            if not isinstance(node, _SCOPE_NODES):
                continue
            if touched_lines is not None and not self._touches(node, touched_lines):
                continue
            worst = max(worst, self._score_unit(node))
        return worst

    @staticmethod
    def _touches(node: ast.AST, touched_lines: set[int]) -> bool:
        """True when any edited line falls inside this function's span.

        ``end_lineno`` has been populated since 3.8 and this project requires
        3.13, but it is Optional in the type stubs, so the start line is the
        fallback rather than an assumption.
        """
        start = getattr(node, "lineno", None)
        if not isinstance(start, int):
            return False
        end = getattr(node, "end_lineno", None) or start
        return any(start <= line <= end for line in touched_lines)

    @staticmethod
    def _score_unit(unit: ast.AST) -> int:
        """Cyclomatic complexity of one scope, excluding nested function bodies.

        The walk stops at a nested ``def`` because that function is scored as a
        unit in its own right. Folding a helper's branches into its parent would
        make an outer function look complex merely for containing a complex inner
        one — the same summing mistake this class exists to undo, one level down.
        """
        complexity = 1
        stack: list[ast.AST] = list(ast.iter_child_nodes(unit))
        while stack:
            node = stack.pop()
            if isinstance(node, _SCOPE_NODES):
                continue
            if isinstance(node, _BRANCH_NODES):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                # `a and b and c` is two decisions, not one: every extra operand
                # is another way through. `values` always holds at least two.
                complexity += len(node.values) - 1
            elif isinstance(node, ast.comprehension):
                # The comprehension's own `for` is not an ast.For, so it is
                # counted here along with each of its filters.
                complexity += 1 + len(node.ifs)
            stack.extend(ast.iter_child_nodes(node))
        return complexity

    def can_analyze(self, source: str | None) -> bool:
        return self.analyze_source(source) is not None
