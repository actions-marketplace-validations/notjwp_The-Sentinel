from typing import Protocol

from sentinel.domain.entities.finding import Finding
from sentinel.domain.value_objects.severity_level import SeverityLevel


class DocumentLLMReviewer(Protocol):
    def explain_issue_safe(
        self,
        code: str,
        issue: str,
        *,
        severity: SeverityLevel | str | None = None,
    ) -> str: ...


class DocumentService:
    DOC_EXTENSIONS = (".md", ".txt")

    # `.txt` files that are machine-readable manifests rather than prose. A
    # dependency pin list has no installation instructions to be missing, so
    # reviewing one as documentation produced two MEDIUM findings per file on
    # every PR that touched it — enough to carry the whole review to MEDIUM and
    # turn the check yellow. Found by Sentinel reviewing its own pull request,
    # the first one to change a requirements file.
    NON_PROSE_TXT_PREFIXES = ("requirements", "constraints", "cmakelists")

    @staticmethod
    def _is_document_file(file_name: str) -> bool:
        normalized = file_name.strip().lower()
        if not normalized.endswith(DocumentService.DOC_EXTENSIONS):
            return False
        if not normalized.endswith(".txt"):
            return True
        stem = normalized.replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".txt")
        return not stem.startswith(DocumentService.NON_PROSE_TXT_PREFIXES)

    @staticmethod
    def _content_for(file_name: str, file_contents: dict[str, str] | None) -> str | None:
        if not file_contents:
            return None

        if file_name in file_contents:
            return file_contents[file_name]

        lowered_name = file_name.lower()
        for key, value in file_contents.items():
            if key.lower().endswith(lowered_name):
                return value

        return None

    @staticmethod
    def _doc_finding(
        *,
        rule: str,
        severity: SeverityLevel,
        description: str,
        recommendation: str,
        file_name: str,
        explanation: str | None = None,
    ) -> Finding:
        return Finding(
            rule=rule,
            match=file_name,
            severity=severity,
            finding_type="documentation",
            category="Documentation",
            owasp_category="N/A",
            description=description,
            file=file_name,
            line=1,
            recommendation=recommendation,
            explanation=explanation,
        )

    def analyze(
        self,
        files: list[str],
        file_contents: dict[str, str] | None = None,
        *,
        enable_llm_review: bool = False,
        llm_reviewer: DocumentLLMReviewer | None = None,
    ) -> list[Finding]:
        findings: list[Finding] = []

        for file_name in files:
            if not isinstance(file_name, str) or not self._is_document_file(file_name):
                continue

            content = self._content_for(file_name, file_contents)
            if content is None:
                continue
            content_lower = content.lower()
            normalized_file_name = file_name.strip().lower()

            if normalized_file_name.startswith("readme") and content.strip() == "":
                findings.append(
                    self._doc_finding(
                        rule="empty_readme",
                        severity=SeverityLevel.MEDIUM,
                        description="README file is empty.",
                        recommendation="Add installation, usage, and architecture notes to README.",
                        file_name=file_name,
                    )
                )

            if content.strip() != "":
                if "install" not in content_lower and "pip install" not in content_lower:
                    findings.append(
                        self._doc_finding(
                            rule="missing_installation",
                            severity=SeverityLevel.MEDIUM,
                            description="Documentation is missing installation instructions.",
                            recommendation="Add a clear installation section with setup commands.",
                            file_name=file_name,
                        )
                    )

                if (
                    "usage" not in content_lower
                    and "example" not in content_lower
                    and "how to run" not in content_lower
                ):
                    findings.append(
                        self._doc_finding(
                            rule="missing_usage",
                            severity=SeverityLevel.MEDIUM,
                            description="Documentation is missing usage guidance.",
                            recommendation="Add usage examples and common execution workflows.",
                            file_name=file_name,
                        )
                    )

            if enable_llm_review and llm_reviewer is not None and content.strip() != "":
                review = llm_reviewer.explain_issue_safe(
                    content,
                    "Review this documentation for clarity and completeness. Keep feedback concise.",
                    severity=SeverityLevel.MEDIUM,
                )
                fallback_reviews = {
                    "Explanation unavailable",
                    "Potential security issue detected. Review code manually.",
                }
                if review and review not in fallback_reviews:
                    findings.append(
                        self._doc_finding(
                            rule="doc_clarity_review",
                            severity=SeverityLevel.LOW,
                            description="AI review identified potential documentation clarity improvements.",
                            recommendation="Address clarity and completeness gaps highlighted by the review.",
                            file_name=file_name,
                            explanation=review,
                        )
                    )

        return findings

    CODE_EXTENSIONS = (".py", ".js", ".ts", ".java", ".go", ".rs", ".rb", ".c", ".cpp")

    @staticmethod
    def _is_code_file(file_name: str) -> bool:
        normalized = file_name.strip().lower()
        return normalized.endswith(DocumentService.CODE_EXTENSIONS)

    def analyze_code(self, code: str, *, source_label: str = "inline") -> list[Finding]:
        """Analyze a raw code string for documentation quality issues."""
        if not isinstance(code, str) or not code.strip():
            return []

        findings: list[Finding] = []
        lines = code.splitlines()
        non_blank = [line for line in lines if line.strip()]

        # Check for missing module-level docstring
        stripped_lines = [line.strip() for line in lines if line.strip()]
        has_module_docstring = False
        if stripped_lines:
            first_meaningful = stripped_lines[0]
            if first_meaningful.startswith(('"""', "'''", '#')):
                has_module_docstring = True
            elif len(stripped_lines) > 1 and stripped_lines[1].startswith(('"""', "'''")):
                has_module_docstring = True
        if not has_module_docstring and len(non_blank) > 3:
            findings.append(
                self._doc_finding(
                    rule="missing_module_docstring",
                    severity=SeverityLevel.LOW,
                    description="Code is missing a module-level docstring or header comment.",
                    recommendation="Add a module docstring describing the purpose and key components.",
                    file_name=source_label,
                )
            )

        # Check for missing function/class docstrings
        import re
        def_pattern = re.compile(r"^\s*(def|class)\s+\w+")
        def_count = 0
        documented_def_count = 0
        for i, line in enumerate(lines):
            if def_pattern.match(line):
                def_count += 1
                # Check next non-blank lines for docstring
                for j in range(i + 1, min(i + 4, len(lines))):
                    stripped = lines[j].strip()
                    if stripped == "":
                        continue
                    if stripped.startswith(('"""', "'''", '#')):
                        documented_def_count += 1
                    break

        if def_count > 0 and documented_def_count == 0:
            findings.append(
                self._doc_finding(
                    rule="no_function_docstrings",
                    severity=SeverityLevel.MEDIUM,
                    description=f"None of the {def_count} function(s)/class(es) have docstrings.",
                    recommendation="Add docstrings to functions and classes describing their purpose and parameters.",
                    file_name=source_label,
                )
            )
        elif def_count > 2 and documented_def_count < def_count // 2:
            findings.append(
                self._doc_finding(
                    rule="low_docstring_coverage",
                    severity=SeverityLevel.LOW,
                    description=f"Only {documented_def_count}/{def_count} functions/classes have docstrings.",
                    recommendation="Improve docstring coverage to at least 50% of public functions and classes.",
                    file_name=source_label,
                )
            )

        # Check for very low comment density in non-trivial code
        comment_lines = sum(1 for line in lines if line.strip().startswith("#"))
        if len(non_blank) > 10 and comment_lines == 0:
            findings.append(
                self._doc_finding(
                    rule="no_inline_comments",
                    severity=SeverityLevel.LOW,
                    description="Code has no inline comments despite being non-trivial.",
                    recommendation="Add comments explaining complex logic, edge cases, and non-obvious decisions.",
                    file_name=source_label,
                )
            )

        return findings
