from sentinel.domain.services.document_service import DocumentService


class _DocReviewer:
    def explain_issue_safe(self, code: str, issue: str, *, severity=None) -> str:
        _ = code
        _ = issue
        _ = severity
        return "Documentation could be clearer and include more examples."


def test_document_service_detects_empty_readme():
    service = DocumentService()

    findings = service.analyze(["README.md"], {"README.md": ""})

    rules = {finding.rule for finding in findings}
    assert "empty_readme" in rules


def test_document_service_detects_missing_installation_and_usage():
    service = DocumentService()
    content = "This document describes architecture only."

    findings = service.analyze(["guide.md"], {"guide.md": content})

    rules = {finding.rule for finding in findings}
    assert "missing_installation" in rules
    assert "missing_usage" in rules


def test_document_service_ignores_non_document_files():
    service = DocumentService()

    findings = service.analyze(["src/main.py"], {"src/main.py": "print('ok')"})

    assert findings == []


def test_document_service_optional_llm_review_adds_doc_clarity_finding():
    service = DocumentService()
    reviewer = _DocReviewer()

    findings = service.analyze(
        ["README.md"],
        {"README.md": "Installation\nUse pip install.\nUsage\nRun main.py."},
        enable_llm_review=True,
        llm_reviewer=reviewer,
    )

    rules = {finding.rule for finding in findings}
    assert "doc_clarity_review" in rules


def test_document_service_complete_readme_has_no_rule_based_findings():
    service = DocumentService()
    complete = "Installation\nUse pip install sentinel\nUsage\npython main.py\nExample\nrun it"

    findings = service.analyze(["README.md"], {"README.md": complete})

    assert findings == []


def test_document_service_handles_missing_file_contents_and_suffix_lookup():
    service = DocumentService()

    # file_contents is None -> no crash, no findings
    assert service.analyze(["README.md"], None) == []

    # Endswith lookup path should resolve README.md to docs/README.md content
    findings = service.analyze(["README.md"], {"docs/README.md": "No run instructions"})
    rules = {finding.rule for finding in findings}
    assert "missing_installation" in rules
    assert "missing_usage" in rules


def test_document_service_skips_missing_content_and_llm_disabled_path():
    service = DocumentService()
    reviewer = _DocReviewer()

    # File not present in map -> content is unknown and should be skipped
    assert service.analyze(["README.md"], {"OTHER.md": "content"}) == []

    findings = service.analyze(
        ["README.md"],
        {"README.md": "usage only"},
        enable_llm_review=False,
        llm_reviewer=reviewer,
    )
    rules = {finding.rule for finding in findings}
    assert "doc_clarity_review" not in rules


def test_document_service_handles_unexpected_file_types_and_large_strings():
    service = DocumentService()
    very_large = "A" * 100000

    findings = service.analyze(
        ["README.md", 123, None, "notes.txt"],
        {
            "README.md": very_large,
            "notes.txt": "",
        },
    )

    # README with no installation/usage should trigger findings; txt empty is allowed
    rules = {finding.rule for finding in findings}
    assert "missing_installation" in rules


# --- Dependency manifests are not prose --------------------------------------


def test_requirements_files_are_not_reviewed_as_documentation():
    """Found by Sentinel reviewing its own PR — the first to touch requirements.

    `.txt` is a documentation extension, so a dependency pin list was being
    judged for "installation instructions" and "usage guidance": two MEDIUM
    findings per file on every dependency change, enough to carry the review
    to MEDIUM and turn the check yellow.
    """
    service = DocumentService()
    manifests = [
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-action.txt",
        "constraints.txt",
        "CMakeLists.txt",
        "deploy/requirements.txt",
        r"deploy\requirements.txt",
    ]
    for name in manifests:
        assert service._is_document_file(name) is False, name
        assert service.analyze([name], {name: "fastapi==0.121.0\nopenai>=2.6.1,<4\n"}) == []


def test_real_documents_are_still_reviewed():
    service = DocumentService()
    for name in ("README.md", "docs/DEPLOYMENT.md", "NOTES.txt", "guide.txt",
                 "requirements_guide.md"):
        assert service._is_document_file(name) is True, name

    findings = service.analyze(["NOTES.txt"], {"NOTES.txt": "short"})
    assert findings, "a genuine .txt document should still be reviewed"
