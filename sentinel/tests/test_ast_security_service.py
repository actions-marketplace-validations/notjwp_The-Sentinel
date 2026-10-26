"""AST security analysis: what structure sees that raw text cannot.

Several tests deliberately assert the CONTRAST with the regex engine — those are
the reason this analyzer exists, and pinning them means a future change that
reintroduces a false positive fails here rather than in someone's PR.
"""

from sentinel.domain.services.ast_security_service import ASTSecurityService
from sentinel.domain.services.security_service import SecurityService
from sentinel.domain.value_objects.severity_level import SeverityLevel

ast_service = ASTSecurityService()
regex_service = SecurityService()


def rules(source: str) -> set[str]:
    findings = ast_service.analyze_source(source, "app.py")
    assert findings is not None, "source unexpectedly failed to parse"
    return {f.rule for f in findings}


def regex_rules(source: str) -> set[str]:
    return {f.rule for f in regex_service.analyze(source)["findings"]}


# --- unparseable input is reported, not guessed at ---------------------------


def test_unparseable_source_returns_none_not_empty():
    """None ("could not look") must not be confused with [] ("looked, found nothing").

    The caller falls back to the regex engine only in the first case.
    """
    assert ast_service.analyze_source("    indented fragment = 1", "a.py") is None
    assert ast_service.analyze_source("def broken(:", "a.py") is None
    assert ast_service.analyze_source("", "a.py") is None
    assert ast_service.analyze_source(None, "a.py") is None

    clean = ast_service.analyze_source("x = 1\n", "a.py")
    assert clean == []


def test_can_analyze_matches_analyze_source():
    assert ast_service.can_analyze("x = 1") is True
    assert ast_service.can_analyze("   nope = (") is False
    assert ast_service.can_analyze("") is False


# --- the false positives the regex engine cannot avoid -----------------------


def test_comments_are_invisible_to_the_ast():
    source = '# password = "hunter2" — an example in a comment, not code\nx = 1\n'
    assert rules(source) == set()
    assert regex_rules(source), "regex is expected to fire here — that's the point"


def test_docstrings_and_prose_do_not_fire():
    source = (
        'def configure():\n'
        '    """Do not write password = "hunter2" in source; use the environment."""\n'
        '    return True\n'
    )
    assert rules(source) == set()
    assert regex_rules(source), "regex is expected to fire on the docstring text"


def test_a_string_mentioning_eval_is_not_an_eval_call():
    source = 'message = "never call eval(user_input) in production"\n'
    assert rules(source) == set()
    assert regex_rules(source), "regex matches the substring inside the literal"


# --- what it still catches ---------------------------------------------------


def test_hardcoded_credentials_are_detected():
    assert "hardcoded_secret" in rules('password = "hunter2"\n')
    assert "hardcoded_secret" in rules('api_key = "sk-abc"\n')
    assert "hardcoded_secret" in rules('DB_PASSWORD = "x"\n')
    assert "hardcoded_secret" in rules('self.access_token = "abc"\n')


def test_credentials_read_at_runtime_are_not_findings():
    """The correct pattern must not be flagged, or the tool trains people to ignore it."""
    assert rules('password = os.environ["DB_PASSWORD"]\n') == set()
    assert rules('api_key = settings.LLM_API_KEY\n') == set()
    assert rules('token = get_token()\n') == set()
    assert rules('password = ""\n') == set()  # empty placeholder


def test_unrelated_names_that_merely_contain_a_keyword_are_ignored():
    assert rules('password_field_label = "Enter your password"\n') == set()


def test_sql_interpolation_is_detected_but_static_sql_is_not():
    assert "sql_injection" in rules('q = f"SELECT * FROM users WHERE id = {uid}"\n')
    assert "sql_injection" in rules('q = "SELECT * FROM users WHERE id = " + uid\n')
    assert "sql_injection" in rules('q = "SELECT name FROM t WHERE id = %s" % uid\n')
    # A fully static query is safe, and an f-string that isn't SQL is not our business.
    assert rules('q = "SELECT * FROM users WHERE id = 1"\n') == set()
    assert rules('msg = f"Hello {name}"\n') == set()


def test_dynamic_execution_is_graded_by_whether_input_is_literal():
    dynamic = ast_service.analyze_source("result = eval(user_input)\n", "a.py")
    literal = ast_service.analyze_source('result = eval("1 + 1")\n', "a.py")
    assert dynamic[0].severity == SeverityLevel.HIGH
    assert literal[0].severity == SeverityLevel.MEDIUM


def test_shell_execution_is_graded_the_same_way():
    dynamic = ast_service.analyze_source("os.system(cmd)\n", "a.py")
    literal = ast_service.analyze_source('os.system("ls -la")\n', "a.py")
    assert dynamic[0].severity == SeverityLevel.CRITICAL
    assert literal[0].severity == SeverityLevel.MEDIUM
    assert dynamic[0].category == "Injection"


def test_subprocess_shell_true_only_fires_with_shell_true():
    assert "subprocess_shell_true" in rules("subprocess.run(cmd, shell=True)\n")
    assert rules("subprocess.run([cmd, arg])\n") == set()
    assert rules("subprocess.run(cmd, shell=False)\n") == set()


# --- findings carry a real location ------------------------------------------


def test_findings_carry_the_file_and_the_true_line_number():
    source = (
        "import os\n"          # 1
        "\n"                   # 2
        "def handler(req):\n"  # 3
        "    x = 1\n"          # 4
        '    password = "s"\n' # 5
        "    return x\n"       # 6
    )
    findings = ast_service.analyze_source(source, "app/handler.py")
    assert len(findings) == 1
    assert findings[0].file == "app/handler.py"
    assert findings[0].line == 5
    assert findings[0].owasp_category == "A02: Cryptographic Failures"


def test_several_findings_in_one_file_are_all_reported():
    source = (
        'password = "a"\n'
        "os.system(cmd)\n"
        'q = f"SELECT x FROM t WHERE i={i}"\n'
    )
    findings = ast_service.analyze_source(source, "a.py")
    assert {f.rule for f in findings} == {"hardcoded_secret", "os_system_call", "sql_injection"}
    assert [f.line for f in sorted(findings, key=lambda f: f.line)] == [1, 2, 3]


# --- Credential shapes: coverage the name heuristic cannot give ---------------


def test_credential_literals_are_caught_regardless_of_variable_name():
    """`key` is too generic to treat as a secret word, so shape has to carry it.

    This is not optional polish: a file handled by this engine is withheld from
    the regex engine entirely, so anything missed here is missed by everything.
    """
    assert "aws_access_key" in rules('aws_key = "AKIAIOSFODNN7EXAMPLE"\n')
    assert "aws_access_key" in rules('cfg = {"k": "AKIAIOSFODNN7EXAMPLE"}\n')
    assert "openai_key" in rules('client = make("sk-abcdefghijklmnopqrstuvwx")\n')
    assert "github_token" in rules('t = "ghp_abcdefghijklmnopqrstuvwxyz0123"\n')


def test_a_named_credential_is_reported_once_not_twice():
    """Name and shape both match here; the finding must not be duplicated."""
    findings = ast_service.analyze_source('api_key = "sk-abcdefghijklmnopqrstuvwx"\n', "a.py")
    assert len(findings) == 1


def test_ordinary_long_strings_are_not_credentials():
    assert rules('msg = "the quick brown fox jumps over the lazy dog"\n') == set()
    assert rules('path = "/usr/local/share/some/long/directory/name"\n') == set()


def test_every_regex_secret_rule_is_still_covered_by_the_ast_engine():
    """Guards the handover: AST must not lose ground the regex engine held.

    If a future rule is added to SecurityService without an AST counterpart,
    files claimed by the AST engine would silently stop reporting it.
    """
    samples = {
        "openai_key": 'x = "sk-abcdefghijklmnopqrstuvwx"',
        "aws_access_key": 'x = "AKIAIOSFODNN7EXAMPLE"',
        "api_key_assignment": 'api_key = "literal-value-here"',
        "password_assignment": 'password = "literal-value-here"',
        "eval_call": "eval(user_input)",
        "exec_call": "exec(user_input)",
        "os_system_call": "os.system(user_cmd)",
        "subprocess_shell_true": "subprocess.run(user_cmd, shell=True)",
        "sql_injection": 'q = f"SELECT * FROM t WHERE i={i}"',
    }
    for regex_rule, source in samples.items():
        caught_by_regex = regex_rule in regex_rules(source)
        caught_by_ast = bool(rules(source))
        assert not caught_by_regex or caught_by_ast, (
            f"regex catches {regex_rule!r} in {source!r} but the AST engine reports nothing"
        )
