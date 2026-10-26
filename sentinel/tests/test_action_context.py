"""Reading a pull request out of the GitHub Actions environment.

``read_action_context`` takes the environment as an argument, so these are
plain function calls — no monkeypatching, no os.environ.
"""

import json
import pathlib
import shutil
import tempfile

import pytest

from sentinel.action.context import ActionContextError, read_action_context, read_event


@pytest.fixture
def event_dir():
    """Scratch space for event payloads.

    Deliberately not pytest's ``event_dir``: that fixture scans a shared
    per-user temp root for old runs, which is not readable on every machine,
    and these tests need a directory rather than all that bookkeeping.
    """
    path = tempfile.mkdtemp(prefix="sentinel-action-")
    try:
        yield pathlib.Path(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _event(event_dir, payload: dict) -> str:
    path = event_dir / "event.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _pull_request_event(fork: bool = False) -> dict:
    head_repo = "contributor/hello" if fork else "octo/hello"
    return {
        "action": "opened",
        "pull_request": {
            "number": 7,
            "head": {"sha": "head-sha-abc", "repo": {"full_name": head_repo}},
            "base": {"repo": {"full_name": "octo/hello"}},
        },
        "repository": {"full_name": "octo/hello", "owner": {"login": "octo"}, "name": "hello"},
    }


def _env(event_dir, payload: dict | None = None, **overrides) -> dict:
    base = {
        "GITHUB_TOKEN": "ghs_test",
        "GITHUB_REPOSITORY": "octo/hello",
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_API_URL": "https://api.github.com",
    }
    if payload is not None:
        base["GITHUB_EVENT_PATH"] = _event(event_dir, payload)
    base.update(overrides)
    return {key: value for key, value in base.items() if value is not None}


def test_reads_owner_repo_pr_and_head_sha(event_dir):
    context = read_action_context(_env(event_dir, _pull_request_event()))

    assert (context.owner, context.repo, context.pr_number) == ("octo", "hello", 7)
    assert context.head_sha == "head-sha-abc"
    assert context.token == "ghs_test"


def test_repo_is_the_bare_name_not_owner_slash_name(event_dir):
    """The API URL is built from owner and repo separately; a slash breaks it."""
    context = read_action_context(_env(event_dir, _pull_request_event()))
    assert "/" not in context.repo


def test_head_sha_comes_from_the_event_not_github_sha(event_dir):
    """GITHUB_SHA is the synthetic merge commit on pull_request.

    A check run attached to it does not appear on the pull request at all, so
    this is a silent-failure guard, not a style preference.
    """
    env = _env(event_dir, _pull_request_event(), GITHUB_SHA="merge-commit-sha")
    assert read_action_context(env).head_sha == "head-sha-abc"


def test_pr_number_falls_back_to_the_ref(event_dir):
    env = _env(event_dir, {"repository": {"full_name": "octo/hello"}},
               GITHUB_REF="refs/pull/42/merge")
    assert read_action_context(env).pr_number == 42


def test_issue_comment_on_a_pull_request_resolves_the_number(event_dir):
    payload = {"issue": {"number": 11, "pull_request": {"url": "..."}}}
    assert read_action_context(_env(event_dir, payload)).pr_number == 11


def test_issue_comment_on_a_plain_issue_is_rejected(event_dir):
    """An issue is not a pull request, however similar the payload looks."""
    with pytest.raises(ActionContextError, match="no pull request"):
        read_action_context(_env(event_dir, {"issue": {"number": 11}}))


def test_missing_token_is_rejected_with_a_usable_message(event_dir):
    env = _env(event_dir, _pull_request_event())
    env["GITHUB_TOKEN"] = "   "
    with pytest.raises(ActionContextError, match="GITHUB_TOKEN"):
        read_action_context(env)


def test_missing_repository_is_rejected(event_dir):
    env = _env(event_dir, {"pull_request": {"number": 7}})
    del env["GITHUB_REPOSITORY"]
    with pytest.raises(ActionContextError, match="repository"):
        read_action_context(env)


def test_repository_falls_back_to_the_event_payload(event_dir):
    env = _env(event_dir, _pull_request_event())
    del env["GITHUB_REPOSITORY"]
    context = read_action_context(env)
    assert (context.owner, context.repo) == ("octo", "hello")


def test_malformed_event_json_is_an_error_not_a_guess(event_dir):
    path = event_dir / "event.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ActionContextError, match="GITHUB_EVENT_PATH"):
        read_action_context(_env(event_dir, GITHUB_EVENT_PATH=str(path)))


def test_a_missing_event_file_is_tolerated_when_the_ref_carries_the_pr(event_dir):
    env = _env(event_dir, GITHUB_EVENT_PATH=str(event_dir / "absent.json"),
               GITHUB_REF="refs/pull/5/merge")
    assert read_action_context(env).pr_number == 5


def test_read_event_without_a_path_is_empty():
    assert read_event(None) == {}


def test_fork_is_detected_from_differing_head_and_base_repos(event_dir):
    assert read_action_context(_env(event_dir, _pull_request_event(fork=True))).is_fork is True


def test_a_same_repo_pull_request_is_not_a_fork(event_dir):
    assert read_action_context(_env(event_dir, _pull_request_event())).is_fork is False


def test_an_inaccessible_head_repo_is_assumed_to_be_a_fork(event_dir):
    """Being wrong this way adds a warning; the other way hides one."""
    payload = {
        "pull_request": {"number": 7, "head": {"sha": "x", "repo": None},
                         "base": {"repo": {"full_name": "octo/hello"}}},
        "repository": {"full_name": "octo/hello"},
    }
    assert read_action_context(_env(event_dir, payload)).is_fork is True


def test_enterprise_api_url_is_honoured(event_dir):
    env = _env(event_dir, _pull_request_event(), GITHUB_API_URL="https://ghe.corp/api/v3")
    assert read_action_context(env).api_base_url == "https://ghe.corp/api/v3"


def test_api_url_defaults_when_unset(event_dir):
    env = _env(event_dir, _pull_request_event())
    del env["GITHUB_API_URL"]
    assert read_action_context(env).api_base_url == "https://api.github.com"
