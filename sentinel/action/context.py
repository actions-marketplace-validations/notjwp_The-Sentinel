"""Reading the pull request out of the GitHub Actions environment.

Pure parsing, deliberately: ``read_action_context`` takes the environment as an
argument rather than reaching for ``os.environ``, so every case below is a
plain function call in a test with no monkeypatching.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_API_BASE_URL = "https://api.github.com"


class ActionContextError(Exception):
    """The environment does not describe a reviewable pull request."""


@dataclass(frozen=True)
class ActionContext:
    owner: str
    repo: str
    pr_number: int
    token: str
    api_base_url: str
    head_sha: str | None
    event_name: str
    is_fork: bool


def _as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def read_event(event_path: str | None) -> dict:
    """Parse the event JSON. Absent file -> {}; unreadable JSON -> an error.

    A missing path is tolerated because the pull request can still be
    recoverable from ``GITHUB_REF``. Malformed JSON is not tolerated: it means
    the environment is lying about its shape, and guessing past that would
    produce a review of the wrong thing.
    """
    if not event_path:
        return {}
    try:
        with open(event_path, encoding="utf-8") as handle:
            return _as_dict(json.load(handle))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise ActionContextError(f"could not read GITHUB_EVENT_PATH ({event_path}): {exc}") from exc


def _pr_number_from_ref(ref: str | None) -> int | None:
    """``refs/pull/42/merge`` -> 42. The fallback when the event body has none."""
    if not isinstance(ref, str):
        return None
    parts = ref.split("/")
    if len(parts) >= 3 and parts[0] == "refs" and parts[1] == "pull":
        try:
            return int(parts[2])
        except ValueError:
            return None
    return None


def _pr_number(event: dict, env: Mapping[str, str]) -> int | None:
    pull_request = _as_dict(event.get("pull_request"))
    number = pull_request.get("number")
    if isinstance(number, int) and not isinstance(number, bool):
        return number

    top_level = event.get("number")
    if isinstance(top_level, int) and not isinstance(top_level, bool):
        return top_level

    # issue_comment fires for issues AND pull requests; only the latter carries
    # a `pull_request` key. Without that check a comment on a plain issue would
    # be reviewed as if it were a PR.
    issue = _as_dict(event.get("issue"))
    if "pull_request" in issue:
        issue_number = issue.get("number")
        if isinstance(issue_number, int) and not isinstance(issue_number, bool):
            return issue_number

    return _pr_number_from_ref(env.get("GITHUB_REF"))


def _owner_and_repo(event: dict, env: Mapping[str, str]) -> tuple[str | None, str | None]:
    repository_env = env.get("GITHUB_REPOSITORY")
    if isinstance(repository_env, str) and "/" in repository_env:
        owner, _, name = repository_env.partition("/")
        if owner and name:
            return owner, name

    repository = _as_dict(event.get("repository"))
    full_name = repository.get("full_name")
    if isinstance(full_name, str) and "/" in full_name:
        owner, _, name = full_name.partition("/")
        if owner and name:
            return owner, name

    owner_login = _as_dict(repository.get("owner")).get("login")
    name = repository.get("name")
    if isinstance(owner_login, str) and isinstance(name, str) and owner_login and name:
        return owner_login, name

    return None, None


def _head_sha(event: dict) -> str | None:
    """The PR's head commit.

    Never ``GITHUB_SHA``: on a ``pull_request`` event that is the synthetic
    merge commit, and on ``pull_request_target`` it is the base commit. A check
    run attached to either is invisible on the pull request.
    """
    head = _as_dict(_as_dict(event.get("pull_request")).get("head"))
    sha = head.get("sha")
    return sha if isinstance(sha, str) and sha.strip() != "" else None


def _is_fork(event: dict) -> bool:
    """Whether the PR comes from a fork, which makes GITHUB_TOKEN read-only."""
    pull_request = _as_dict(event.get("pull_request"))
    head_repo = _as_dict(_as_dict(pull_request.get("head")).get("repo"))
    base_repo = _as_dict(_as_dict(pull_request.get("base")).get("repo"))

    head_name = head_repo.get("full_name")
    base_name = base_repo.get("full_name")
    if not isinstance(head_name, str) or not isinstance(base_name, str):
        # A deleted or inaccessible head repo reads as absent. Assume fork:
        # being wrong that way only adds a warning, the other way hides one.
        return bool(pull_request)
    return head_name != base_name


def read_action_context(env: Mapping[str, str]) -> ActionContext:
    """Build the review context from the Actions environment.

    Raises ``ActionContextError`` with a message naming the missing piece —
    these are the errors a user reads in a red workflow log, so they say what
    to fix rather than what went wrong.
    """
    token = env.get("GITHUB_TOKEN") or ""
    if token.strip() == "":
        raise ActionContextError(
            "GITHUB_TOKEN is not set. Pass it to the action: "
            "`with: {github-token: ${{ github.token }}}`."
        )

    event = read_event(env.get("GITHUB_EVENT_PATH"))

    owner, repo = _owner_and_repo(event, env)
    if not owner or not repo:
        raise ActionContextError(
            "could not determine the repository: GITHUB_REPOSITORY is unset and the "
            "event payload carries no repository."
        )

    pr_number = _pr_number(event, env)
    if pr_number is None:
        raise ActionContextError(
            "this event has no pull request to review. Trigger the workflow with "
            "`on: pull_request`."
        )

    return ActionContext(
        owner=owner,
        repo=repo,
        pr_number=pr_number,
        token=token,
        # GITHUB_API_URL is always set by the runner and is correct on GitHub
        # Enterprise Server, where api.github.com is the wrong host entirely.
        api_base_url=env.get("GITHUB_API_URL") or DEFAULT_API_BASE_URL,
        head_sha=_head_sha(event),
        event_name=env.get("GITHUB_EVENT_NAME") or "",
        is_fork=_is_fork(event),
    )
