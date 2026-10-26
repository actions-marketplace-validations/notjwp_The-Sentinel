import base64
import types
import urllib.error

import pytest

import sentinel.infrastructure.github.github_client as gh
from sentinel.infrastructure.github.github_client import GitHubClient


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _ = exc_type
        _ = exc
        _ = tb
        return False


class _FakeHTTPError(urllib.error.HTTPError):
    """A real HTTPError so the client's typed handler sees it as GitHub would."""

    def __init__(self, message: str = "http error", code: int = 500, headers=None) -> None:
        super().__init__("https://api.github.com/x", code, message, headers or {}, None)


class _FakeURLError(urllib.error.URLError):
    pass


def _patch_imports(
    monkeypatch,
    *,
    jwt_encode=None,
    jwt_raises: bool = False,
    response_payload: bytes = b"{}",
    urlopen_error: Exception | None = None,
) -> None:
    """Stub the network and the JWT signer.

    Patches the real `urllib.request.urlopen` and `jwt.encode` that
    `github_client` imports at module scope. (Before M9 the client called
    `__import__` inside each method purely so this helper could intercept it;
    patching the bound names is the ordinary way and keeps the production code
    idiomatic.) Retry backoff is neutralized so error-path tests stay fast.
    """
    if jwt_raises:
        monkeypatch.setattr(gh, "jwt", None)
    else:
        encode = jwt_encode or (lambda payload, private_key, algorithm: "token")
        monkeypatch.setattr(gh, "jwt", types.SimpleNamespace(encode=encode))

    def fake_urlopen(request, timeout=10):
        _ = request
        _ = timeout
        if urlopen_error is not None:
            raise urlopen_error
        return _FakeResponse(response_payload)

    monkeypatch.setattr(gh.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(gh.GitHubClient, "_sleep", staticmethod(lambda _s: None))


def test_build_app_jwt_success_with_bytes(monkeypatch):
    captured = {}

    def fake_encode(payload, private_key, algorithm):
        captured["payload"] = payload
        captured["private_key"] = private_key
        captured["algorithm"] = algorithm
        return b"jwt-token"

    _patch_imports(monkeypatch, jwt_encode=fake_encode)
    client = GitHubClient(app_id="123", installation_id="1", private_key="private")

    token = client._build_app_jwt()

    assert token == "jwt-token"
    assert captured["private_key"] == "private"
    assert captured["algorithm"] == "RS256"
    assert captured["payload"]["iss"] == "123"


def test_build_app_jwt_success_with_string_token(monkeypatch):
    _patch_imports(
        monkeypatch,
        jwt_encode=lambda payload, private_key, algorithm: "string-token",
    )
    client = GitHubClient(app_id="123", installation_id="1", private_key="private")

    assert client._build_app_jwt() == "string-token"


def test_build_app_jwt_returns_none_for_missing_credentials_or_jwt_failure(monkeypatch):
    no_creds = GitHubClient(app_id=None, installation_id="1", private_key=None)
    assert no_creds._build_app_jwt() is None

    _patch_imports(monkeypatch, jwt_raises=True)
    client = GitHubClient(app_id="123", installation_id="1", private_key="private")
    assert client._build_app_jwt() is None


def test_parse_expiry_handles_missing_and_invalid_values():
    now = GitHubClient._now()
    missing = GitHubClient._parse_expiry(None)
    invalid = GitHubClient._parse_expiry("not-a-date")

    assert missing >= now + 299
    assert invalid >= now + 299


def test_http_json_returns_empty_dict_on_empty_body(monkeypatch):
    _patch_imports(monkeypatch, response_payload=b"")
    client = GitHubClient(app_id="123", installation_id="1", private_key="private")

    result = client._http_json("POST", "https://api.github.com/x", headers={}, data={})

    assert result == {}


def test_http_json_returns_empty_dict_for_non_dict_json(monkeypatch):
    _patch_imports(monkeypatch, response_payload=b"[1,2,3]")
    client = GitHubClient(app_id="123", installation_id="1", private_key="private")

    result = client._http_json("POST", "https://api.github.com/x", headers={}, data={})

    assert result == {}


def test_http_json_returns_dict_payload(monkeypatch):
    _patch_imports(monkeypatch, response_payload=b'{"id": 42}')
    client = GitHubClient(app_id="123", installation_id="1", private_key="private")

    result = client._http_json("POST", "https://api.github.com/x", headers={}, data={})

    assert result == {"id": 42}


def test_http_json_handles_network_and_http_errors(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="1", private_key="private")

    _patch_imports(monkeypatch, urlopen_error=_FakeURLError("network"))
    assert client._http_json("POST", "https://api.github.com/x", headers={}, data={}) is None

    _patch_imports(monkeypatch, urlopen_error=_FakeHTTPError("api failure"))
    assert client._http_json("POST", "https://api.github.com/x", headers={}, data={}) is None


def test_http_json_handles_http_error_read_failure(monkeypatch):
    class _UnreadableHTTPError(_FakeHTTPError):
        def read(self) -> bytes:
            raise RuntimeError("cannot read body")

    _patch_imports(monkeypatch, urlopen_error=_UnreadableHTTPError("api failure"))
    client = GitHubClient(app_id="123", installation_id="1", private_key="private")

    assert client._http_json("POST", "https://api.github.com/x", headers={}, data={}) is None


def test_http_json_handles_generic_exception(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="1", private_key="private")

    _patch_imports(monkeypatch, urlopen_error=RuntimeError("boom"))
    assert client._http_json("POST", "https://api.github.com/x", headers={}, data={}) is None


def test_access_token_retrieval_success_and_cache(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    calls = {"count": 0}

    def fake_http_json(method: str, url: str, headers: dict[str, str], data=None):
        _ = method
        _ = headers
        _ = data
        calls["count"] += 1
        assert url.endswith("/app/installations/999/access_tokens")
        return {
            "token": "installation-token",
            "expires_at": "2099-01-01T00:00:00Z",
        }

    monkeypatch.setattr(client, "_build_app_jwt", lambda: "app-jwt")
    monkeypatch.setattr(client, "_http_json", fake_http_json)

    first = client._get_installation_token()
    second = client._get_installation_token()

    assert first == "installation-token"
    assert second == "installation-token"
    assert calls["count"] == 1


@pytest.mark.parametrize(
    "response",
    [None, {}, {"token": 123}, {"token": "   "}],
)
def test_access_token_failure_for_invalid_response(monkeypatch, response):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    monkeypatch.setattr(client, "_build_app_jwt", lambda: "app-jwt")
    monkeypatch.setattr(client, "_http_json", lambda method, url, headers, data=None: response)

    assert client._get_installation_token() is None


def test_invalid_installation_id_returns_none():
    client = GitHubClient(app_id="123", installation_id="", private_key="private")

    assert client._get_installation_token() is None


def test_get_installation_token_returns_none_when_jwt_generation_fails(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    monkeypatch.setattr(client, "_build_app_jwt", lambda: None)

    assert client._get_installation_token() is None


def test_post_comment_success_and_failures(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")

    monkeypatch.setattr(client, "_get_installation_token", lambda: "installation-token")
    monkeypatch.setattr(client, "_http_json", lambda method, url, headers, data=None: {"id": 1})
    assert client.post_comment("octo", "repo", 7, "hello") is True

    monkeypatch.setattr(client, "_http_json", lambda method, url, headers, data=None: None)
    assert client.post_comment("octo", "repo", 7, "hello") is False

    monkeypatch.setattr(client, "_get_installation_token", lambda: None)
    assert client.post_comment("octo", "repo", 7, "hello") is False


def test_post_comment_rejects_empty_inputs():
    client = GitHubClient(app_id=None, installation_id=None, private_key=None)

    assert client.post_comment("", "repo", 1, "body") is False
    assert client.post_comment("octo", "", 1, "body") is False
    assert client.post_comment("octo", "repo", 1, "") is False


# --- PR fetch (M1) ---------------------------------------------------------


def test_added_lines_from_patch_extracts_added_code_only():
    patch = (
        "@@ -1,3 +1,4 @@\n"
        "+++ b/app.py\n"          # file header must be ignored
        " unchanged = 1\n"        # context line dropped
        "-removed = 2\n"          # removed line dropped
        "+password = \"hunter2\"\n"
        "+api_key = \"sk-abc\"\n"
    )

    result = GitHubClient._added_lines_from_patch(patch)

    assert result == 'password = "hunter2"\napi_key = "sk-abc"'


@pytest.mark.parametrize("patch", [None, "", 123, [], "@@ -1 +1 @@\n context"])
def test_added_lines_from_patch_safe_for_non_added_content(patch):
    assert GitHubClient._added_lines_from_patch(patch) == ""


def test_http_json_list_returns_list_empty_and_none(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="1", private_key="private")

    _patch_imports(monkeypatch, response_payload=b'[{"filename": "a.py"}]')
    assert client._http_json_list("GET", "https://api.github.com/x", headers={}) == [
        {"filename": "a.py"}
    ]

    _patch_imports(monkeypatch, response_payload=b"")
    assert client._http_json_list("GET", "https://api.github.com/x", headers={}) == []

    # A dict body (not a list) must not be surfaced as a list.
    _patch_imports(monkeypatch, response_payload=b'{"message": "not found"}')
    assert client._http_json_list("GET", "https://api.github.com/x", headers={}) is None

    _patch_imports(monkeypatch, urlopen_error=_FakeURLError("network"))
    assert client._http_json_list("GET", "https://api.github.com/x", headers={}) is None


def test_get_pull_request_files_success_and_failures(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    monkeypatch.setattr(client, "_get_installation_token", lambda: "installation-token")

    def fake_list(method, url, headers, data=None):
        assert method == "GET"
        assert url.endswith("/repos/octo/repo/pulls/7/files?per_page=100&page=1")
        assert headers["Authorization"] == "token installation-token"
        return [{"filename": "a.py", "patch": "+x = 1"}, "junk", {"filename": "b.py"}]

    monkeypatch.setattr(client, "_http_json_list", fake_list)
    files = client.get_pull_request_files("octo", "repo", 7)
    assert files == [{"filename": "a.py", "patch": "+x = 1"}, {"filename": "b.py"}]

    # None response -> []
    monkeypatch.setattr(client, "_http_json_list", lambda *a, **k: None)
    assert client.get_pull_request_files("octo", "repo", 7) == []


def test_get_pull_request_files_returns_empty_without_owner_or_token(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    monkeypatch.setattr(client, "_get_installation_token", lambda: "installation-token")
    assert client.get_pull_request_files("", "repo", 7) == []
    assert client.get_pull_request_files("octo", "", 7) == []

    monkeypatch.setattr(client, "_get_installation_token", lambda: None)
    assert client.get_pull_request_files("octo", "repo", 7) == []


def test_get_pull_request_code_assembles_added_lines(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")

    monkeypatch.setattr(
        client,
        "get_pull_request_files",
        lambda owner, repo, pr_number: [
            {"filename": "a.py", "patch": "@@ -0,0 +1 @@\n+import os\n+os.system(cmd)"},
            {"filename": "bin.png"},  # no patch -> skipped
            {"filename": "b.py", "patch": "+value = 2"},
        ],
    )

    code = client.get_pull_request_code("octo", "repo", 7)
    assert code == "import os\nos.system(cmd)\nvalue = 2"


def test_get_pull_request_code_empty_when_no_files(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    monkeypatch.setattr(client, "get_pull_request_files", lambda *a, **k: [])
    assert client.get_pull_request_code("octo", "repo", 7) == ""


def test_list_issue_comments_success_and_failures(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    monkeypatch.setattr(client, "_get_installation_token", lambda: "installation-token")

    def fake_list(method, url, headers, data=None):
        assert method == "GET"
        assert url.endswith("/repos/octo/repo/issues/7/comments?per_page=100&page=1")
        assert headers["Authorization"] == "token installation-token"
        return [{"id": 1, "body": "hi"}, "junk", {"id": 2, "body": "yo"}]

    monkeypatch.setattr(client, "_http_json_list", fake_list)
    assert client.list_issue_comments("octo", "repo", 7) == [
        {"id": 1, "body": "hi"},
        {"id": 2, "body": "yo"},
    ]

    # None (error) response -> []
    monkeypatch.setattr(client, "_http_json_list", lambda *a, **k: None)
    assert client.list_issue_comments("octo", "repo", 7) == []


def test_list_issue_comments_empty_without_owner_or_token(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    monkeypatch.setattr(client, "_get_installation_token", lambda: "installation-token")
    assert client.list_issue_comments("", "repo", 7) == []
    assert client.list_issue_comments("octo", "", 7) == []

    monkeypatch.setattr(client, "_get_installation_token", lambda: None)
    assert client.list_issue_comments("octo", "repo", 7) == []


def test_update_comment_success_and_failures(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    monkeypatch.setattr(client, "_get_installation_token", lambda: "installation-token")

    def fake_patch(method, url, headers, data=None):
        assert method == "PATCH"
        assert url.endswith("/repos/octo/repo/issues/comments/55")
        assert data == {"body": "updated"}
        return {"id": 55}

    monkeypatch.setattr(client, "_http_json", fake_patch)
    assert client.update_comment("octo", "repo", 55, "updated") is True

    monkeypatch.setattr(client, "_http_json", lambda *a, **k: None)
    assert client.update_comment("octo", "repo", 55, "updated") is False

    monkeypatch.setattr(client, "_get_installation_token", lambda: None)
    assert client.update_comment("octo", "repo", 55, "updated") is False


def test_update_comment_rejects_empty_inputs():
    client = GitHubClient(app_id=None, installation_id=None, private_key=None)
    assert client.update_comment("", "repo", 1, "body") is False
    assert client.update_comment("octo", "", 1, "body") is False
    assert client.update_comment("octo", "repo", 1, "") is False


def test_upsert_comment_updates_existing_marked_comment(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    marker = GitHubClient.SENTINEL_COMMENT_MARKER
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        client,
        "list_issue_comments",
        lambda *a, **k: [
            {"id": 10, "body": "unrelated human comment"},
            {"id": 11, "body": f"{marker}\n# old Sentinel review"},
        ],
    )

    def fake_update(owner, repo, comment_id, body):
        calls["update"] = (owner, repo, comment_id, body)
        return True

    def fake_post(*a, **k):
        calls["post"] = True
        return True

    monkeypatch.setattr(client, "update_comment", fake_update)
    monkeypatch.setattr(client, "post_comment", fake_post)

    assert client.upsert_comment("octo", "repo", 7, "# fresh review") is True
    # PATCHed the marked comment (id 11); never created a new one.
    assert calls["update"][:3] == ("octo", "repo", 11)
    assert calls["update"][3] == f"{marker}\n# fresh review"  # marker prepended once
    assert "post" not in calls


def test_upsert_comment_creates_when_no_marked_comment(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    marker = GitHubClient.SENTINEL_COMMENT_MARKER
    posted: dict[str, object] = {}

    monkeypatch.setattr(
        client, "list_issue_comments", lambda *a, **k: [{"id": 10, "body": "just a human"}]
    )
    monkeypatch.setattr(client, "update_comment", lambda *a, **k: pytest.fail("should not update"))

    def fake_post(owner, repo, pr_number, body):
        posted["args"] = (owner, repo, pr_number, body)
        return True

    monkeypatch.setattr(client, "post_comment", fake_post)

    assert client.upsert_comment("octo", "repo", 7, "# review") is True
    assert posted["args"] == ("octo", "repo", 7, f"{marker}\n# review")


def test_upsert_comment_falls_back_to_post_when_list_fails(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")

    def boom(*a, **k):
        raise RuntimeError("list blew up")

    monkeypatch.setattr(client, "list_issue_comments", boom)
    monkeypatch.setattr(client, "post_comment", lambda *a, **k: True)

    assert client.upsert_comment("octo", "repo", 7, "# review") is True


def test_upsert_comment_does_not_double_prepend_marker(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    marker = GitHubClient.SENTINEL_COMMENT_MARKER
    seen: dict[str, str] = {}

    def fake_post(owner, repo, pr, body):
        seen["body"] = body
        return True

    monkeypatch.setattr(client, "list_issue_comments", lambda *a, **k: [])
    monkeypatch.setattr(client, "post_comment", fake_post)

    already_marked = f"{marker}\n# review"
    assert client.upsert_comment("octo", "repo", 7, already_marked) is True
    assert seen["body"] == already_marked  # unchanged; marker not duplicated


def test_upsert_comment_rejects_empty_inputs():
    client = GitHubClient(app_id=None, installation_id=None, private_key=None)
    assert client.upsert_comment("", "repo", 1, "body") is False
    assert client.upsert_comment("octo", "", 1, "body") is False
    assert client.upsert_comment("octo", "repo", 1, "") is False


def test_paginated_assembles_multiple_pages(monkeypatch):
    """A full page (100 items) triggers a second request; a short page ends the loop."""
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    monkeypatch.setattr(client, "_get_installation_token", lambda: "installation-token")

    requested_urls: list[str] = []

    def fake_list(method, url, headers, data=None):
        requested_urls.append(url)
        if url.endswith("&page=1"):
            return [{"id": i} for i in range(100)]
        return [{"id": 100}, {"id": 101}, {"id": 102}]

    monkeypatch.setattr(client, "_http_json_list", fake_list)
    comments = client.list_issue_comments("octo", "repo", 7)

    assert len(comments) == 103
    assert len(requested_urls) == 2
    assert requested_urls[0].endswith("?per_page=100&page=1")
    assert requested_urls[1].endswith("?per_page=100&page=2")


def test_paginated_stops_at_page_cap(monkeypatch):
    """Endless full pages stop at MAX_LIST_PAGES rather than looping forever."""
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    monkeypatch.setattr(client, "_get_installation_token", lambda: "installation-token")

    calls = {"n": 0}

    def always_full(method, url, headers, data=None):
        calls["n"] += 1
        return [{"id": i} for i in range(100)]

    monkeypatch.setattr(client, "_http_json_list", always_full)
    files = client.get_pull_request_files("octo", "repo", 7)

    assert calls["n"] == GitHubClient.MAX_LIST_PAGES
    assert len(files) == 100 * GitHubClient.MAX_LIST_PAGES


def test_paginated_keeps_partial_results_on_mid_pagination_error(monkeypatch):
    """An error (None) on page 2 returns page 1's items, not []."""
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    monkeypatch.setattr(client, "_get_installation_token", lambda: "installation-token")

    def fake_list(method, url, headers, data=None):
        if url.endswith("&page=1"):
            return [{"id": i} for i in range(100)]
        return None

    monkeypatch.setattr(client, "_http_json_list", fake_list)
    comments = client.list_issue_comments("octo", "repo", 7)

    assert len(comments) == 100
    assert comments[0] == {"id": 0}


def test_upsert_comment_finds_marked_comment_on_second_page(monkeypatch):
    """Sentinel's marker beyond page 1 is still found -> PATCH, not a duplicate POST."""
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    marker = GitHubClient.SENTINEL_COMMENT_MARKER
    monkeypatch.setattr(client, "_get_installation_token", lambda: "installation-token")

    def fake_list(method, url, headers, data=None):
        if url.endswith("&page=1"):
            return [{"id": i, "body": f"human comment {i}"} for i in range(100)]
        return [{"id": 200, "body": f"{marker}\n# old Sentinel review"}]

    calls: dict[str, object] = {}

    def fake_update(owner, repo, comment_id, body):
        calls["update"] = comment_id
        return True

    monkeypatch.setattr(client, "_http_json_list", fake_list)
    monkeypatch.setattr(client, "update_comment", fake_update)
    monkeypatch.setattr(client, "post_comment", lambda *a, **k: pytest.fail("should not create"))

    assert client.upsert_comment("octo", "repo", 7, "# fresh review") is True
    assert calls["update"] == 200


def test_get_pull_request_data_assembles_names_contents_and_code(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")

    monkeypatch.setattr(
        client,
        "get_pull_request_files",
        lambda owner, repo, pr_number: [
            {"filename": "a.py", "patch": "@@ -0,0 +1 @@\n+import os\n+os.system(cmd)"},
            {"filename": "bin.png"},  # no patch -> listed in files, absent from contents
            {"filename": "README.md", "patch": "+# Title"},
        ],
    )

    data = client.get_pull_request_data("octo", "repo", 7)

    assert data["files"] == ["a.py", "bin.png", "README.md"]
    assert data["file_contents"] == {
        "a.py": "@@ -0,0 +1 @@\n+import os\n+os.system(cmd)",
        "README.md": "+# Title",
    }
    assert data["code"] == "import os\nos.system(cmd)\n# Title"


def test_get_pull_request_data_empty_when_no_files(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    monkeypatch.setattr(client, "get_pull_request_files", lambda *a, **k: [])
    assert client.get_pull_request_data("octo", "repo", 7) == {
        "code": "",
        "files": [],
        "file_contents": {},
        "line_map": [],
    }


def test_get_pull_request_refs_extracts_shas(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    monkeypatch.setattr(client, "_get_installation_token", lambda: "installation-token")

    def fake_json(method, url, headers, data=None):
        assert method == "GET"
        assert url.endswith("/repos/octo/repo/pulls/7")
        assert headers["Authorization"] == "token installation-token"
        return {"head": {"sha": "h1"}, "base": {"sha": "b1"}}

    monkeypatch.setattr(client, "_http_json", fake_json)
    assert client.get_pull_request_refs("octo", "repo", 7) == {"head_sha": "h1", "base_sha": "b1"}

    # Error response -> {}; missing owner -> {} without any request.
    monkeypatch.setattr(client, "_http_json", lambda *a, **k: None)
    assert client.get_pull_request_refs("octo", "repo", 7) == {}
    assert client.get_pull_request_refs("", "repo", 7) == {}


def test_get_content_by_url_decodes_base64_with_newlines(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    monkeypatch.setattr(client, "_get_installation_token", lambda: "installation-token")

    encoded = base64.b64encode(b"Install with pip install x\nUsage: run it\n").decode()
    wrapped = "\n".join(encoded[i : i + 16] for i in range(0, len(encoded), 16))
    monkeypatch.setattr(
        client, "_http_json", lambda *a, **k: {"encoding": "base64", "content": wrapped}
    )

    text = client.get_content_by_url("https://api.test/repos/o/r/contents/README.md?ref=abc")
    assert text is not None
    assert "pip install x" in text
    assert "Usage: run it" in text


def test_get_content_by_url_edge_cases(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    monkeypatch.setattr(client, "_get_installation_token", lambda: "installation-token")

    # Genuinely empty file -> "" (NOT None, which signals failure).
    monkeypatch.setattr(client, "_http_json", lambda *a, **k: {"encoding": "base64", "content": ""})
    assert client.get_content_by_url("https://api.test/contents/EMPTY.md?ref=x") == ""

    # Unsupported encoding -> None.
    monkeypatch.setattr(
        client, "_http_json", lambda *a, **k: {"encoding": "utf-8", "content": "raw"}
    )
    assert client.get_content_by_url("https://api.test/contents/x?ref=x") is None

    # Error response -> None; non-http url -> None without a request.
    monkeypatch.setattr(client, "_http_json", lambda *a, **k: None)
    assert client.get_content_by_url("https://api.test/contents/x?ref=x") is None
    assert client.get_content_by_url("not-a-url") is None


def _corpus_tree_and_blobs(py_source: bytes) -> tuple[dict, dict]:
    encoded = base64.b64encode(py_source).decode()
    tree = {
        "tree": [
            {"type": "blob", "path": "app.py", "sha": "s1", "size": 30},
            {"type": "blob", "path": "README.md", "sha": "s2", "size": 10},  # not .py
            {"type": "tree", "path": "pkg", "sha": "s3"},  # not a blob
            {"type": "blob", "path": ".hidden/tool.py", "sha": "s4", "size": 10},  # dot path
            {"type": "blob", "path": "big.py", "sha": "s5", "size": 999_999},  # over size cap
            {"type": "blob", "path": "util.py", "sha": "s6", "size": 25},
        ]
    }
    blob = {"encoding": "base64", "content": encoded}
    return tree, blob


def test_get_repo_code_corpus_assembles_filters_and_caches(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    monkeypatch.setattr(client, "_get_installation_token", lambda: "installation-token")
    tree, blob = _corpus_tree_and_blobs(b"def add(a, b):\n    return a + b\n")
    calls = {"tree": 0, "blobs": []}

    def fake_json(method, url, headers, data=None):
        if "/git/trees/" in url:
            calls["tree"] += 1
            assert url.endswith("/repos/octo/repo/git/trees/base-sha?recursive=1")
            return tree
        if "/git/blobs/" in url:
            calls["blobs"].append(url.rsplit("/", 1)[1])
            return blob
        pytest.fail(f"unexpected url {url}")

    monkeypatch.setattr(client, "_http_json", fake_json)
    corpus = client.get_repo_code_corpus("octo", "repo", "base-sha")

    assert len(corpus) == 2  # only app.py and util.py qualify
    assert all("def add" in text for text in corpus)
    assert calls["blobs"] == ["s1", "s6"]

    # Second call for the same ref is served from the cache: no new requests.
    assert client.get_repo_code_corpus("octo", "repo", "base-sha") == corpus
    assert calls["tree"] == 1
    assert len(calls["blobs"]) == 2


def test_get_repo_code_corpus_respects_file_cap(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    monkeypatch.setattr(client, "_get_installation_token", lambda: "installation-token")
    encoded = base64.b64encode(b"def f(a, b, c):\n    return a\n").decode()
    tree = {
        "tree": [
            {"type": "blob", "path": f"mod_{i}.py", "sha": f"sha{i}", "size": 20}
            for i in range(GitHubClient.CORPUS_MAX_FILES + 15)
        ]
    }
    blobs_fetched = []

    def fake_json(method, url, headers, data=None):
        if "/git/trees/" in url:
            return tree
        blobs_fetched.append(url)
        return {"encoding": "base64", "content": encoded}

    monkeypatch.setattr(client, "_http_json", fake_json)
    corpus = client.get_repo_code_corpus("octo", "repo", "cap-ref")
    assert len(corpus) == GitHubClient.CORPUS_MAX_FILES
    assert len(blobs_fetched) == GitHubClient.CORPUS_MAX_FILES


def test_get_repo_code_corpus_failures(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    monkeypatch.setattr(client, "_get_installation_token", lambda: "installation-token")

    # Tree fetch error -> [].
    monkeypatch.setattr(client, "_http_json", lambda *a, **k: None)
    assert client.get_repo_code_corpus("octo", "repo", "dead-ref") == []

    # One blob failing -> partial corpus, not [].
    encoded = base64.b64encode(b"def keep(a, b):\n    return a - b\n").decode()

    def fake_json(method, url, headers, data=None):
        if "/git/trees/" in url:
            return {
                "tree": [
                    {"type": "blob", "path": "bad.py", "sha": "bad", "size": 10},
                    {"type": "blob", "path": "good.py", "sha": "good", "size": 10},
                ]
            }
        if url.endswith("/bad"):
            return None
        return {"encoding": "base64", "content": encoded}

    monkeypatch.setattr(client, "_http_json", fake_json)
    corpus = client.get_repo_code_corpus("octo", "repo", "partial-ref")
    assert len(corpus) == 1
    assert "def keep" in corpus[0]


def test_get_pull_request_data_fetches_full_source_for_docs_and_python(monkeypatch):
    """Both need whole files, for different reasons.

    Docs are judged on their entire text; .py files need a complete module
    because an AST cannot parse a diff fragment. Other file types keep the patch
    text, which is all the regex engine requires.
    """
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    items = [
        {"filename": "app.py", "patch": "+x = 1", "contents_url": "https://api.test/c/app"},
        {"filename": "README.md", "patch": "+one line", "contents_url": "https://api.test/c/rm"},
        {"filename": "styles.css", "patch": "+body {}", "contents_url": "https://api.test/c/css"},
    ]
    monkeypatch.setattr(client, "get_pull_request_files", lambda *a, **k: items)

    fetched = []

    def fake_content(url):
        fetched.append(url)
        return "FULL:" + url.rsplit("/", 1)[-1]

    monkeypatch.setattr(client, "get_content_by_url", fake_content)
    data = client.get_pull_request_data("octo", "repo", 7)

    assert data["file_contents"]["README.md"] == "FULL:rm"
    assert data["file_contents"]["app.py"] == "FULL:app"
    assert data["file_contents"]["styles.css"] == "+body {}"  # patch text is enough
    assert sorted(fetched) == ["https://api.test/c/app", "https://api.test/c/rm"]


def test_get_pull_request_data_doc_fetch_failure_falls_back_to_patch(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    items = [{"filename": "README.md", "patch": "+patched", "contents_url": "https://api.test/c"}]
    monkeypatch.setattr(client, "get_pull_request_files", lambda *a, **k: items)
    monkeypatch.setattr(client, "get_content_by_url", lambda url: None)

    data = client.get_pull_request_data("octo", "repo", 7)
    assert data["file_contents"]["README.md"] == "+patched"


def test_added_lines_with_positions_tracks_hunk_headers():
    patch = (
        "@@ -10,3 +10,5 @@ def existing():\n"
        " context line\n"
        "+first added\n"
        " another context\n"
        "-removed line\n"
        "+second added\n"
        "\\ No newline at end of file\n"
        "@@ -40,2 +42,3 @@\n"
        " ctx\n"
        "+third added\n"
    )
    assert GitHubClient._added_lines_with_positions(patch) == [
        (11, "first added"),
        (13, "second added"),
        (43, "third added"),
    ]


def test_added_lines_with_positions_junk_and_headerless():
    assert GitHubClient._added_lines_with_positions(None) == []
    assert GitHubClient._added_lines_with_positions("") == []
    # Header-less shorthand (test fixtures) counts from line 1 — back-compat.
    assert GitHubClient._added_lines_with_positions("+a = 1\n+b = 2") == [
        (1, "a = 1"),
        (2, "b = 2"),
    ]


def test_get_pull_request_data_line_map_aligns_with_code(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    items = [
        {"filename": "a.py", "patch": "@@ -0,0 +5,2 @@\n+one\n+two"},
        {"filename": "b.py", "patch": "@@ -1,1 +8,2 @@\n ctx\n+three"},
    ]
    monkeypatch.setattr(client, "get_pull_request_files", lambda *a, **k: items)

    data = client.get_pull_request_data("octo", "repo", 7)

    assert data["code"].splitlines() == ["one", "two", "three"]
    assert data["line_map"] == [("a.py", 5), ("a.py", 6), ("b.py", 9)]


def test_create_check_run_posts_payload_with_caps(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    monkeypatch.setattr(client, "_get_installation_token", lambda: "installation-token")
    captured: dict = {}

    def fake_json(method, url, headers, data=None):
        captured.update({"method": method, "url": url, "headers": headers, "data": data})
        return {"id": 55}

    monkeypatch.setattr(client, "_http_json", fake_json)
    annotations = [
        {
            "path": "a.py",
            "start_line": i,
            "end_line": i,
            "annotation_level": "failure",
            "message": "m",
            "title": "t",
        }
        for i in range(1, 60)
    ]
    ok = client.create_check_run(
        "octo",
        "repo",
        "sha123",
        conclusion="failure",
        title="T",
        summary="S",
        text="X" * 70_000,
        annotations=annotations,
    )

    assert ok is True
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/repos/octo/repo/check-runs")
    body = captured["data"]
    assert body["name"] == GitHubClient.CHECK_RUN_NAME
    assert body["head_sha"] == "sha123"
    assert body["status"] == "completed"
    assert body["conclusion"] == "failure"
    assert body["output"]["title"] == "T"
    assert len(body["output"]["annotations"]) == GitHubClient.MAX_CHECK_ANNOTATIONS
    assert len(body["output"]["text"]) == GitHubClient.MAX_CHECK_TEXT


def test_create_check_run_guards_and_failure(monkeypatch):
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    monkeypatch.setattr(
        client, "_get_installation_token", lambda: pytest.fail("guards must short-circuit")
    )
    kwargs = {"conclusion": "success", "title": "T", "summary": "S"}
    assert client.create_check_run("", "repo", "sha", **kwargs) is False
    assert client.create_check_run("octo", "repo", "", **kwargs) is False
    assert client.create_check_run("octo", "repo", "sha", conclusion="", title="T", summary="S") is False

    # Error response (e.g. 403: app lacks Checks:write) -> False, never raises.
    monkeypatch.setattr(client, "_get_installation_token", lambda: "installation-token")
    monkeypatch.setattr(client, "_http_json", lambda *a, **k: None)
    assert client.create_check_run("octo", "repo", "sha", **kwargs) is False


# --- Retry, throttling, and logging (M9 Tier 2) ---


def _client() -> GitHubClient:
    return GitHubClient(app_id="123", installation_id="1", private_key="private")


def _urlopen_sequence(monkeypatch, outcomes):
    """Serve one outcome per call: an Exception is raised, bytes are returned."""
    calls = {"n": 0}

    def fake_urlopen(request, timeout=10):
        _ = request, timeout
        outcome = outcomes[min(calls["n"], len(outcomes) - 1)]
        calls["n"] += 1
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeResponse(outcome)

    monkeypatch.setattr(gh, "jwt", types.SimpleNamespace(encode=lambda *a, **k: "t"))
    monkeypatch.setattr(gh.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(gh.GitHubClient, "_sleep", staticmethod(lambda _s: None))
    return calls


def test_transient_5xx_is_retried_then_succeeds(monkeypatch):
    calls = _urlopen_sequence(
        monkeypatch, [_FakeHTTPError("boom", code=503), b'{"id": 7}']
    )
    assert _client()._http_json("GET", "https://api.github.com/x", headers={}) == {"id": 7}
    assert calls["n"] == 2


def test_retries_are_bounded(monkeypatch):
    calls = _urlopen_sequence(monkeypatch, [_FakeHTTPError("boom", code=500)])
    assert _client()._http_json("GET", "https://api.github.com/x", headers={}) is None
    assert calls["n"] == GitHubClient.MAX_RETRIES + 1


def test_permission_errors_are_not_retried(monkeypatch):
    """401/404 are configuration problems; hammering them helps nobody."""
    for code in (401, 404):
        calls = _urlopen_sequence(monkeypatch, [_FakeHTTPError("nope", code=code)])
        assert _client()._http_json("GET", "https://api.github.com/x", headers={}) is None
        assert calls["n"] == 1, code


def test_secondary_rate_limit_honors_retry_after(monkeypatch):
    slept: list[float] = []
    throttled = _FakeHTTPError("slow down", code=403, headers={"Retry-After": "2"})
    calls = _urlopen_sequence(monkeypatch, [throttled, b"{}"])
    monkeypatch.setattr(gh.GitHubClient, "_sleep", staticmethod(slept.append))

    assert _client()._http_json("GET", "https://api.github.com/x", headers={}) == {}
    assert calls["n"] == 2
    assert slept == [2.0]


def test_primary_rate_limit_waits_until_reset(monkeypatch):
    slept: list[float] = []
    limited = _FakeHTTPError(
        "rate limited", code=403,
        headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(int(gh.GitHubClient._now()) + 3)},
    )
    _urlopen_sequence(monkeypatch, [limited, b"{}"])
    monkeypatch.setattr(gh.GitHubClient, "_sleep", staticmethod(slept.append))

    assert _client()._http_json("GET", "https://api.github.com/x", headers={}) == {}
    assert slept and 0 < slept[0] <= 3.0


def test_rate_limit_wait_is_clamped(monkeypatch):
    """A reset an hour out must not park a worker for an hour."""
    far = _FakeHTTPError(
        "rate limited", code=403,
        headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(int(gh.GitHubClient._now()) + 3600)},
    )
    assert GitHubClient._retry_after_seconds(far) == GitHubClient.MAX_RETRY_SLEEP


def test_failures_are_logged_with_their_status(monkeypatch, caplog):
    """The whole point: a 401 must be distinguishable from a clean run."""
    _urlopen_sequence(monkeypatch, [_FakeHTTPError("unauthorized", code=401)])
    with caplog.at_level("ERROR"):
        _client()._http_json("GET", "https://api.github.com/repos/o/r/pulls/1", headers={})
    messages = [r.getMessage() for r in caplog.records]
    assert any("401" in m and "/repos/o/r/pulls/1" in m for m in messages), messages


def test_log_label_omits_query_strings_and_host():
    label = GitHubClient._endpoint_label("https://api.github.com/repos/o/r/pulls?per_page=100&page=2")
    assert label == "/repos/o/r/pulls"


def test_list_endpoint_distinguishes_empty_body_from_failure(monkeypatch):
    _urlopen_sequence(monkeypatch, [b""])
    assert _client()._http_json_list("GET", "https://api.github.com/x", headers={}) == []

    _urlopen_sequence(monkeypatch, [b'{"not": "a list"}'])
    assert _client()._http_json_list("GET", "https://api.github.com/x", headers={}) is None


# --- Corpus relevance ranking (M10 Tier 4) ---


def test_corpus_noise_paths_are_excluded():
    """Test bodies repeat by nature and vendored trees aren't ours."""
    for path in ("tests/test_x.py", "app/test_helpers.py", "app/helpers_test.py",
                 "vendor/lib/x.py", "node_modules/pkg/y.py", "db/migrations/0001.py",
                 "build/gen.py", "a/__pycache__/z.py"):
        assert GitHubClient._is_corpus_noise(path), path
    for path in ("app/billing.py", "sentinel/domain/services/debt_service.py", "main.py",
                 "app/latest.py"):
        assert not GitHubClient._is_corpus_noise(path), path


def test_corpus_relevance_prefers_the_same_directory_then_package():
    changed = ["sentinel/domain/services/debt_service.py"]
    same_dir = GitHubClient._corpus_relevance("sentinel/domain/services/security_service.py", changed)
    same_top = GitHubClient._corpus_relevance("sentinel/api/webhook_controller.py", changed)
    unrelated = GitHubClient._corpus_relevance("scripts/tool.py", changed)
    assert same_dir > same_top > unrelated
    assert unrelated == 0


def test_corpus_relevance_is_neutral_without_a_preference():
    assert GitHubClient._corpus_relevance("any/path.py", None) == 0
    assert GitHubClient._corpus_relevance("any/path.py", []) == 0


def test_corpus_ranks_by_relevance_before_truncating(monkeypatch):
    """The point of ranking: a fixed budget spent on the PR's neighbours.

    Tree order would have taken the first CORPUS_MAX_FILES alphabetically,
    which is unrelated to what the PR touched.
    """
    tree = {"tree": [
        {"type": "blob", "path": f"aaa/filler_{i}.py", "sha": f"filler{i}", "size": 10}
        for i in range(GitHubClient.CORPUS_MAX_FILES)
    ] + [
        {"type": "blob", "path": "zzz/target.py", "sha": "target", "size": 10},
        {"type": "blob", "path": "zzz/tests/test_target.py", "sha": "noise", "size": 10},
    ]}
    fetched: list[str] = []

    def fake_http_json(method, url, headers, data=None):
        if "git/trees" in url:
            return tree
        fetched.append(url.rsplit("/", 1)[-1])
        return {"encoding": "base64", "content": base64.b64encode(b"x = 1").decode()}

    client = _client()
    monkeypatch.setattr(client, "_get_installation_token", lambda: "tok")
    monkeypatch.setattr(client, "_http_json", fake_http_json)

    client.get_repo_code_corpus("o", "r", "sha", prefer_paths=["zzz/changed.py"])

    assert "target" in fetched, "the PR's own directory must make the cut"
    assert "noise" not in fetched, "test files must be excluded"


def test_corpus_cache_is_keyed_by_preference(monkeypatch):
    """Same ref, different PR -> different ranking, so the ref alone can't be the key."""
    tree = {"tree": [
        {"type": "blob", "path": "a/one.py", "sha": "one", "size": 10},
        {"type": "blob", "path": "b/two.py", "sha": "two", "size": 10},
    ]}
    calls = {"n": 0}

    def fake_http_json(method, url, headers, data=None):
        if "git/trees" in url:
            return tree
        calls["n"] += 1
        return {"encoding": "base64", "content": base64.b64encode(b"x = 1").decode()}

    client = _client()
    monkeypatch.setattr(client, "_get_installation_token", lambda: "tok")
    monkeypatch.setattr(client, "_http_json", fake_http_json)

    client.get_repo_code_corpus("o", "r", "sha", prefer_paths=["a/x.py"])
    first = calls["n"]
    client.get_repo_code_corpus("o", "r", "sha", prefer_paths=["a/x.py"])
    assert calls["n"] == first, "identical preference should hit the cache"
    client.get_repo_code_corpus("o", "r", "sha", prefer_paths=["b/y.py"])
    assert calls["n"] > first, "a different preference must not reuse the cache"


# --- Token auth: the same client, running under a GitHub Action --------------


def test_access_token_short_circuits_the_app_jwt_exchange(monkeypatch):
    """A supplied token IS the credential: nothing is signed, nothing minted."""
    client = GitHubClient.from_access_token("ghs_supplied")
    monkeypatch.setattr(
        client, "_build_app_jwt", lambda: pytest.fail("must not sign an App JWT")
    )

    assert client._get_installation_token() == "ghs_supplied"


def test_from_access_token_carries_no_app_credentials():
    client = GitHubClient.from_access_token("ghs_supplied")
    assert (client.app_id, client.installation_id, client.private_key) == (None, None, None)


def test_access_token_authorises_with_the_token_scheme(monkeypatch):
    """GitHub's `token` scheme is what a GITHUB_TOKEN uses; Bearer is for the JWT."""
    seen = {}

    def fake_http_json(method, url, headers, data=None):
        seen.update(headers)
        return {"id": 1}

    client = GitHubClient.from_access_token("ghs_supplied")
    monkeypatch.setattr(client, "_http_json", fake_http_json)

    client.post_comment("o", "r", 1, "body")

    assert seen["Authorization"] == "token ghs_supplied"


def test_the_app_path_is_unchanged_when_no_access_token_is_set(monkeypatch):
    """An explicit guard, so a refactor cannot quietly invert the branch."""
    client = _client()
    assert client.access_token is None

    _patch_imports(monkeypatch, response_payload=b'{"token": "minted", "expires_at": ""}')
    assert client._get_installation_token() == "minted"


def test_full_source_fetches_are_capped_and_the_rest_fall_back_to_patch(monkeypatch):
    """The cap bounds API cost, and the overflow is a precision cliff, not a gap.

    A .py file past the budget still gets analyzed — by the regex engine, over
    its diff — so nothing is skipped. But regex cannot distinguish a credential
    from a string that looks like one, which is the whole reason the AST engine
    exists. This pins the mechanism so the trade-off stays visible; the cap's
    value is tuned separately.
    """
    client = GitHubClient(app_id="123", installation_id="999", private_key="private")
    over = client.MAX_SOURCE_FETCHES + 3
    items = [
        {"filename": f"m{i:03d}.py", "patch": "+x = 1", "contents_url": f"https://api.test/c/{i}"}
        for i in range(over)
    ]
    monkeypatch.setattr(client, "get_pull_request_files", lambda *a, **k: items)

    fetched: list[str] = []

    def fake_content(url):
        fetched.append(url)
        return "FULL SOURCE"

    monkeypatch.setattr(client, "get_content_by_url", fake_content)
    data = client.get_pull_request_data("octo", "repo", 7)

    assert len(fetched) == client.MAX_SOURCE_FETCHES
    # Everything still has content; only its fidelity differs.
    assert len(data["file_contents"]) == over
    assert data["file_contents"]["m000.py"] == "FULL SOURCE"
    assert data["file_contents"][f"m{over - 1:03d}.py"] == "+x = 1"


def test_the_fetch_budget_covers_a_review_sized_pull_request():
    """Pinned because 20 was not enough, and the symptom was a false positive.

    A PR touching 26 Python files overflowed the old budget and the tool
    reported test fixtures as hardcoded credentials. A regression here is
    silent — the review still posts, it is just wrong — so assert the floor.
    """
    assert GitHubClient.MAX_SOURCE_FETCHES >= 40
