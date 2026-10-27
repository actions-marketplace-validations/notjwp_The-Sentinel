import base64
import datetime
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - exercised implicitly; absence is handled at call time
    import jwt
except ImportError:  # pragma: no cover
    jwt = None

from sentinel.domain.services.document_service import DocumentService
from sentinel.monitoring.logger import get_logger
from sentinel.monitoring.metrics import metrics

logger = get_logger(__name__)


@dataclass
class GitHubClient:
    app_id: str | None
    installation_id: str | None
    private_key: str | None
    api_base_url: str = "https://api.github.com"
    timeout: float = 10.0
    # A ready-made token, used INSTEAD of the App JWT exchange. This is what a
    # GitHub Action has: the runner is handed a GITHUB_TOKEN scoped to the one
    # repository, so there is no app to authenticate as and no installation to
    # exchange for. Defaulted and last, so every existing construction is
    # unaffected.
    access_token: str | None = None

    # Transient statuses worth another attempt. 403 is deliberately absent: it is
    # usually a permission problem, and only retried when it carries throttling
    # headers (see _retry_after_seconds).
    RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
    MAX_RETRIES = 2  # 3 attempts total
    RETRY_BASE_SECONDS = 0.25
    MAX_RETRY_SLEEP = 8.0

    # Hidden marker embedded in Sentinel's own PR comments so re-runs can find and
    # update the existing review in place instead of stacking duplicates. Bare (no
    # annotation) so @dataclass treats it as a class attribute, not a field.
    SENTINEL_COMMENT_MARKER = "<!-- sentinel-review -->"

    # Upper bound on pages fetched per list endpoint (10 x per_page=100 = 1000 items).
    MAX_LIST_PAGES = 10

    # Bounds for repo-content fetches (semantic corpus + full doc content).
    CORPUS_MAX_FILES = 20  # most .py blobs fetched per corpus build
    # Distinct corpora held per client. Deliberately small, because the cost
    # is multiplicative: one worker keeps up to MAX_CACHED_CLIENTS clients,
    # each with its own cache, each entry up to CORPUS_MAX_FILES files. Hit
    # rate is the lever worth pulling here, not capacity — see the key below.
    CORPUS_CACHE_ENTRIES = 4
    # Concurrent blob fetches. Sized to overlap network latency, not to push
    # throughput: every request spends the same installation token, and the
    # rate limit is per installation rather than per connection.
    CORPUS_FETCH_WORKERS = 8
    # Full-source fetches for AST analysis cost one API call per file, so the
    # budget is capped; files beyond it fall back to regex over their diff.
    #
    # The cap is a precision budget as much as a cost one, which is why it is
    # not tight. A .py file that misses out is analyzed by the regex engine,
    # and regex cannot tell a credential from a string that looks like one —
    # at 20, a 26-file PR had its last six files scanned that way and the tool
    # reported two of its own test fixtures as hardcoded API keys. Files arrive
    # in GitHub's order (path-sorted), so the ones that lose are consistently
    # whatever sorts last, which in this repo is the test suite. 50 covers any
    # review-sized PR; against the Action's 1,000 req/hr token it is cheap.
    MAX_SOURCE_FETCHES = 50
    CORPUS_MAX_FILE_BYTES = 50_000  # skip larger blobs (tree items carry size)
    MAX_CONTENT_BYTES = 200_000  # hard cap on any decoded file content

    # Check-run limits (the Checks API accepts at most 50 annotations per request;
    # output.text is capped server-side at 65535 chars — stay under it).
    CHECK_RUN_NAME = "Sentinel AI Code Review"
    MAX_CHECK_ANNOTATIONS = 50
    MAX_CHECK_TEXT = 60_000

    # Unified-diff hunk header: '@@ -old_start,old_count +new_start,new_count @@'.
    _HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")

    @classmethod
    def from_access_token(
        cls,
        access_token: str,
        *,
        api_base_url: str = "https://api.github.com",
        timeout: float = 10.0,
    ) -> "GitHubClient":
        """A client authenticated by a supplied token rather than as a GitHub App.

        This is the GitHub Action's constructor: the runner hands the job a
        token already scoped to the one repository, so there is no app to sign
        as and no installation to exchange.
        """
        return cls(
            app_id=None,
            installation_id=None,
            private_key=None,
            api_base_url=api_base_url,
            timeout=timeout,
            access_token=access_token,
        )

    def __post_init__(self) -> None:
        self._installation_token: str | None = None
        self._installation_token_expiry: float = 0.0
        # Corpus per (owner, repo, ref, directories); a ref pins content, so
        # entries never go stale — capacity-bounded only. A plain dict keeps
        # insertion order, and a hit reinserts, which makes eviction LRU.
        self._corpus_cache: dict[tuple[str, str, str, str], list[str]] = {}

    @staticmethod
    def _now() -> float:
        return float(time.time())

    @staticmethod
    def _sleep(seconds: float) -> None:
        """Isolated so tests can neutralize retry backoff without real waiting."""
        try:
            time.sleep(seconds)
        except Exception:  # pragma: no cover - defensive
            return

    @staticmethod
    def _parse_expiry(expires_at: str | None) -> float:
        if not expires_at:
            return GitHubClient._now() + 300.0
        try:
            parsed = datetime.datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            return float(parsed.timestamp())
        except Exception:
            return GitHubClient._now() + 300.0

    def _build_app_jwt(self) -> str | None:
        if not self.app_id or not self.private_key:
            return None
        if jwt is None:
            logger.error("PyJWT is not installed; cannot authenticate as a GitHub App")
            return None

        now = int(self._now())
        payload = {
            "iat": now - 60,
            "exp": now + 540,
            "iss": self.app_id,
        }

        try:
            token = jwt.encode(payload, self.private_key, algorithm="RS256")
        except Exception:
            # Almost always a malformed GITHUB_PRIVATE_KEY. Never log the key.
            logger.exception("Failed to sign the GitHub App JWT; check GITHUB_PRIVATE_KEY")
            return None

        if isinstance(token, bytes):
            return token.decode("utf-8")
        return str(token)

    @staticmethod
    def _endpoint_label(url: str) -> str:
        """Path only, for logs. Keeps query strings and any host credentials out."""
        without_scheme = url.split("://", 1)[-1]
        path = without_scheme.split("/", 1)[1] if "/" in without_scheme else without_scheme
        return "/" + path.split("?", 1)[0]

    @classmethod
    def _retry_after_seconds(cls, exc: Exception) -> float | None:
        """Seconds GitHub asked us to wait, or None if it didn't.

        Covers both throttles: secondary limits send ``Retry-After``, primary
        limits send ``X-RateLimit-Remaining: 0`` plus an absolute
        ``X-RateLimit-Reset`` epoch. Clamped — a reset can be an hour out, and
        blocking a worker that long is worse than failing the review.
        """
        headers = getattr(exc, "headers", None)
        if headers is None:
            return None
        try:
            retry_after = headers.get("Retry-After")
            if retry_after is not None:
                return max(0.0, min(float(retry_after), cls.MAX_RETRY_SLEEP))
            if str(headers.get("X-RateLimit-Remaining")) == "0":
                reset = headers.get("X-RateLimit-Reset")
                if reset is not None:
                    return max(0.0, min(float(reset) - cls._now(), cls.MAX_RETRY_SLEEP))
        except (TypeError, ValueError):
            return None
        return None

    def _request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        data: dict[str, Any] | None = None,
    ) -> Any | None:
        """The one HTTP primitive. Returns parsed JSON, or None on failure.

        Retries transient failures (5xx, connection errors) and throttles, honoring
        ``Retry-After`` when GitHub sends one. Every failure is logged with its
        status — previously this layer swallowed 401s, 403s and 404s identically
        and silently, which made a misconfigured install indistinguishable from a
        clean run. Headers are never logged: they carry the installation token.
        """
        payload_bytes = json.dumps(data).encode("utf-8") if data is not None else None
        label = self._endpoint_label(url)

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                request = urllib.request.Request(
                    url=url, data=payload_bytes, headers=headers, method=method
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
                if raw.strip() == "":
                    return {}
                return json.loads(raw)

            except urllib.error.HTTPError as exc:
                status = getattr(exc, "code", None)
                retry_after = self._retry_after_seconds(exc)
                throttled = retry_after is not None
                if (status in self.RETRYABLE_STATUS or throttled) and attempt < self.MAX_RETRIES:
                    delay = retry_after if throttled else self._backoff_seconds(attempt)
                    logger.warning(
                        "GitHub %s %s -> HTTP %s%s; retrying in %.1fs (attempt %s/%s)",
                        method, label, status, " (throttled)" if throttled else "",
                        delay, attempt + 1, self.MAX_RETRIES,
                    )
                    metrics.counter_inc("sentinel_github_api_errors_total",
                                        {"status": str(status), "outcome": "retried"})
                    self._sleep(delay)
                    continue
                logger.error("GitHub %s %s failed: HTTP %s", method, label, status)
                metrics.counter_inc("sentinel_github_api_errors_total",
                                    {"status": str(status), "outcome": "failed"})
                return None

            except urllib.error.URLError as exc:
                if attempt < self.MAX_RETRIES:
                    delay = self._backoff_seconds(attempt)
                    logger.warning(
                        "GitHub %s %s unreachable (%s); retrying in %.1fs",
                        method, label, exc.reason, delay,
                    )
                    self._sleep(delay)
                    continue
                logger.error("GitHub %s %s unreachable: %s", method, label, exc.reason)
                metrics.counter_inc("sentinel_github_api_errors_total",
                                    {"status": "network", "outcome": "failed"})
                return None

            except Exception:
                logger.exception("GitHub %s %s raised unexpectedly", method, label)
                metrics.counter_inc("sentinel_github_api_errors_total",
                                    {"status": "unknown", "outcome": "failed"})
                return None

        return None

    @classmethod
    def _backoff_seconds(cls, attempt: int) -> float:
        return min(cls.RETRY_BASE_SECONDS * (2**attempt), cls.MAX_RETRY_SLEEP)

    def _http_json(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """A JSON object endpoint. {} for a non-object body, None on failure."""
        parsed = self._request(method, url, headers, data)
        if parsed is None:
            return None
        return parsed if isinstance(parsed, dict) else {}

    def _http_json_list(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        data: dict[str, Any] | None = None,
    ) -> list[Any] | None:
        """A JSON array endpoint (e.g. list PR files). None on failure or non-array.

        Distinct from _http_json because that one flattens non-objects to {},
        which would turn a page of results into "empty page" and silently end
        pagination.
        """
        parsed = self._request(method, url, headers, data)
        if parsed is None:
            return None
        if isinstance(parsed, dict) and not parsed:
            return []  # empty body
        return parsed if isinstance(parsed, list) else None

    def _get_paginated(self, endpoint: str, headers: dict[str, str]) -> list[dict[str, Any]]:
        """GET all pages of a GitHub list endpoint (up to MAX_LIST_PAGES).

        Failure-safe: a non-list page (error) ends pagination, returning whatever
        was collected so far — partial results beat none for a best-effort client.
        """
        results: list[dict[str, Any]] = []
        for page in range(1, self.MAX_LIST_PAGES + 1):
            url = f"{endpoint}?per_page=100&page={page}"
            batch = self._http_json_list("GET", url, headers, data=None)
            if not isinstance(batch, list):
                break
            results.extend(item for item in batch if isinstance(item, dict))
            if len(batch) < 100:  # short page == last page
                break
        return results

    def _get_installation_token(self) -> str | None:
        # A supplied token wins outright: there is nothing to mint, nothing to
        # cache and nothing to expire. Every caller goes through here, so this
        # one branch is what lets the whole client run under an Action's
        # GITHUB_TOKEN without a second code path.
        if self.access_token:
            return self.access_token

        if self._installation_token and self._now() < (self._installation_token_expiry - 30):
            return self._installation_token

        if not self.installation_id:
            return None

        app_jwt = self._build_app_jwt()
        if not app_jwt:
            return None

        endpoint = (
            f"{self.api_base_url.rstrip('/')}/app/installations/"
            f"{self.installation_id}/access_tokens"
        )
        response = self._http_json(
            "POST",
            endpoint,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {app_jwt}",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
            },
            data={},
        )

        if not response:
            return None

        token = response.get("token")
        if not isinstance(token, str) or token.strip() == "":
            return None

        self._installation_token = token
        self._installation_token_expiry = self._parse_expiry(response.get("expires_at"))
        return self._installation_token

    def post_comment(self, owner: str, repo: str, pr_number: int, body: str) -> bool:
        if not owner or not repo or not body:
            return False

        token = self._get_installation_token()
        if not token:
            return False

        endpoint = (
            f"{self.api_base_url.rstrip('/')}/repos/{owner}/{repo}"
            f"/issues/{pr_number}/comments"
        )
        response = self._http_json(
            "POST",
            endpoint,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"token {token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
            },
            data={"body": body},
        )

        return isinstance(response, dict) and isinstance(response.get("id"), int)

    def list_issue_comments(self, owner: str, repo: str, pr_number: int) -> list[dict[str, Any]]:
        """GET the issue comments on a PR (all pages, capped). Returns [] on any failure."""
        if not owner or not repo:
            return []

        token = self._get_installation_token()
        if not token:
            return []

        endpoint = (
            f"{self.api_base_url.rstrip('/')}/repos/{owner}/{repo}"
            f"/issues/{pr_number}/comments"
        )
        return self._get_paginated(
            endpoint,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"token {token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

    def update_comment(self, owner: str, repo: str, comment_id: int, body: str) -> bool:
        """PATCH an existing issue comment in place. Returns False on any failure."""
        if not owner or not repo or not body:
            return False

        token = self._get_installation_token()
        if not token:
            return False

        endpoint = (
            f"{self.api_base_url.rstrip('/')}/repos/{owner}/{repo}"
            f"/issues/comments/{comment_id}"
        )
        response = self._http_json(
            "PATCH",
            endpoint,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"token {token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
            },
            data={"body": body},
        )

        return isinstance(response, dict) and isinstance(response.get("id"), int)

    def upsert_comment(self, owner: str, repo: str, pr_number: int, body: str) -> bool:
        """Post Sentinel's review, or update its existing comment in place.

        Idempotent: the body carries a hidden ``SENTINEL_COMMENT_MARKER`` so later
        runs locate the prior review and PATCH it instead of stacking new comments.
        Falls back to creating a comment when none is found or the lookup fails.
        """
        if not owner or not repo or not body:
            return False

        if self.SENTINEL_COMMENT_MARKER in body:
            marked_body = body
        else:
            marked_body = f"{self.SENTINEL_COMMENT_MARKER}\n{body}"

        try:
            for comment in self.list_issue_comments(owner, repo, pr_number):
                existing_body = comment.get("body")
                comment_id = comment.get("id")
                if (
                    isinstance(existing_body, str)
                    and self.SENTINEL_COMMENT_MARKER in existing_body
                    and isinstance(comment_id, int)
                ):
                    return self.update_comment(owner, repo, comment_id, marked_body)
        except Exception:
            pass

        return self.post_comment(owner, repo, pr_number, marked_body)

    @staticmethod
    def _added_lines_with_positions(patch: str | None) -> list[tuple[int, str]]:
        """Extract a patch's added lines paired with their line numbers at head.

        Hunk headers ('@@ -a,b +c,d @@') reset the new-file line counter; context
        lines advance it; '+' lines advance it AND are collected; removed ('-')
        and '\\ No newline...' lines don't touch it. Header-less shorthand (as in
        test fixtures) counts from line 1. [] for non-string/empty input.
        """
        if not isinstance(patch, str) or not patch:
            return []
        added: list[tuple[int, str]] = []
        new_line = 1
        for line in patch.splitlines():
            header = GitHubClient._HUNK_HEADER_RE.match(line)
            if header:
                new_line = int(header.group(1))
                continue
            if line.startswith("+++") or line.startswith("---"):
                continue
            if line.startswith("+"):
                added.append((new_line, line[1:]))
                new_line += 1
            elif line.startswith("-") or line.startswith("\\"):
                continue
            else:
                new_line += 1
        return added

    @staticmethod
    def _added_lines_from_patch(patch: str | None) -> str:
        """Reconstruct the code a PR introduces from a unified-diff patch hunk."""
        return "\n".join(
            text for _, text in GitHubClient._added_lines_with_positions(patch)
        )

    def get_pull_request_files(self, owner: str, repo: str, pr_number: int) -> list[dict[str, Any]]:
        """GET the changed files of a PR (all pages, capped). Returns [] on any failure."""
        if not owner or not repo:
            return []

        token = self._get_installation_token()
        if not token:
            return []

        endpoint = (
            f"{self.api_base_url.rstrip('/')}/repos/{owner}/{repo}"
            f"/pulls/{pr_number}/files"
        )
        return self._get_paginated(
            endpoint,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"token {token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

    def _token_headers(self, token: str) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"token {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def get_pull_request_refs(self, owner: str, repo: str, pr_number: int) -> dict[str, str]:
        """GET the PR's head/base commit SHAs. Returns {} on any failure."""
        if not owner or not repo:
            return {}

        token = self._get_installation_token()
        if not token:
            return {}

        endpoint = f"{self.api_base_url.rstrip('/')}/repos/{owner}/{repo}/pulls/{pr_number}"
        response = self._http_json("GET", endpoint, headers=self._token_headers(token))
        if not isinstance(response, dict):
            return {}

        refs: dict[str, str] = {}
        for key in ("head", "base"):
            sha = response.get(key, {}).get("sha") if isinstance(response.get(key), dict) else None
            if isinstance(sha, str) and sha:
                refs[f"{key}_sha"] = sha
        return refs

    def _decode_content_response(self, response: dict[str, Any] | None) -> str | None:
        """Decode a contents/blob API response body to text.

        None = failure/unsupported encoding; "" = a genuinely empty file. The
        distinction matters: callers fall back to patch text only on failure.
        """
        if not isinstance(response, dict):
            return None
        if response.get("encoding") != "base64":
            return None
        content = response.get("content")
        if not isinstance(content, str):
            return None
        if content.strip() == "":
            return ""
        try:
            # The API wraps base64 with newlines; b64decode tolerates them only
            # after they're stripped.
            raw = base64.b64decode(content.replace("\n", ""), validate=False)
        except Exception:
            return None
        return raw[: self.MAX_CONTENT_BYTES].decode("utf-8", errors="replace")

    def get_content_by_url(self, contents_url: str) -> str | None:
        """GET a PR file's full content via its contents_url (pinned to head ref).

        Returns the decoded text, "" for an empty file, or None on any failure.
        """
        if not isinstance(contents_url, str) or not contents_url.startswith("http"):
            return None

        token = self._get_installation_token()
        if not token:
            return None

        response = self._http_json("GET", contents_url, headers=self._token_headers(token))
        return self._decode_content_response(response)

    # Paths whose contents say nothing useful about duplication in application
    # code: test bodies repeat by nature, and vendored trees are not ours.
    CORPUS_NOISE_SEGMENTS = (
        "test", "tests", "vendor", "vendored", "third_party", "node_modules",
        "migrations", "site-packages", "build", "dist", "__pycache__",
    )

    @classmethod
    def _is_corpus_noise(cls, path: str) -> bool:
        segments = path.replace("\\", "/").lower().split("/")
        if any(segment in cls.CORPUS_NOISE_SEGMENTS for segment in segments[:-1]):
            return True
        filename = segments[-1]
        return filename.startswith("test_") or filename.endswith("_test.py")

    @staticmethod
    def _preference_key(prefer_paths: list[str] | None) -> str:
        """The part of prefer_paths that can change a ranking, as a cache key.

        Must stay in lockstep with ``_corpus_relevance``: it looks at a changed
        path's directory and its top-level package, and nothing else. The
        top-level is a prefix of the directory, so the directory set alone
        determines every score the ranking can produce.
        """
        directories = {
            p.replace("\\", "/").rsplit("/", 1)[0] if "/" in p.replace("\\", "/") else ""
            for p in (prefer_paths or [])
            if isinstance(p, str) and p
        }
        return ",".join(sorted(directories))

    @staticmethod
    def _corpus_relevance(path: str, prefer_paths: list[str] | None) -> int:
        """How likely this file is to be what the PR duplicated.

        Proximity is the whole heuristic: code is most often copied from a
        neighbour. Same directory scores highest, same top-level package next,
        everything else zero — which still keeps a sensible corpus when the PR's
        own paths are unknown.
        """
        if not prefer_paths:
            return 0
        normalized = path.replace("\\", "/")
        directory = normalized.rsplit("/", 1)[0] if "/" in normalized else ""
        top = normalized.split("/", 1)[0] if "/" in normalized else ""

        score = 0
        for changed in prefer_paths:
            if not isinstance(changed, str) or not changed:
                continue
            changed = changed.replace("\\", "/")
            changed_dir = changed.rsplit("/", 1)[0] if "/" in changed else ""
            changed_top = changed.split("/", 1)[0] if "/" in changed else ""
            if directory and directory == changed_dir:
                return 2  # same directory — nothing beats this
            if top and top == changed_top:
                score = max(score, 1)
        return score

    def _fetch_blobs(
        self, base: str, owner: str, repo: str, shas: list[str], token: str
    ) -> list[str]:
        """Fetch blobs concurrently, preserving the caller's ranking order.

        These are independent round trips with nothing to coordinate, and they
        dominated the review: 20 of them took 7.8 seconds of the 45 a real run
        spent, against ~0.0s for all three analysis engines put together. The
        pool is small because the point is to overlap network latency, not to
        hammer the API — and because every request shares one installation
        token whose rate limit is per installation, not per connection.

        Order matters: the caller ranks candidates by relevance to the PR
        before truncating, and `detect_duplicates` compares against the corpus
        in sequence, so results are reassembled by index rather than by
        completion. Failure-safe per blob — one that fails or decodes empty is
        dropped, exactly as the sequential version dropped it.
        """
        if not shas:
            return []

        def fetch(sha: str) -> str | None:
            try:
                blob = self._http_json(
                    "GET",
                    f"{base}/repos/{owner}/{repo}/git/blobs/{sha}",
                    headers=self._token_headers(token),
                )
                return self._decode_content_response(blob)
            except Exception:
                logger.exception("Corpus blob fetch failed; skipping it")
                return None

        workers = min(self.CORPUS_FETCH_WORKERS, len(shas))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            texts = list(pool.map(fetch, shas))

        return [text for text in texts if text]

    def _rank_corpus_blobs(self, tree: list, prefer_paths: list[str] | None) -> list[str]:
        """Pick which blobs to fetch for the corpus, best candidates first.

        Rank before truncating. Taking the first N in tree order meant the
        corpus was whatever sorted earliest — often a directory of __init__.py
        files — regardless of what the PR touched. Sorting by relevance first
        spends a fixed budget on the code the PR is most likely to duplicate.

        Every field is re-validated because this is API output: a tree item
        missing ``size`` is kept rather than skipped, and falls back to the
        MAX_CONTENT_BYTES truncation when its content is decoded.
        """
        candidates: list[tuple[int, str, str]] = []  # (-score, path, sha)
        for item in tree:
            if not isinstance(item, dict) or item.get("type") != "blob":
                continue
            path = item.get("path")
            sha = item.get("sha")
            size = item.get("size")
            if not isinstance(path, str) or not path.endswith(".py") or path.startswith("."):
                continue
            if not isinstance(sha, str) or not sha:
                continue
            if isinstance(size, int) and size > self.CORPUS_MAX_FILE_BYTES:
                continue
            if self._is_corpus_noise(path):
                continue
            candidates.append((-self._corpus_relevance(path, prefer_paths), path, sha))

        candidates.sort()
        return [sha for _, _, sha in candidates[: self.CORPUS_MAX_FILES]]

    def get_repo_code_corpus(
        self, owner: str, repo: str, ref: str, prefer_paths: list[str] | None = None
    ) -> list[str]:
        """Fetch the repo's Python sources at a ref (the semantic-duplicate corpus).

        Bounded (CORPUS_MAX_FILES / CORPUS_MAX_FILE_BYTES) and failure-safe:
        partial results beat none; [] when nothing could be fetched. Cached per
        (owner, repo, ref) — a ref pins content, so cache entries never go stale.
        """
        if not owner or not repo or not ref:
            return []

        # prefer_paths is part of the key: the same ref yields a different
        # corpus for a PR touching a different directory, so keying on the ref
        # alone would serve one PR's ranking to the next.
        #
        # Keyed on the DIRECTORIES, not the filenames. _corpus_relevance reads
        # nothing else — it compares a candidate's directory and its top-level
        # package, both derivable from the directory — so two path lists that
        # differ only in filenames rank identically and may share an entry.
        # Keying on full paths made the key near PR-unique: a second push that
        # touched one more file missed, which is most of why this cache almost
        # never hit.
        cache_key = (owner, repo, ref, self._preference_key(prefer_paths))
        cached = self._corpus_cache.get(cache_key)
        if cached is not None:
            # Reinsert so recency, not age, decides the next eviction.
            self._corpus_cache[cache_key] = self._corpus_cache.pop(cache_key)
            return cached

        token = self._get_installation_token()
        if not token:
            return []

        base = self.api_base_url.rstrip("/")
        tree_response = self._http_json(
            "GET",
            f"{base}/repos/{owner}/{repo}/git/trees/{ref}?recursive=1",
            headers=self._token_headers(token),
        )
        tree = tree_response.get("tree") if isinstance(tree_response, dict) else None
        if not isinstance(tree, list):
            return []

        blob_shas = self._rank_corpus_blobs(tree, prefer_paths)
        corpus = self._fetch_blobs(base, owner, repo, blob_shas, token)

        self._corpus_cache[cache_key] = corpus
        while len(self._corpus_cache) > self.CORPUS_CACHE_ENTRIES:
            # First key is the least recently used: writes append, and a hit
            # above reinserts.
            self._corpus_cache.pop(next(iter(self._corpus_cache)))
        return corpus

    def get_pull_request_data(self, owner: str, repo: str, pr_number: int) -> dict[str, Any]:
        """Assemble everything a review needs from one PR-files fetch.

        Returns ``{"code": str, "files": list[str], "file_contents": dict[str, str]}``:
        added-line code joined across patches, the changed file names, and per-file
        content for the doc reviewer. Documentation files (.md/.txt) carry their FULL
        text fetched from the PR head (via the item's contents_url), so doc rules judge
        the whole document rather than a patch fragment; on fetch failure — and for all
        other files — the patch text remains the content source (the M3 behavior, which
        also mirrors the sync path's patch fallback). Empty values on failure.

        ``line_map`` has one ``(filename, head_line)`` entry per line of ``code``
        (built in the same loop, exact parallel), so a finding's line number in
        the assembled code resolves to a real file location for check annotations.
        """
        items = self.get_pull_request_files(owner, repo, pr_number)
        segments: list[str] = []
        files: list[str] = []
        file_contents: dict[str, str] = {}
        line_map: list[tuple[str, int]] = []
        source_fetches = 0
        for item in items:
            name = item.get("filename")
            patch = item.get("patch")
            if isinstance(name, str) and name:
                files.append(name)
                full_content: str | None = None
                lowered = name.strip().lower()
                # Docs are judged on their whole text; .py files need full source
                # because an AST needs a complete module — a diff fragment does
                # not parse. Both come from the same contents_url fetch.
                wants_full_source = lowered.endswith(DocumentService.DOC_EXTENSIONS) or (
                    lowered.endswith(".py") and source_fetches < self.MAX_SOURCE_FETCHES
                )
                if wants_full_source:
                    contents_url = item.get("contents_url")
                    if isinstance(contents_url, str):
                        full_content = self.get_content_by_url(contents_url)
                        if lowered.endswith(".py"):
                            source_fetches += 1
                if full_content is not None:
                    file_contents[name] = full_content
                elif isinstance(patch, str) and patch:
                    file_contents[name] = patch
            positioned = self._added_lines_with_positions(patch)
            if positioned:
                segments.append("\n".join(text for _, text in positioned))
                map_name = name if isinstance(name, str) and name else "unknown"
                line_map.extend((map_name, line_no) for line_no, _ in positioned)
        return {
            "code": "\n".join(segments),
            "files": files,
            "file_contents": file_contents,
            "line_map": line_map,
        }

    def get_pull_request_code(self, owner: str, repo: str, pr_number: int) -> str:
        """Assemble the added-line code introduced by a PR. Returns "" on failure."""
        return self.get_pull_request_data(owner, repo, pr_number)["code"]

    def create_check_run(
        self,
        owner: str,
        repo: str,
        head_sha: str,
        *,
        conclusion: str,
        title: str,
        summary: str,
        text: str | None = None,
        annotations: list[dict[str, Any]] | None = None,
    ) -> bool:
        """POST a completed check run for a commit. Returns False on any failure.

        Renders in the PR's Checks tab / merge box; annotations point at file
        lines at the head ref. Requires the app's "Checks: write" permission —
        without it the POST fails and this returns False (callers treat that as
        a skipped, non-fatal step).
        """
        if not owner or not repo or not head_sha or not conclusion:
            return False

        token = self._get_installation_token()
        if not token:
            return False

        output: dict[str, Any] = {
            "title": title or self.CHECK_RUN_NAME,
            "summary": summary or "",
        }
        if isinstance(text, str) and text:
            output["text"] = text[: self.MAX_CHECK_TEXT]
        if annotations:
            output["annotations"] = list(annotations)[: self.MAX_CHECK_ANNOTATIONS]

        endpoint = f"{self.api_base_url.rstrip('/')}/repos/{owner}/{repo}/check-runs"
        response = self._http_json(
            "POST",
            endpoint,
            headers={**self._token_headers(token), "Content-Type": "application/json"},
            data={
                "name": self.CHECK_RUN_NAME,
                "head_sha": head_sha,
                "status": "completed",
                "conclusion": conclusion,
                "output": output,
            },
        )

        return isinstance(response, dict) and isinstance(response.get("id"), int)
