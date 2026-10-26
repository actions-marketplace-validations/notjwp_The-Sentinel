"""End-to-end check of a locally running Sentinel, against real GitHub.

    .venv\\Scripts\\python.exe e2e.py                    # synthetic PR, self-cleaning
    .venv\\Scripts\\python.exe e2e.py --pr 12            # review an existing PR
    .venv\\Scripts\\python.exe e2e.py --repo owner/name --pr 3

Two modes:

REVIEW (--pr N)
    Trigger a review of a pull request that already exists and report what
    Sentinel posted. Works for any repo the App is installed on, any PR, any
    language, any file type. Asserts only that a review was actually produced —
    there is nothing to compare against, because nobody declared what the PR
    contains.

SYNTHETIC (default)
    Plant text in an existing file on a throwaway branch, open a PR, review it,
    and assert the deployed service found what it should.

    "What it should" is not written down here. The same engines that run inside
    the container are run locally over the planted text, and their verdict is
    the expectation. So the payload can be anything — a preset, a string, or a
    file — and the assertion stays meaningful without anyone maintaining a table
    of expected severities. It doubles as a consistency check: local engines and
    deployed service must agree.

    The text is planted away from line 1 on purpose. A PR that adds a new file
    maps assembled-code line N to file line N, so it passes whether or not the
    patch hunk arithmetic is right; only editing an existing file mid-way can
    catch a regression in the M6 line_map.

Options:
    --pr N              review an existing PR instead of creating one
    --repo o/n          target repo                (default notjwp/test-repo)
    --workdir DIR       local clone                (default D:/test-repo)
    --file PATH         file to edit               (default: longest tracked file)
    --payload NAME      a preset (see --list-payloads)
    --payload-text STR  plant this literal text instead of a preset
    --payload-file PATH plant the contents of this file
    --list-payloads     show the presets and exit
    --no-start / --down / --keep
"""

import argparse
import ast
import hashlib
import hmac
import json
import pathlib
import secrets
import string
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sentinel.application.risk_engine import RiskEngine  # noqa: E402
from sentinel.config.settings import get_settings  # noqa: E402
from sentinel.domain.services.ast_security_service import ASTSecurityService  # noqa: E402
from sentinel.infrastructure.github.github_client import GitHubClient  # noqa: E402

BASE_URL = "http://localhost:8000"
SEVERITY_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def _nonce(length: int = 20, alphabet: str = string.ascii_letters + string.digits) -> str:
    return "".join(secrets.choice(alphabet) for _ in range(length))


# Convenience shorthands only — NOT the mechanism. Nothing here declares an
# expected outcome; expectations are computed by the local engines at run time,
# so an unlisted payload works exactly as well as a listed one.
#
# Each is a callable so credential-shaped payloads get FRESH values per run.
# A literal committed on every run is a standing fake credential in the repo's
# history, is liable to trip GitHub push protection or a secret scanner, and
# makes one run's findings indistinguishable from another's. The generated
# values still satisfy the detection regexes (`sk-[A-Za-z0-9]{20,}`,
# `AKIA[0-9A-Z]{16}`) but carry an "e2e" marker so their provenance is obvious
# to anyone who finds one.
#
# SAFETY: every value below is inert text. Nothing is evaluated, imported, or
# executed by this script — it is written into a throwaway branch of a scratch
# repo so Sentinel's regex engine can flag it, and the branch is deleted at the
# end. The eval/os.system entries exist precisely BECAUSE they are dangerous
# patterns: they are what the rules are supposed to catch.
PAYLOADS = {
    "secrets": lambda: (
        f'password = "e2e-{_nonce(12)}"\n'
        f'api_key = "sk-e2e{_nonce(20)}"'
    ),
    "aws": lambda: f'aws_key = "AKIAE2E{_nonce(13, string.ascii_uppercase + string.digits)}"',
    "sql": lambda: f'query = f"SELECT * FROM users WHERE id = {{user_id_{_nonce(4)}}}"',
    "eval": lambda: f"result = eval(user_input_{_nonce(4)})",
    "command": lambda: f"os.system(user_cmd_{_nonce(4)})",
    # Negative control: must come back LOW with no findings, proving the
    # harness can distinguish a real detection from crying wolf.
    "clean": lambda: f"total_{_nonce(4)} = sum(values)",
}

# Below this, the insertion point is too close to line 1 to distinguish correct
# hunk arithmetic from a naive line-1 assumption.
MIN_FILE_LINES = 6


def say(step, msg):
    print(f"[{step}] {msg}", flush=True)


def run(cmd, cwd=None, check=True):
    """Run a command and return its stdout.

    Decoding is pinned to UTF-8 with replacement rather than left to the
    platform default: on Windows that default is cp1252, and docker's build
    output contains bytes it cannot decode. The failure is nasty — the reader
    thread dies inside subprocess, stdout comes back as None, and the caller
    trips over .strip() with a traceback that points nowhere near the cause.
    """
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True,
        encoding="utf-8", errors="replace",
    )
    stdout = result.stdout or ""
    if check and result.returncode != 0:
        print(stdout + (result.stderr or ""), file=sys.stderr)
        raise SystemExit(f"command failed: {' '.join(cmd)}")
    return stdout.strip()


def wait_for(predicate, timeout, interval=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return None


def health():
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def trigger(secret: bytes, repo: str, pr: int) -> str:
    body = json.dumps({"repo": repo, "pr_number": pr}).encode()
    sig = "sha256=" + hmac.new(secret, body, hashlib.sha256).hexdigest()
    req = urllib.request.Request(
        f"{BASE_URL}/webhook", body,
        {"Content-Type": "application/json", "X-Hub-Signature-256": sig},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.read().decode()
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code}: {exc.read().decode()[:120]}"


def predict(snippet: str) -> tuple[str, set[str]]:
    """What the engines say about this text — the expectation, computed not declared.

    Both engines are consulted because either may handle the file: the deployed
    service uses AST analysis for .py files whose full source parses and the
    regex engine otherwise, and the two name their rules differently
    (`hardcoded_secret` vs `password_assignment`). The returned rule set is the
    union of what either could report, and the severity is the LOWER of the two
    so the assertion holds whichever engine ends up doing the work.

    Semantic analysis is absent here (no corpus locally), so the deployed
    service may also report a higher severity than this.
    """
    regex_result = RiskEngine().assess(code=snippet)
    severity = regex_result["severity"]
    regex_severity = severity.value if hasattr(severity, "value") else str(severity)
    rules = {f.rule for f in regex_result["security"]["findings"]}

    severities = [regex_severity]
    ast_findings = ASTSecurityService().analyze_source(snippet, "snippet.py")
    if ast_findings is not None:
        rules |= {f.rule for f in ast_findings}
        if ast_findings:
            severities.append(
                max((f.severity.value for f in ast_findings),
                    key=SEVERITY_ORDER.index)
            )

    return min(severities, key=SEVERITY_ORDER.index), rules


def require_length(path: pathlib.Path, workdir: pathlib.Path) -> None:
    """Refuse a file too short for the insertion point to prove anything.

    Planting at line 2 of a 2-line file passes whether the hunk arithmetic is
    right or wrong. Failing loudly beats reporting a pass the test didn't earn.
    """
    length = len(path.read_text(encoding="utf-8").split("\n"))
    if length < MIN_FILE_LINES:
        raise SystemExit(
            f"{path.relative_to(workdir).as_posix()} has {length} lines; "
            f"need >= {MIN_FILE_LINES} for a meaningful mid-file edit. "
            "Pick a longer file with --file, or commit one to main."
        )


def pick_file(workdir: pathlib.Path) -> pathlib.Path:
    """The longest tracked text file — most room for a mid-file edit.

    Length rather than a filename or extension is what keeps this repo-agnostic:
    every repo has a longest file, and the engines are regex over text, so no
    language is privileged.
    """
    best, best_len = None, -1
    for rel in run(["git", "ls-files"], cwd=workdir).splitlines():
        path = workdir / rel
        try:
            length = len(path.read_text(encoding="utf-8").split("\n"))
        except (OSError, UnicodeDecodeError):
            continue  # binary or unreadable
        if length > best_len:
            best, best_len = path, length
    if best is None:
        raise SystemExit(f"no readable tracked files in {workdir}")
    require_length(best, workdir)
    return best


def _indent_for(lines: list[str], at: int) -> str:
    """Indentation that keeps an insertion at `at` syntactically plausible."""
    probe = at
    while probe > 0 and not lines[probe - 1].strip():
        probe -= 1
    previous = lines[probe - 1] if probe > 0 else ""
    indent = previous[: len(previous) - len(previous.lstrip())]
    # A line ending in ':' opens a block, so the next line must go one level
    # deeper. Copying the header's own indent produces an IndentationError.
    if previous.rstrip().endswith(":"):
        indent += "    "
    return indent


def plant(path: pathlib.Path, snippet: str) -> tuple[int, int]:
    """Insert snippet near the middle of the file. Returns (first, last), 1-based.

    For Python targets the result is verified to parse, trying insertion points
    outward from the midpoint until one works. This matters more than it looks:
    an unparseable file silently disqualifies itself from AST analysis, so a
    careless planter would quietly test only the regex engine and report a pass.
    """
    original = path.read_text(encoding="utf-8").split("\n")
    is_python = path.suffix.lower() == ".py"
    midpoint = max(1, len(original) // 2)

    # Walk outward from the midpoint: midpoint, +1, -1, +2, -2, ...
    offsets = [0]
    for step in range(1, len(original)):
        offsets.extend([step, -step])

    for offset in offsets:
        at = midpoint + offset
        if not 1 <= at <= len(original):
            continue
        candidate = list(original)
        payload_lines = [
            (_indent_for(original, at) + line) if line.strip() else line
            for line in snippet.split("\n")
        ]
        candidate[at:at] = payload_lines
        text = "\n".join(candidate)

        if is_python:
            try:
                ast.parse(text)
            except SyntaxError:
                continue  # this spot breaks the file; try the next

        path.write_text(text, encoding="utf-8")
        return at + 1, at + len(payload_lines)

    raise SystemExit(
        f"could not plant into {path.name} without breaking its syntax; "
        "try --file with a different target"
    )


def resolve_payload(args) -> str:
    if args.payload_text:
        return args.payload_text
    if args.payload_file:
        return pathlib.Path(args.payload_file).read_text(encoding="utf-8").rstrip("\n")
    return PAYLOADS[args.payload]()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-start", action="store_true")
    ap.add_argument("--down", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--pr", type=int)
    ap.add_argument("--repo", default="notjwp/test-repo")
    ap.add_argument("--workdir", default="D:/test-repo")
    ap.add_argument("--file")
    ap.add_argument("--payload", choices=sorted(PAYLOADS), default="secrets")
    ap.add_argument("--payload-text")
    ap.add_argument("--payload-file")
    ap.add_argument("--list-payloads", action="store_true")
    args = ap.parse_args()

    if args.list_payloads:
        for key, make in sorted(PAYLOADS.items()):
            text = make()
            severity, rules = predict(text)
            print(f"{key:<9} -> {severity:<9} {sorted(rules) or '(no security findings)'}")
            for line in text.split("\n"):
                print(f"            {line}")
        return 0

    owner, name = args.repo.split("/", 1)
    workdir = pathlib.Path(args.workdir)

    settings = get_settings()
    if not settings.GITHUB_WEBHOOK_SECRET:
        raise SystemExit("GITHUB_WEBHOOK_SECRET is not set in .env")
    secret = settings.GITHUB_WEBHOOK_SECRET.encode()

    gh = GitHubClient(
        app_id=settings.GITHUB_APP_ID,
        installation_id=settings.GITHUB_INSTALLATION_ID,
        private_key=settings.GITHUB_PRIVATE_KEY,
        api_base_url=settings.GITHUB_API_BASE_URL,
    )
    token = gh._get_installation_token()
    if not token:
        raise SystemExit("could not authenticate as the GitHub App; check .env")
    api = settings.GITHUB_API_BASE_URL.rstrip("/")
    headers = {**gh._token_headers(token), "Content-Type": "application/json"}

    # 1 ─ stack ---------------------------------------------------------------
    if not args.no_start:
        say(1, "starting the stack (docker compose up --build -d)")
        run(["docker", "compose", "up", "--build", "-d"], cwd=ROOT)
    if not wait_for(health, timeout=120):
        raise SystemExit("the app never became healthy — check: docker compose logs sentinel")
    say(1, "healthy at " + BASE_URL)

    branch = None
    pr_number = args.pr
    expected_lines: list[int] = []
    expected_severity, expected_rules = None, set()
    rel_path = None

    # 2 ─ subject under test --------------------------------------------------
    if pr_number is None:
        snippet = resolve_payload(args)
        expected_severity, expected_rules = predict(snippet)

        if not workdir.exists():
            say(2, f"cloning {args.repo} into {workdir}")
            run(["git", "clone", f"https://github.com/{args.repo}.git", str(workdir)])
        branch = "e2e/" + time.strftime("%Y%m%d-%H%M%S")
        run(["git", "checkout", "-q", "main"], cwd=workdir)
        run(["git", "pull", "-q", "--ff-only"], cwd=workdir)
        run(["git", "checkout", "-qb", branch], cwd=workdir)
        say(2, f"branched {branch} off main")

        if args.file:
            target = workdir / args.file
            if not target.exists():
                raise SystemExit(f"{target} does not exist in {workdir}")
            require_length(target, workdir)
        else:
            target = pick_file(workdir)

        first, last = plant(target, snippet)
        expected_lines = list(range(first, last + 1))
        rel_path = target.relative_to(workdir).as_posix()
        say(2, f"planted at {rel_path}:{first}-{last} (mid-file)")
        say(2, f"local engines predict: {expected_severity}  {sorted(expected_rules) or '(none)'}")

        run(["git", "commit", "-qam", "e2e: planted defect mid-file"], cwd=workdir)
        run(["git", "push", "-q", "-u", "origin", branch], cwd=workdir)

        created = gh._http_json("POST", f"{api}/repos/{owner}/{name}/pulls", headers=headers, data={
            "title": f"e2e: automated review test ({branch})",
            "head": branch, "base": "main",
            "body": f"Opened by e2e.py. Text planted at {rel_path}:{first}-{last} "
                    "to exercise annotation line mapping.",
        })
        pr_number = (created or {}).get("number")
        if not pr_number:
            raise SystemExit("could not open the PR")
        say(2, f"opened PR #{pr_number}")
    else:
        say(2, f"reviewing existing {args.repo}#{pr_number} (report only — nothing to assert against)")

    # 3 ─ review --------------------------------------------------------------
    say(3, "triggering")
    print("      " + trigger(secret, args.repo, pr_number))

    head_sha = gh.get_pull_request_refs(owner, name, pr_number).get("head_sha")

    def sentinel_comment():
        for c in gh.list_issue_comments(owner, name, pr_number):
            if gh.SENTINEL_COMMENT_MARKER in (c.get("body") or ""):
                return c
        return None

    say(3, "waiting for the worker...")
    comment = wait_for(sentinel_comment, timeout=120)

    # 4 ─ results -------------------------------------------------------------
    print()
    failures: list[str] = []

    if comment is None:
        say(4, "NO REVIEW POSTED — recent worker log:")
        print(run(["docker", "compose", "logs", "--tail", "25", "sentinel"], cwd=ROOT, check=False))
        print("\nFAIL")
        return 1

    body = comment.get("body") or ""
    risk = next((ln.split(":", 1)[1].strip() for ln in body.split("\n")
                 if ln.startswith("## Risk Score:")), "?")
    say(4, f"risk            ->  {risk}")

    if expected_severity is not None:
        # Deployed may exceed the local prediction (it also runs semantic
        # analysis against a real corpus, which cannot run here), but must
        # never fall below it.
        try:
            if SEVERITY_ORDER.index(risk) < SEVERITY_ORDER.index(expected_severity):
                failures.append(f"risk {risk} is below the predicted {expected_severity}")
        except ValueError:
            failures.append(f"unrecognized risk value {risk!r}")

    def sentinel_check_runs():
        runs = gh._http_json("GET", f"{api}/repos/{owner}/{name}/commits/{head_sha}/check-runs",
                             headers=gh._token_headers(token)) or {}
        return [r for r in runs.get("check_runs", [])
                if r.get("name") == GitHubClient.CHECK_RUN_NAME] or None

    # The worker posts the comment first and the check run immediately after, so
    # the comment appearing does not mean the run has. Poll rather than assume —
    # otherwise this races the second POST and reports a failure Sentinel didn't
    # commit.
    ours = wait_for(sentinel_check_runs, timeout=60, interval=1.5) or []
    if not ours:
        failures.append("no Sentinel check run found (waited 60s after the comment)")
    for r in ours:
        say(4, f"check run       ->  {r.get('conclusion')}  ({(r.get('output') or {}).get('title')})")
        anns = gh._http_json_list(
            "GET", f"{api}/repos/{owner}/{name}/check-runs/{r['id']}/annotations",
            headers=gh._token_headers(token)) or []
        hit_lines, hit_rules = set(), set()
        for a in anns:
            say(4, f"  annotation    ->  {a['path']}:{a['start_line']}  "
                   f"[{a['annotation_level']}]  {a.get('title')}")
            hit_lines.add(a["start_line"])
            hit_rules.add(a.get("title"))
        if expected_lines:
            on_target = {ln for ln in hit_lines if ln in expected_lines}
            if hit_rules and not on_target:
                failures.append(
                    f"annotations on {sorted(hit_lines)}, expected within {expected_lines} "
                    "— hunk arithmetic may be wrong"
                )
            # Either engine may have handled the file, so require an overlap
            # rather than containment — demanding every predicted rule would
            # fail simply because the other engine did the work.
            if expected_rules and not (hit_rules & expected_rules):
                failures.append(
                    f"none of the predicted rules fired; predicted {sorted(expected_rules)}, "
                    f"got {sorted(hit_rules)}"
                )

    enriched = "Potential security issue detected" not in body
    say(4, f"LLM enrichment  ->  {'real explanations' if enriched else 'FALLBACK (check the API key)'}")
    print(f"\n      https://github.com/{args.repo}/pull/{pr_number}\n")

    # 5 ─ cleanup -------------------------------------------------------------
    if branch and not args.keep:
        say(5, f"closing PR #{pr_number} and deleting {branch}")
        gh._http_json("PATCH", f"{api}/repos/{owner}/{name}/pulls/{pr_number}",
                      headers=headers, data={"state": "closed"})
        run(["git", "checkout", "-q", "main"], cwd=workdir)
        run(["git", "push", "-q", "origin", "--delete", branch], cwd=workdir, check=False)
        run(["git", "branch", "-qD", branch], cwd=workdir, check=False)
    elif branch:
        # Return the local clone to main even when keeping the PR: leaving it on
        # the planted branch means the NEXT run plants on top of this one's edit,
        # which is how a previous session ended up with an unparseable fixture.
        run(["git", "checkout", "-q", "main"], cwd=workdir, check=False)
        say(5, f"left open: PR #{pr_number}, branch {branch} (local clone back on main)")

    if args.down:
        say(5, "stopping the stack")
        run(["docker", "compose", "down"], cwd=ROOT, check=False)

    if failures:
        print("\nFAIL")
        for f in failures:
            print("  - " + f)
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
