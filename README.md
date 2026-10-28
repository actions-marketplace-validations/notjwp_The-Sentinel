
# The Sentinel

[![GitHub Marketplace](https://img.shields.io/badge/Marketplace-Sentinel%20AI%20Code%20Review-purple?logo=github)](https://github.com/marketplace/actions/sentinel-ai-code-review)
[![CI](https://github.com/notjwp/The-Sentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/notjwp/The-Sentinel/actions/workflows/ci.yml)

AI code review for pull requests: security findings with OWASP classification, technical
debt, documentation gaps and duplicate-code detection — posted as a PR comment and a native
check run with inline annotations.

**The quickest way to use it is [as a GitHub Action](#use-it-as-a-github-action)** — four
lines in a workflow, no server to host and no account to create. The same engines also run
as an event-driven FastAPI service with a durable Redis-backed job queue, Prometheus-format
metrics and optional report translation; that path is for running it yourself as an
always-on GitHub App.

A warm Action run takes roughly 30 seconds, nearly all of it network I/O — the analysis
engines themselves account for well under a tenth of a second.

## Architecture

Clean Architecture with strict layer boundaries. The dependency rule (the domain imports
nothing from FastAPI, sklearn, or infrastructure) is **enforced by tests** in
`sentinel/tests/hardening/test_architecture_extended.py`.

```
API (FastAPI) → Application (Orchestration) → Domain (Pure Logic) ← Infrastructure (sklearn / LLM / GitHub / Redis)
                         ↕
                   Workers (Async)
```

### Analysis Engines

| Engine | Layer | Purpose |
|---|---|---|
| **Technical Debt** | domain | **Per-function cyclomatic complexity** from the AST — the worst function the PR actually edited, not the sum of every branch in the diff. Falls back to a keyword count for fragments that do not parse |
| **Security** | domain | Regex-based vulnerability detection + OWASP classification: SQL injection, hardcoded secrets (OpenAI/AWS keys, api_key/password assignments), command injection, `eval`/`exec`, `os.system`, `subprocess shell=True` |
| **Semantic** | domain + infra | Duplicate code detection via embedding cosine similarity, compared against a **real corpus fetched from the PR's base branch** (chunked per top-level function/class) |
| **Documentation** | domain | Docs/docstring/comment checks; `.md`/`.txt` files are judged on their **full text** fetched from the PR head (patch text is only the fallback) |
| **LLM Enrichment** (optional) | infra | Explanation + fix suggestions for security findings via any OpenAI-compatible provider |

`RiskEngine` aggregates debt + security + semantic into an overall `SeverityLevel`
(a CRITICAL security finding forces overall CRITICAL; a semantic duplicate forces HIGH).
**Technical debt is capped at MEDIUM** on the way into that score — complexity is measured
over the whole diff, so it tracks size as much as tangle, and a metric should not fail
your build when the review found nothing. The uncapped figure is still in the report.
`AuditOrchestrator` layers on LLM enrichment, documentation findings, markdown report
building, check-run payloads, and optional translation.

### Request Pipeline

`POST /webhook` identifies a pull request and queues a review of it. It never analyzes code posted in the request body — the worker fetches the real diff from GitHub itself. Requests are verified against
`X-Hub-Signature-256` when `GITHUB_WEBHOOK_SECRET` is set, and re-sent deliveries are
deduplicated by `X-GitHub-Delivery` (HTTP 200 `{"status": "duplicate"}`, no re-processing).

**Which deliveries are reviewed.** When an `X-GitHub-Event` header is present, only
`pull_request` events whose action is `opened`, `synchronize`, `reopened`, or
`ready_for_review` start a review. Everything else — `ping`, `issue_comment`, `push`,
`pull_request.closed`/`labeled`/`edited` — is answered HTTP 200 `{"status": "ignored"}`
with no work done, so a PR is not re-reviewed every time someone comments on it. A request
with no `X-GitHub-Event` header is never filtered (local `curl`, manual replays).

1. **Queued (async) mode** — payload has repo/pr_number but **no `code`**. Enqueues a job
   (durable in Redis when `REDIS_URL` is set); the `BackgroundWorker` then fetches the PR's
   real diff from GitHub (`GitHubClient.get_pull_request_data`, paginated beyond 100 files),
   builds the semantic corpus from the base ref, assesses risk, LLM-enriches the findings,
   and posts back **both** a PR comment and a native **check run** — conclusion
   `failure`/`neutral`/`success` by severity, with annotations mapped to the exact file lines
   at the head commit (patch hunk arithmetic). If the PR fetch itself fails, no comment is
   posted and the check run is `neutral` ("Review skipped") — a GitHub outage never reads as
   a clean PR.
A synchronous mode that analyzed a `code` field posted in the body was removed in M9:
it turned a public endpoint into an arbitrary-code-analysis service, and real GitHub
deliveries never used it.

Both paths post comments via `GitHubClient.upsert_comment`, which tags the comment with a
hidden marker and **updates Sentinel's existing review in place** on re-runs instead of
stacking duplicates.

`assess` raises on engine failure; `assess_resilient` swallows failures and returns safe
defaults. All LLM/GitHub/Redis calls are failure-safe — on any error they fall back to static
strings, skip, or retry with exponential backoff, never crashing the request.

### Durability & Observability

- **Redis-backed queue** (opt-in via `REDIS_URL`): at-least-once delivery — jobs survive
  process crashes, an interrupted job is recovered from the processing list on restart, and
  webhook dedup state survives restarts too. Safe to re-run because posting is idempotent.
  Retries are bounded: after `MAX_ATTEMPTS` recoveries a job is parked on
  `sentinel:jobs:dead` rather than resurrected forever.
- **Graceful shutdown**: a job in flight when the process is asked to stop is allowed to
  finish and acknowledge (bounded by `SHUTDOWN_GRACE_SECONDS`) instead of being abandoned.
- **`GET /metrics`**: Prometheus text format, no client library — jobs processed, per-job
  duration, risk severities, webhook modes, LLM success/fallback, GitHub post outcomes,
  Redis errors, live queue depth.
- **Redis outages stay quiet**: one traceback per failing streak, a one-line summary at most
  every 60s, exponential retry backoff (capped at 5s), and a single "reconnected" line.

## Project Structure

```
main.py                                     # Composition root: queue selection, worker lifespan, routers
pyproject.toml                              # Build system, pytest & mutmut config
requirements.txt                            # Runtime dependencies (the production image installs only these)
requirements-dev.txt                        # Dev tooling (pytest/ruff/mutmut/fakeredis; includes runtime set)
Dockerfile / docker-compose.yml             # Non-root production image; compose runs app + AOF-persisted Redis
mutation.Dockerfile                         # Containerized mutation testing
sentinel/
├── config/
│   └── settings.py                         # Feature flags + credentials (read via get_settings())
├── api/
│   ├── health_controller.py                # GET /health
│   ├── metrics_controller.py               # GET /metrics (Prometheus text format)
│   ├── webhook_controller.py               # POST /webhook (signature -> filter -> dedup -> queue)
│   ├── webhook_security.py                 # X-Hub-Signature-256 verification (route dependency)
│   └── delivery_dedup.py                   # In-memory X-GitHub-Delivery dedup (Redis variant in infra)
├── application/
│   ├── audit_orchestrator.py               # Enrichment, doc findings, report + check payloads, translation
│   └── risk_engine.py                      # Multi-engine risk aggregator (assess / assess_resilient)
├── domain/
│   ├── entities/                           # Finding, PullRequest
│   ├── services/                           # debt / security / semantic / document services
│   └── value_objects/                      # SeverityLevel (LOW/MEDIUM/HIGH/CRITICAL)
├── infrastructure/
│   ├── llm/                                # LLMProvider abstraction + OpenAI-compatible client, safe LLMService
│   ├── github/                             # App JWT → installation token → diffs/corpus/comments/check runs
│   ├── redis/                              # RedisJobQueue (reliable queue) + RedisDeliveryDeduper
│   └── semantic/embedding_engine.py        # sklearn HashingVectorizer (128-dim, L2-norm)
├── monitoring/                             # logger + hand-rolled MetricsRegistry (stdlib only)
├── workers/                                # BackgroundWorker loop + async FIFO JobQueue
└── tests/
    ├── fixtures/                           # Sample code files for debt analysis
    ├── hardening/                          # Architecture, concurrency, fault injection, adversarial inputs
    └── snapshots/                          # OpenAPI schema snapshots
```

## Use it as a GitHub Action

Listed on the Marketplace as
**[Sentinel AI Code Review](https://github.com/marketplace/actions/sentinel-ai-code-review)**.
The review runs on GitHub's runners under your repository's own token, so there is no
server to host and no account to create — and your API key, if you supply one, never
leaves your repository.

Add `.github/workflows/sentinel.yml`:

```yaml
name: Sentinel review

on:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]

permissions:
  contents: read         # read the diff, and blobs for the duplicate-code corpus
  pull-requests: write   # create or update the review comment
  checks: write          # create the check run with inline annotations

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: notjwp/The-Sentinel@v1
        with:
          llm-api-key: ${{ secrets.NVIDIA_API_KEY }}   # optional
          fail-on: high
```

That `permissions:` block is required. Without `pull-requests: write` and `checks: write`
the API calls return 403, and Sentinel — being failure-safe — will post nothing rather
than crash.

**The API key is optional.** Without one, every engine still runs and every finding is
still reported, with static recommendations instead of AI-written explanations and fixes.
Any OpenAI-compatible provider works; see `.env.example` for presets.

### Inputs

| Input | Default | What it does |
|---|---|---|
| `github-token` | `${{ github.token }}` | Reads the PR and posts the review |
| `fail-on` | `high` | Lowest severity that fails the job: `critical`\|`high`\|`medium`\|`low`\|`never` |
| `fail-on-error` | `true` | Fail the job when Sentinel cannot run at all |
| `llm-api-key` | — | Enables AI explanations and fix suggestions |
| `llm-base-url` | NVIDIA NIM | Any OpenAI-compatible endpoint |
| `llm-model` | `nvidia/nemotron-3.5-lightning-30b-a3b` | Model at that endpoint |
| `llm-max-calls` | `1` | Per-review LLM call budget |
| `enable-llm` | `true` | Set `false` to skip enrichment even with a key present |
| `enable-checks` | `true` | Post the native check run |
| `enable-doc-review` | `true` | Report documentation findings |
| `python-version` | `3.13` | Passed to `actions/setup-python` |

Outputs: `severity` and `skipped`, for later steps to branch on.

**On the default model.** It reasons before it answers, so a review's LLM call takes
roughly 20–45 seconds and needs a generous token budget — `LLM_TIMEOUT` defaults to 90s
for that reason. Point `llm-base-url`/`llm-model` at a non-reasoning model if you would
rather trade the prose quality for speed. If the model you name has been retired, the
provider returns `410` and the review falls back to static recommendations rather than
failing; the job log says so.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Reviewed; below the `fail-on` threshold |
| 1 | Reviewed; at or above `fail-on` |
| 2 | Could not run — no PR context, no token, or the PR could not be fetched |

Exit 2 is deliberately distinct from a pass. A review that could not read the code must
never report the same result as one that read it and found nothing.

### Pull requests from forks

GitHub makes `GITHUB_TOKEN` **read-only** for fork pull requests and withholds secrets,
whatever the `permissions:` block says. Sentinel still reviews the PR in full, but it
cannot post the comment or the check run.

So the report is **always** written to the workflow run's job summary, which is a plain
file write and works regardless of token scope. On a fork PR that is where the review
lives — one click from the PR's checks list — and the job's pass/fail status still
reflects the verdict.

`pull_request_target` grants a write token and would restore the comment. If you use it,
**never add `actions/checkout` of the PR's head** — Sentinel reads the diff through the
API and needs no checkout, and checking out fork code under a privileged token is the
standard way this goes wrong. Note also that fork PRs get no secrets, so your LLM key is
absent there and the review falls back to static recommendations.

## Setup

Python 3.13+ is required.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1    # Windows
source .venv/bin/activate      # Linux/macOS
pip install -r requirements-dev.txt   # dev install (tests/lint); runtime-only: requirements.txt
```

### Configuration (`.env`)

Settings are read only via `get_settings()`. Optional integrations are inert without their
credentials even when the flag is on.

| Variable | Default | Purpose |
|---|---|---|
| `ENABLE_LLM` | `true` | Enable LLM enrichment (requires a resolved `LLM_API_KEY`) |
| `ENABLE_GITHUB` | `true` | Enable PR comment/check posting (requires GitHub App vars) |
| `ENABLE_CHECKS` | `true` | Post native check runs (app needs **Checks: Read & write**) |
| `ENABLE_DOC_REVIEW` | `true` | Enable documentation findings |
| `ENABLE_TRANSLATION` | `false` | Append translated report sections (needs LLM **and** budget: set `LLM_MAX_CALLS≥3` — 1 enrichment + 1 per language) |
| `LLM_MAX_CALLS` | `1` | Per-request LLM call budget (shared by enrichment + translation) |
| `LLM_TIMEOUT` | `90.0` | LLM request timeout (seconds) |
| `LLM_BASE_URL` | `https://integrate.api.nvidia.com/v1` | LLM endpoint (any OpenAI-compatible provider) |
| `LLM_MODEL` | `nvidia/nemotron-3.5-lightning-30b-a3b` | Model id at that endpoint |
| `LLM_API_KEY` | — | LLM credential; **falls back to `NVIDIA_API_KEY`** when unset |
| `NVIDIA_API_KEY` | — | Back-compat credential / fallback for `LLM_API_KEY` |
| `GITHUB_APP_ID` / `GITHUB_INSTALLATION_ID` / `GITHUB_PRIVATE_KEY` | — | GitHub App auth (RS256) |
| `GITHUB_API_BASE_URL` | `https://api.github.com` | GitHub API base URL |
| `GITHUB_WEBHOOK_SECRET` | — | When set, `POST /webhook` enforces `X-Hub-Signature-256`; set it before public exposure |
| `REDIS_URL` | — | e.g. `redis://localhost:6379/0` — durable queue + dedup; unset = in-memory (lost on restart) |

The LLM provider is **any OpenAI-compatible endpoint** (the `openai` SDK pointed at
`LLM_BASE_URL`). Defaults target NVIDIA NIM; switch to Groq, Gemini, or a local Ollama by
editing `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` — no code change. See
[`.env.example`](.env.example) for ready-to-paste provider presets.

### GitHub App setup

> Running it as an always-on service, or making the App public so others can install it?
> See **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** — free always-on hosting, TLS, the
> two firewalls Oracle gives you, and the per-installation LLM budget.

Create a GitHub App (Settings → Developer settings → GitHub Apps) with **Repository
permissions**: Pull requests (Read & write), Issues (Read & write), Checks (Read & write),
Contents (Read-only); webhook can stay disabled for manual triggering.

One deployment serves **every** installation: the installation id travels with each
delivery, so `GITHUB_INSTALLATION_ID` is only a fallback for manual triggers.

> **Changing permissions on an already-installed App needs re-approval.** GitHub raises a
> pending request on the installation; until you accept it the token keeps its old scopes.
> A missing **Checks: Read & write** is the usual cause of "the comment posted but no check
> run appeared" — `create_check_run` fails safely and only logs `check run=False`.

**Going public.** Set `GITHUB_WEBHOOK_SECRET` (and the same value on the App's webhook),
`METRICS_TOKEN`, and `ENVIRONMENT=production` — the last makes the secret mandatory, so the
app refuses to start rather than serve an unsigned endpoint. Subscribe the App to
**Pull requests only**: Sentinel ignores every other event anyway. Install it on the
target repo, then set `GITHUB_APP_ID`, `GITHUB_INSTALLATION_ID` (from the installation URL),
and `GITHUB_PRIVATE_KEY` (the downloaded PEM, single line with `\n` separators is accepted).

Trigger a review manually — Sentinel only makes *outbound* calls, so no tunnel is needed:

```bash
curl -X POST http://localhost:8000/webhook \
  -H "Content-Type: application/json" \
  -d '{"repo": "owner/name", "pr_number": 1}'
```

## Run

```bash
# Dev
uvicorn main:app --reload

# Production-shaped: app + AOF-persisted Redis, REDIS_URL pre-wired (expects a local .env)
docker compose up --build
```

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Root health message |
| `GET` | `/health` | Health status |
| `GET` | `/metrics` | Prometheus-format metrics (counters, durations, live queue depth). Requires `Authorization: Bearer $METRICS_TOKEN` when that is set; open otherwise |
| `POST` | `/webhook` | Receives PR events and queues a review |

## Test

```bash
# Run all tests (config in pyproject.toml, testpaths = sentinel/tests)
pytest

# Single file / test
pytest sentinel/tests/test_risk_engine.py
pytest sentinel/tests/test_risk_engine.py::test_name -q

# Coverage (branch; CI enforces --cov-fail-under=85)
pytest --cov=sentinel --cov-branch --cov-report=term-missing

# Hardening tests only
pytest sentinel/tests/hardening/ -v

# Lint (enforced in CI)
ruff check sentinel main.py
```

Redis-backed tests need no server — they run against `fakeredis` injected through the
`client=` seam.

## Mutation Testing

Configuration is in `pyproject.toml` under `[tool.mutmut]`
(targets: `sentinel/domain/`, `sentinel/application/`, `sentinel/infrastructure/`).

```bash
# Local
mutmut run && mutmut results

# Docker (recommended)
docker build -f mutation.Dockerfile -t sentinel-mutmut .
docker run --rm sentinel-mutmut

# Docker with persistent results
docker run --rm -v sentinel-mutmut-data:/app sentinel-mutmut
docker run --rm -v sentinel-mutmut-data:/app sentinel-mutmut results
```
