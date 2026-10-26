# Deploying the hosted App

> **This path is not taken.** Sentinel is distributed as a GitHub Action, which
> reaches anyone with a GitHub account and needs no server, so no host was ever
> provisioned. Nothing here is pending work — it is the route to take *if* an
> always-on hosted App is ever wanted, written down while the reasoning was
> fresh. The one thing it has never done is receive a real inbound webhook over
> a network.

This covers running Sentinel as an always-on GitHub App. If you only want to
review pull requests on your own repositories, **use the GitHub Action instead**
(see the README) — it needs none of this.

## Why the host has to stay awake

The background worker runs **in-process**. A platform that suspends the
container between requests drops whatever is mid-review, and with the
in-memory queue it drops everything queued as well. That rules out the usual
free tiers:

| Platform | Verdict |
|---|---|
| Oracle Cloud Always Free | Works. The only genuinely always-on free compute. |
| Render free tier | Sleeps after ~15 min idle, ~1 min cold start. Reviews arrive late. |
| Koyeb free tier | Reported closed to new accounts. |
| Cloud Run / Lambda | Scale-to-zero by default; paying for a warm instance is not free. |

Before committing to Oracle, know what you are signing up for: the Always Free
ARM allowance was **cut from 4 OCPU / 24 GB to 2 OCPU / 12 GB in June 2026**, a
credit card is required for identity verification even though the tier costs
nothing, ARM (Ampere A1) capacity is frequently unavailable in popular regions,
and **idle instances can be reclaimed**. Verify the current terms at signup
rather than trusting this page. For an academic demo this is fine; it is not a
production guarantee.

## 1. Provision

Create an **Ampere A1 (ARM)** VM, Ubuntu LTS, Always Free shape. The image is
built from `python:3.13-slim` and every dependency ships an `aarch64` wheel, so
ARM needs no changes.

If ARM capacity is unavailable, the AMD **Micro** shape is the fallback — but at
1 GB RAM it is genuinely tight for scikit-learn plus the Python runtime. Add
swap and treat it as degraded, not equivalent.

Install Docker and the Compose plugin, then clone the repository.

## 2. Open the firewall — both of them

Oracle images ship **restrictive iptables rules inside the VM** on top of the
VCN security list. Opening 80/443 in the cloud console alone is not enough, and
this is the single most common reason a freshly provisioned Oracle box looks
unreachable.

```bash
# 1. VCN security list (cloud console): ingress 80 and 443 from 0.0.0.0/0
# 2. Inside the VM:
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
```

## 3. TLS

GitHub verifies the webhook endpoint's certificate, so deliveries to a
self-signed or plain-HTTP endpoint fail before reaching the app. `deploy/Caddyfile`
gets a Let's Encrypt certificate automatically — point a hostname at the VM's
public IP first (a free DuckDNS subdomain is fine) and set it in the Caddyfile
along with your email.

## 4. Configure

```bash
cp .env.example .env
```

Set at minimum:

```ini
ENVIRONMENT=production
GITHUB_WEBHOOK_SECRET=<python -c "import secrets; print(secrets.token_hex(32))">
METRICS_TOKEN=<another random token>
REDIS_URL=redis://redis:6379/0
GITHUB_APP_ID=<from the App settings page>
GITHUB_PRIVATE_KEY=<the PEM, newlines escaped as \n>
LLM_API_KEY=<optional>
```

`ENVIRONMENT=production` makes the first four effectively mandatory:
`verify_startup_config` **refuses to boot** without a webhook secret (the
endpoint would accept forged deliveries) or without the app credentials
(reviews would run and silently never post). Both are deliberate — a service
that starts and does nothing is worse than one that does not start.

`GITHUB_INSTALLATION_ID` is **not** needed. Since multi-tenancy each job carries
its own installation; the setting is only a fallback for manual triggers.

**Redis is not optional here.** Without it every restart drops the queue, and
the in-memory deduper is per-process, so a redelivered webhook gets reviewed
twice.

## 5. Run

```bash
docker compose -f docker-compose.yml -f deploy/docker-compose.prod.yml up -d --build
```

The overlay stops publishing port 8000 on the host (so nothing can bypass TLS),
sets memory limits, and adds Caddy. It uses the `!override` tag for the port
list, which needs Compose v2.24 or newer — check with `docker compose version`.

`restart: unless-stopped` plus Docker's own service being enabled is what brings
the stack back after a reboot. The app waits for Redis to answer rather than
merely to exist, because on a reboot both start together and `recover_pending()`
against a Redis still loading its AOF would find an empty queue.

## 6. Point the App at it

In the GitHub App settings: webhook URL `https://<your-host>/webhook`, the same
secret as `.env`, subscribed to **Pull requests** only.

## 7. Verify

Each of these fails loudly if the step before it was wrong.

```bash
# TLS and liveness
curl https://<host>/health                       # {"status":"ok"}

# The metrics gate actually gates
curl -i https://<host>/metrics                   # 404 (blocked at Caddy)
curl -H "Authorization: Bearer $METRICS_TOKEN" http://localhost:8000/metrics   # on the VM
```

- Open a pull request on a repository with the App installed, and confirm a
  review comment and a `Sentinel AI Code Review` check run appear.
- Check the App's **Recent Deliveries** tab for a `pull_request` delivery with a
  200 response. This is the first time an inbound webhook has crossed a network
  for this project — everything before it was replayed locally.
- Reboot the VM. Confirm the stack returns unattended.
- Deliberately break it: comment out `GITHUB_WEBHOOK_SECRET` and restart. The
  container must refuse to start. If it boots, `ENVIRONMENT` is not set to
  `production` and the endpoint is accepting unsigned requests.

## 8. Going public

App settings → Advanced → **Make this GitHub App public**. `github.com/apps/<name>`
then becomes an install page anyone can use. A Marketplace listing is separate
and not required for public installs.

Two things a stranger judges you on at the install screen:

- **The permission set.** Pull requests R/W, Contents R, Checks R/W, Metadata R
  — keep it exactly there. "Contents: Read" reads as *this app can read all my
  private code*, because it can, and every extra scope visibly costs installs.
- **A privacy policy URL** stating plainly that code snippets are sent to a
  third-party LLM provider, and what that provider does with them. This is the
  main reason someone declines, and omitting it does not make the question go
  away.

### Cap what one key can cost

Public means one API key serving everyone, under a provider free tier of
roughly 40 requests/minute. Set a per-installation daily budget:

```ini
LLM_DAILY_QUOTA=20     # enriched reviews per installation per day
REDIS_URL=redis://redis:6379/0
```

Past the cap, reviews **still run and still post** — with the static
recommendations the no-key path already produces. Only the AI-written prose is
withheld, and `sentinel_llm_quota_exhausted_total` counts how often. The
counter needs Redis because it has to survive restarts; with `LLM_DAILY_QUOTA=0`
(the default) or no Redis, nothing is metered.

The budget fails **open**: if Redis is unreachable the review keeps its
enrichment, on the grounds that a broken counter should not quietly downgrade
every review on the platform.

What this does not cap is compute. One installation on a repository where bots
open hundreds of pull requests will still occupy the single worker. Per-tenant
concurrency is not built.

## Known limits of this deployment

- **One worker processes one job at a time.** A busy installation delays every
  other tenant's reviews. Splitting the web process from N worker processes is
  the fix, and the code does not support it yet.
- **A hung worker strands its job** until restart. Crashes are covered
  (`recover_pending`, dead-letter after 3 attempts); hangs need a visibility
  timeout, which is not built.
- **Oracle can reclaim idle instances.** There is no SLA on free compute.
