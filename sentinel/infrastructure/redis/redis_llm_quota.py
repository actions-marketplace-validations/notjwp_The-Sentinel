"""Per-installation daily budget for LLM-enriched reviews.

With the App public, one API key serves every installation under a provider
rate limit that was never sized for that. This caps what any single
installation can spend in a day.

Exceeding the cap does NOT skip the review: the engines still run and the
findings are still posted, with the static recommendations that the no-key
path already produces and already tests. The only thing lost is the
AI-written explanation, which is the right thing to degrade.

Keys expire rather than being reset on a schedule, so there is no sweeper and
nothing to run at midnight: the day is encoded in the key and yesterday's
simply ages out.

Fails OPEN on Redis errors — a review with enrichment is better than a review
without it, and an outage in the budget counter should not quietly downgrade
every review on the platform.
"""

import datetime
from typing import Any

import redis.asyncio as redis_asyncio

from sentinel.monitoring.logger import get_logger

logger = get_logger(__name__)


class RedisLLMQuota:
    KEY_PREFIX = "sentinel:llmquota:"
    # Two days, so a key written just before midnight UTC is not evicted while
    # still being counted against.
    TTL_SECONDS = 48 * 60 * 60

    def __init__(
        self,
        url: str,
        *,
        daily_quota: int,
        client: Any | None = None,
    ) -> None:
        # ``client`` is a test seam (fakeredis.FakeAsyncRedis); from_url is lazy.
        self._client = client if client is not None else redis_asyncio.from_url(
            url, decode_responses=True
        )
        self._daily_quota = daily_quota

    @staticmethod
    def _today() -> str:
        return datetime.datetime.now(datetime.UTC).strftime("%Y%m%d")

    def _key(self, installation_id: str) -> str:
        return f"{self.KEY_PREFIX}{self._today()}:{installation_id}"

    async def consume(self, installation_id: object) -> bool:
        """Claim one enriched review. True when it may use the LLM.

        The increment happens before the check, so concurrent workers cannot
        both pass on the last unit of budget. A job with no installation id
        (manual trigger, e2e harness) is not metered: it is ours, not a
        tenant's.
        """
        if self._daily_quota <= 0:
            return True
        if not installation_id:
            return True

        key = self._key(str(installation_id))
        try:
            used = await self._client.incr(key)
            # Only the first write needs the expiry; setting it every time
            # would slide the window forward and the key would never age out.
            if used == 1:
                await self._client.expire(key, self.TTL_SECONDS)
        except Exception:
            logger.exception("LLM quota check failed; allowing enrichment for this review")
            return True

        if used > self._daily_quota:
            logger.info(
                "Installation %s has used its daily LLM budget (%s); "
                "this review will post with static recommendations.",
                installation_id,
                self._daily_quota,
            )
            return False
        return True
