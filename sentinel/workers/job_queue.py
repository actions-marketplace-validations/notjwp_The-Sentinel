import asyncio

from sentinel.monitoring.logger import get_logger

logger = get_logger(__name__)


class JobQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict] = asyncio.Queue()

    async def enqueue(self, job: dict) -> None:
        if not isinstance(job, dict):
            raise TypeError("job must be a dictionary")
        await self._queue.put(job)
        logger.info("Job enqueued")

    async def dequeue(self) -> dict:
        return await self._queue.get()

    async def ack(self, job: dict) -> None:
        """No-op: in-memory jobs are gone once dequeued (at-most-once by nature).

        Exists so callers use one interface for this queue and the durable
        RedisJobQueue (which really does ack) without hasattr branching.
        """

    async def reclaim_expired(self) -> int:
        """No-op: nothing is leased, because nothing survives this process.

        Present for the same reason as ``ack`` and ``recover_pending`` — callers
        use one interface for both queues without ``hasattr`` branching.
        """
        return 0

    async def recover_pending(self) -> int:
        """No-op counterpart to RedisJobQueue.recover_pending. Nothing to recover."""
        return 0

    async def depth(self) -> int:
        """Jobs currently waiting (the /metrics queue-depth gauge)."""
        return self._queue.qsize()
