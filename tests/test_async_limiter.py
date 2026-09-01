"""Tests for cancellation-aware capacity shared across event loops."""

from __future__ import annotations

import asyncio
import threading

import pytest

from graphiti_core.async_limiter import AsyncCapacityLimiter


@pytest.mark.asyncio
async def test_cancelled_waiter_storm_is_removed_immediately():
    limiter = AsyncCapacityLimiter(1)
    held_lease = await limiter.acquire()
    waiters = [asyncio.create_task(limiter.acquire()) for _ in range(200)]

    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert len(limiter._waiters) == 200

    for waiter in waiters:
        waiter.cancel()
    await asyncio.gather(*waiters, return_exceptions=True)

    assert not limiter._waiters
    held_lease.release()
    replacement = await asyncio.wait_for(limiter.acquire(), timeout=1)
    replacement.release()


def test_capacity_is_shared_across_event_loop_threads():
    limiter = AsyncCapacityLimiter(2)
    acquired = [threading.Event() for _ in range(3)]
    release = [threading.Event() for _ in range(3)]
    errors: list[BaseException] = []

    def run_worker(index: int) -> None:
        async def worker() -> None:
            lease = await limiter.acquire()
            acquired[index].set()
            try:
                await asyncio.to_thread(release[index].wait)
            finally:
                lease.release()

        try:
            asyncio.run(worker())
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=run_worker, args=(0,))
    second = threading.Thread(target=run_worker, args=(1,))
    third = threading.Thread(target=run_worker, args=(2,))
    threads = [first, second, third]
    try:
        first.start()
        second.start()
        assert acquired[0].wait(timeout=1)
        assert acquired[1].wait(timeout=1)

        third.start()
        assert not acquired[2].wait(timeout=0.05)

        release[0].set()
        assert acquired[2].wait(timeout=1)
    finally:
        for event in release:
            event.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=1)

    assert all(not thread.is_alive() for thread in threads)
    assert not errors


def test_capacity_must_be_positive():
    with pytest.raises(ValueError, match='at least 1'):
        AsyncCapacityLimiter(0)
