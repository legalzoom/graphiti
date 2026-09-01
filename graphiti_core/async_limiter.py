"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

DEFAULT_MAX_WAITERS_PER_SLOT = 16


class AsyncCapacityOverloadedError(Exception):
    """Raised immediately when a capacity limiter's bounded wait queue is full."""

    retry_after_seconds = 1

    def __init__(self, capacity: int, max_waiters: int) -> None:
        self.capacity = capacity
        self.max_waiters = max_waiters
        super().__init__(f'capacity is exhausted and all {max_waiters} waiting slots are occupied')


class AsyncCapacityLease:
    """One idempotently released slot from an :class:`AsyncCapacityLimiter`.

    A lease can be retained by a thread-pool future. This matters when its asyncio task is
    cancelled: the coroutine can stop immediately without returning capacity while synchronous
    work that cannot be cancelled is still running.
    """

    def __init__(self, limiter: AsyncCapacityLimiter) -> None:
        self._limiter = limiter
        self._lock = threading.Lock()
        self._pending_holds = 0
        self._release_requested = False
        self._released = False

    def hold_until_complete(self, future: Future[Any]) -> None:
        """Defer a requested release until ``future`` finishes, independent of its event loop."""
        with self._lock:
            if self._release_requested:
                raise RuntimeError('cannot retain a lease after release was requested')
            self._pending_holds += 1

        try:
            future.add_done_callback(self._finish_hold)
        except BaseException:
            self._finish_hold(future)
            raise

    def _finish_hold(self, _future: Future[Any]) -> None:
        release_capacity = False
        with self._lock:
            if self._pending_holds < 1:
                raise RuntimeError('capacity lease hold completed too many times')
            self._pending_holds -= 1
            if self._release_requested and self._pending_holds == 0 and not self._released:
                self._released = True
                release_capacity = True

        if release_capacity:
            self._limiter.release()

    def release(self) -> None:
        release_capacity = False
        with self._lock:
            if self._release_requested:
                return
            self._release_requested = True
            if self._pending_holds == 0:
                self._released = True
                release_capacity = True

        if release_capacity:
            self._limiter.release()


_CURRENT_CAPACITY_LEASE: ContextVar[AsyncCapacityLease | None] = ContextVar(
    'graphiti_current_capacity_lease', default=None
)


@contextmanager
def capacity_lease_context(lease: AsyncCapacityLease) -> Iterator[None]:
    """Make ``lease`` available to synchronous executor boundaries in this task context."""
    token = _CURRENT_CAPACITY_LEASE.set(lease)
    try:
        yield
    finally:
        _CURRENT_CAPACITY_LEASE.reset(token)


def current_capacity_lease() -> AsyncCapacityLease | None:
    """Return the capacity lease bound to the current context, if any."""
    return _CURRENT_CAPACITY_LEASE.get()


class AsyncCapacityLimiter:
    """Share bounded capacity across event loops without blocking an executor.

    Queued waiters use thread-safe futures so a single limiter can serve multiple event loops.
    Cancellation removes a waiter from the queue immediately; no abandoned blocking acquisition
    remains to delay later work.
    """

    def __init__(self, capacity: int, *, max_waiters: int | None = None) -> None:
        if capacity < 1:
            raise ValueError('capacity must be at least 1')
        if max_waiters is None:
            max_waiters = capacity * DEFAULT_MAX_WAITERS_PER_SLOT
        if max_waiters < 0:
            raise ValueError('max_waiters cannot be negative')
        self._capacity = capacity
        self._available = capacity
        self._max_waiters = max_waiters
        self._waiters: OrderedDict[Future[None], None] = OrderedDict()
        self._lock = threading.Lock()

    async def acquire(self) -> AsyncCapacityLease:
        waiter: Future[None] | None = None
        with self._lock:
            if self._available > 0:
                self._available -= 1
            elif len(self._waiters) >= self._max_waiters:
                raise AsyncCapacityOverloadedError(self._capacity, self._max_waiters)
            else:
                waiter = Future()
                self._waiters[waiter] = None

        if waiter is None:
            return AsyncCapacityLease(self)

        try:
            await asyncio.wrap_future(waiter)
        except asyncio.CancelledError:
            grant_received = False
            with self._lock:
                if waiter.cancel():
                    self._waiters.pop(waiter, None)
                else:
                    grant_received = True
            if grant_received:
                self.release()
            raise

        return AsyncCapacityLease(self)

    def release(self) -> None:
        with self._lock:
            while self._waiters:
                waiter, _ = self._waiters.popitem(last=False)
                if waiter.set_running_or_notify_cancel():
                    waiter.set_result(None)
                    return

            if self._available >= self._capacity:
                raise ValueError('capacity released too many times')
            self._available += 1
