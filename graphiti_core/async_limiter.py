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
from concurrent.futures import Future


class AsyncCapacityLease:
    """One idempotently released slot from an :class:`AsyncCapacityLimiter`."""

    def __init__(self, limiter: AsyncCapacityLimiter) -> None:
        self._limiter = limiter
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._limiter.release()


class AsyncCapacityLimiter:
    """Share bounded capacity across event loops without blocking an executor.

    Queued waiters use thread-safe futures so a single limiter can serve multiple event loops.
    Cancellation removes a waiter from the queue immediately; no abandoned blocking acquisition
    remains to delay later work.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError('capacity must be at least 1')
        self._capacity = capacity
        self._available = capacity
        self._waiters: OrderedDict[Future[None], None] = OrderedDict()
        self._lock = threading.Lock()

    async def acquire(self) -> AsyncCapacityLease:
        waiter: Future[None] | None = None
        with self._lock:
            if self._available > 0:
                self._available -= 1
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
