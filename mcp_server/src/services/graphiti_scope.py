"""Call-scoped Graphiti clients for group-routed database operations."""

import asyncio
from copy import copy
from dataclasses import dataclass, field
from functools import partial

from graphiti_core import Graphiti
from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.helpers import validate_group_id
from graphiti_core.namespaces import EdgeNamespace, NodeNamespace

_DRIVER_CACHE_ATTRIBUTE = '_graphiti_mcp_group_driver_cache'


@dataclass
class _GroupDriverCache:
    """Single-flight Falkor drivers owned by one base Graphiti client."""

    base_driver: GraphDriver
    initialization_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    tasks: dict[str, asyncio.Task[GraphDriver]] = field(default_factory=dict)


async def _initialize_group_driver(
    base_driver: GraphDriver,
    group_id: str,
    initialization_lock: asyncio.Lock,
) -> GraphDriver:
    # Falkor clones share one underlying async connection. Serialize first-time
    # index builds across groups so they cannot collide on that connection.
    async with initialization_lock:
        scoped_driver = base_driver.clone(database=group_id)
        init_task = getattr(scoped_driver, '_init_task', None)
        if init_task is not None:
            await asyncio.shield(init_task)
        return scoped_driver


def _evict_failed_group_driver(
    cache: _GroupDriverCache,
    group_id: str,
    task: asyncio.Task[GraphDriver],
) -> None:
    """Observe initialization failures and leave the cache retryable."""
    failed = task.cancelled()
    if not failed:
        failed = task.exception() is not None
    if failed and cache.tasks.get(group_id) is task:
        cache.tasks.pop(group_id, None)


async def driver_for_group(client: Graphiti, group_id: str) -> GraphDriver:
    """Return one initialized, cached Falkor driver for ``group_id``.

    Falkor ``clone`` constructs a driver and schedules index creation. Cache that
    initialization task on the owning Graphiti client so concurrent requests share
    one clone and a cancelled waiter cannot cancel initialization for other callers.
    Other providers keep logical groups in one database and need no driver clone.
    """
    validate_group_id(group_id)
    base_driver = client.driver
    if base_driver.provider != GraphProvider.FALKORDB:
        return base_driver
    if group_id == getattr(base_driver, 'default_group_id', None):
        # Falkor's logical default group ('_') lives in the database selected on
        # the base driver, which may be a configured name other than default_db.
        init_task = getattr(base_driver, '_init_task', None)
        if init_task is not None:
            await asyncio.shield(init_task)
        return base_driver

    cache = getattr(client, _DRIVER_CACHE_ATTRIBUTE, None)
    if not isinstance(cache, _GroupDriverCache) or cache.base_driver is not base_driver:
        cache = _GroupDriverCache(base_driver=base_driver)
        setattr(client, _DRIVER_CACHE_ATTRIBUTE, cache)

    task = cache.tasks.get(group_id)
    if task is None:
        task = asyncio.create_task(
            _initialize_group_driver(base_driver, group_id, cache.initialization_lock)
        )
        cache.tasks[group_id] = task
        task.add_done_callback(partial(_evict_failed_group_driver, cache, group_id))

    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        if task.cancelled() and cache.tasks.get(group_id) is task:
            cache.tasks.pop(group_id, None)
        raise
    except Exception:
        if cache.tasks.get(group_id) is task:
            cache.tasks.pop(group_id, None)
        raise


async def graphiti_for_group(client: Graphiti, group_id: str) -> Graphiti:
    """Shallow-copy Graphiti with every driver-bound surface routed to one group.

    FalkorDB stores groups in separate graph databases. A shared Graphiti instance
    must therefore never be rebound for an individual request or queue worker:
    concurrent groups would race on ``client.driver``. Neo4j and Neptune return
    logical groups share the base driver, so this remains a cheap call-scoped copy.
    """
    scoped_driver = await driver_for_group(client, group_id)
    scoped_client = copy(client)
    scoped_client.driver = scoped_driver
    scoped_client.clients = client.clients.model_copy(update={'driver': scoped_driver})
    scoped_client.nodes = NodeNamespace(scoped_driver, scoped_client.embedder)
    scoped_client.edges = EdgeNamespace(scoped_driver, scoped_client.embedder)
    return scoped_client
