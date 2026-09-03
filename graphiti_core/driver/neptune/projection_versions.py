"""Durable Neptune generations for OpenSearch vector projections."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Collection, Coroutine
from functools import wraps
from typing import Any, ParamSpec, Protocol, TypeVar

from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.helpers import (
    NEPTUNE_INTERNAL_PROPERTY_PREFIX,
    get_neptune_projection_versions,
)

logger = logging.getLogger(__name__)

_PARAMS = ParamSpec('_PARAMS')
_RESULT = TypeVar('_RESULT')


class QueryRunner(Protocol):
    async def run(self, query: str, **kwargs: Any) -> Any: ...


def validate_projection_operation_interface(driver: object) -> None:
    """Fail closed when legacy custom operations cannot honor vector lifecycle invariants."""
    if (
        getattr(driver, 'vector_projection_enabled', False)
        and getattr(driver, 'graph_operations_interface', None) is not None
    ):
        raise RuntimeError(
            'Neptune vector projection is incompatible with the legacy '
            'graph_operations_interface because custom handlers cannot participate in the '
            'projection-generation lifecycle. Remove the custom interface or keep '
            'vector_projection_enabled=false until it is migrated.'
        )


def validate_projection_attributes(
    attributes: dict[str, object] | None,
    canonical_properties: Collection[str] = (),
) -> None:
    """Reject caller-controlled properties that can corrupt projection identity or state."""
    canonical = frozenset(canonical_properties)
    reserved = sorted(
        key
        for key in (attributes or {})
        if key.startswith(NEPTUNE_INTERNAL_PROPERTY_PREFIX) or key in canonical
    )
    if reserved:
        raise ValueError(
            'Entity attributes may not overwrite Neptune canonical or projection-reserved '
            'properties: ' + ', '.join(reserved)
        )


def validate_batch_size(batch_size: int) -> None:
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError(f'batch_size must be a positive integer; got {batch_size!r}')


def defer_cancellation_until_complete(
    operation: Callable[_PARAMS, Coroutine[object, object, _RESULT]],
) -> Callable[_PARAMS, Awaitable[_RESULT]]:
    """Finish a graph/projection consistency boundary before delivering cancellation.

    Neptune and OpenSearch cannot participate in one transaction. Once a lifecycle operation
    starts, cancellation must not strand only one side of the materialized view. The inner task is
    therefore shielded to completion; if it succeeds after cancellation was requested, the caller
    still receives ``CancelledError``.
    """

    @wraps(operation)
    async def wrapped(*args: _PARAMS.args, **kwargs: _PARAMS.kwargs) -> _RESULT:
        task = asyncio.create_task(operation(*args, **kwargs))
        cancellation_requested = False
        while True:
            try:
                result = await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                cancellation_requested = True
                if task.cancelled():
                    # The inner operation itself was cancelled rather than merely shielded from
                    # cancellation of this waiter.
                    return await task
            except BaseException as error:
                if not cancellation_requested:
                    raise
                # The consistency boundary failed after its caller was cancelled. Retrieve and
                # report only bounded metadata from that failure, then preserve the caller's
                # cancellation contract so long-running reconcilers and shutdown hooks terminate.
                logger.error(
                    'Consistency-boundary operation failed after caller cancellation '
                    '(operation=%s error_type=%s); delivering cancellation',
                    operation.__qualname__,
                    type(error).__name__,
                )
                raise asyncio.CancelledError from None

        if cancellation_requested:
            raise asyncio.CancelledError
        return result

    return wrapped


async def reserve_projection_versions(
    executor: QueryExecutor,
    projection_kind: str,
    uuids: list[str],
    tx: Transaction | None = None,
    batch_size: int = 100,
) -> dict[str, int]:
    """Atomically reserve one monotonically increasing generation per UUID in Neptune."""
    validate_batch_size(batch_size)
    unique_uuids = list(dict.fromkeys(uuids))
    if not unique_uuids:
        return {}
    query = """
        UNWIND $projections AS requested
        MERGE (projection:GraphitiProjectionVersion {
            projection_id: requested.projection_id
        })
        SET projection._graphiti_projection_lock = true
        REMOVE projection._graphiti_projection_lock
        WITH requested, projection,
            coalesce(projection.generation, 0) + 1 AS projection_version
        SET projection.generation = projection_version
        RETURN requested.uuid AS uuid, projection_version
    """
    versions: dict[str, int] = {}
    for start in range(0, len(unique_uuids), batch_size):
        chunk = unique_uuids[start : start + batch_size]
        projections = [
            {'uuid': uuid, 'projection_id': f'{projection_kind}:{uuid}'} for uuid in chunk
        ]
        if tx is not None:
            result = await tx.run(query, projections=projections)
        else:
            result = await executor.execute_query(query, projections=projections)
        versions.update(get_neptune_projection_versions(result))
    if set(versions) != set(unique_uuids):
        raise RuntimeError(
            f'Neptune reserved {len(versions)}/{len(unique_uuids)} {projection_kind} '
            'projection generations'
        )
    return versions


async def clear_projection_sync_pending(
    executor: QueryExecutor,
    projection_kind: str,
    versions: dict[str, int],
    tx: QueryRunner | None = None,
    batch_size: int = 100,
) -> None:
    """Acknowledge exact graph generations after their vector writes are durable."""
    validate_batch_size(batch_size)
    if not versions:
        return
    if projection_kind == 'node':
        match = 'MATCH (projection:Entity {uuid: completed.uuid})'
    elif projection_kind == 'edge':
        match = 'MATCH ()-[projection:RELATES_TO {uuid: completed.uuid}]->()'
    else:
        raise ValueError(f'Unsupported projection kind {projection_kind!r}')

    query = f"""
        UNWIND $completed AS completed
        {match}
        SET projection._graphiti_projection_lock = true
        REMOVE projection._graphiti_projection_lock
        WITH projection, completed
        WHERE projection._graphiti_projection_version = completed.projection_version
          AND projection._graphiti_vector_sync_pending = completed.projection_version
          AND coalesce(projection._graphiti_vector_delete_pending, false) = false
        REMOVE projection._graphiti_vector_sync_pending
        RETURN projection.uuid AS uuid
    """
    items = list(versions.items())
    for start in range(0, len(items), batch_size):
        completed = [
            {'uuid': uuid, 'projection_version': version}
            for uuid, version in items[start : start + batch_size]
        ]
        if tx is not None:
            await tx.run(query, completed=completed)
        else:
            await executor.execute_query(query, completed=completed)
