"""Repair durable Neptune-to-OpenSearch vector projection work.

Neptune and OpenSearch cannot share a transaction. Entity mutations therefore leave an
exact-generation marker in Neptune before touching OpenSearch. This module owns the shared,
idempotent repair sweep used by every long-running service and by the operational CLI.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

from graphiti_core.driver.neptune.projection_versions import (
    clear_projection_sync_pending,
    validate_batch_size,
)

logger = logging.getLogger(__name__)

DEFAULT_RECONCILIATION_BATCH_SIZE = 500


class ProjectionReconciliationDriver(Protocol):
    """Driver operations required by one projection-reconciliation sweep."""

    embedding_dim: int

    async def execute_query(
        self, cypher_query_: str, /, **kwargs: Any
    ) -> tuple[list[dict[str, Any]], None, None]: ...

    async def save_vector_to_aoss_async(self, name: str, data: list[dict[str, Any]]) -> int: ...

    async def delete_from_aoss_async(
        self,
        name: str,
        *,
        uuids: list[str] | None = None,
        group_ids: list[str] | None = None,
        versions: dict[str, int] | None = None,
    ) -> int: ...


@dataclass
class ProjectionReconciliationStats:
    nodes_saved: int = 0
    edges_saved: int = 0
    nodes_deleted: int = 0
    edges_deleted: int = 0
    failures: int = 0

    @property
    def reconciled(self) -> int:
        return self.nodes_saved + self.edges_saved + self.nodes_deleted + self.edges_deleted


_PENDING_NODE_SAVE_QUERY = """
    MATCH (projection:Entity)
    WHERE projection._graphiti_vector_sync_pending =
            coalesce(projection._graphiti_projection_version, 0)
      AND coalesce(projection._graphiti_vector_delete_pending, false) = false
      {group_filter}
      {cursor_filter}
    RETURN projection.uuid AS uuid, projection.group_id AS group_id,
        CASE WHEN projection.name_embedding IS NULL OR projection.name_embedding = ''
            THEN null
            ELSE [x IN split(projection.name_embedding, ",") | toFloat(x)]
        END AS embedding,
        coalesce(projection._graphiti_projection_version, 0) AS projection_version
    ORDER BY projection.uuid DESC
    LIMIT $batch_size
"""

_PENDING_EDGE_SAVE_QUERY = """
    MATCH ()-[projection:RELATES_TO]->()
    WHERE projection._graphiti_vector_sync_pending =
            coalesce(projection._graphiti_projection_version, 0)
      AND coalesce(projection._graphiti_vector_delete_pending, false) = false
      {group_filter}
      {cursor_filter}
    RETURN projection.uuid AS uuid, projection.group_id AS group_id,
        CASE WHEN projection.fact_embedding IS NULL OR projection.fact_embedding = ''
            THEN null
            ELSE [x IN split(projection.fact_embedding, ",") | toFloat(x)]
        END AS embedding,
        coalesce(projection._graphiti_projection_version, 0) AS projection_version
    ORDER BY projection.uuid DESC
    LIMIT $batch_size
"""

_PENDING_NODE_DELETE_QUERY = """
    MATCH (projection:Entity)
    WHERE coalesce(projection._graphiti_vector_delete_pending, false) = true
      {group_filter}
      {cursor_filter}
    RETURN projection.uuid AS uuid,
        coalesce(projection._graphiti_projection_version, 0) AS projection_version
    ORDER BY projection.uuid DESC
    LIMIT $batch_size
"""

_PENDING_EDGE_DELETE_QUERY = """
    MATCH ()-[projection:RELATES_TO]->()
    WHERE coalesce(projection._graphiti_vector_delete_pending, false) = true
      {group_filter}
      {cursor_filter}
    RETURN projection.uuid AS uuid,
        coalesce(projection._graphiti_projection_version, 0) AS projection_version
    ORDER BY projection.uuid DESC
    LIMIT $batch_size
"""


async def _iter_pending_batches(
    driver: ProjectionReconciliationDriver,
    query_template: str,
    *,
    group_id: str | None,
    batch_size: int,
) -> AsyncIterator[list[dict[str, Any]]]:
    cursor: str | None = None
    while True:
        group_filter = 'AND projection.group_id = $group_id' if group_id is not None else ''
        cursor_filter = 'AND projection.uuid < $cursor' if cursor is not None else ''
        query = query_template.format(
            group_filter=group_filter,
            cursor_filter=cursor_filter,
        )
        params: dict[str, Any] = {'batch_size': batch_size, 'routing_': 'r'}
        if group_id is not None:
            params['group_id'] = group_id
        if cursor is not None:
            params['cursor'] = cursor

        records, _, _ = await driver.execute_query(query, **params)
        if not records:
            return
        yield records

        last_uuid = records[-1].get('uuid')
        if not isinstance(last_uuid, str) or not last_uuid:
            raise RuntimeError('Pending projection query returned an invalid pagination UUID')
        cursor = last_uuid
        if len(records) < batch_size:
            return


def _projection_identity(record: dict[str, Any]) -> tuple[str, int]:
    uuid = record.get('uuid')
    version = record.get('projection_version')
    if not isinstance(uuid, str) or not uuid:
        raise ValueError('Pending projection record requires a non-empty string UUID')
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise ValueError(f'Pending projection {uuid!r} has an invalid generation')
    return uuid, version


def _projection_document(
    driver: ProjectionReconciliationDriver,
    record: dict[str, Any],
) -> dict[str, Any]:
    uuid, version = _projection_identity(record)
    group_id = record.get('group_id')
    if not isinstance(group_id, str):
        raise ValueError(f'Pending projection {uuid!r} requires a string group_id')

    document: dict[str, Any] = {
        'uuid': uuid,
        'group_id': group_id,
        '_version': version,
    }
    embedding = record.get('embedding')
    if embedding is None:
        return document
    if not isinstance(embedding, list):
        raise ValueError(f'Pending projection {uuid!r} has a malformed embedding')
    expected_dimension = driver.embedding_dim
    if (
        not isinstance(expected_dimension, int)
        or isinstance(expected_dimension, bool)
        or expected_dimension <= 0
    ):
        raise ValueError('Vector projection driver requires a positive embedding dimension')
    if len(embedding) != expected_dimension or any(
        isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value)
        for value in embedding
    ):
        raise ValueError(f'Pending projection {uuid!r} has a malformed embedding')
    document['embedding'] = embedding
    return document


def _failure_context(record: dict[str, Any]) -> tuple[object, object]:
    """Return bounded identifiers only; never include graph or embedding content in logs."""
    return record.get('uuid', '<invalid>'), record.get('projection_version', '<invalid>')


async def _reconcile_save_batch(
    driver: ProjectionReconciliationDriver,
    projection_kind: str,
    index_name: str,
    batch: list[dict[str, Any]],
    stats: ProjectionReconciliationStats,
    batch_size: int,
) -> None:
    try:
        documents = [_projection_document(driver, record) for record in batch]
        indexed = await driver.save_vector_to_aoss_async(index_name, documents)
        if indexed != len(documents):
            raise RuntimeError(f'OpenSearch acknowledged {indexed}/{len(documents)} projections')
        await clear_projection_sync_pending(
            driver,  # type: ignore[arg-type]
            projection_kind,
            {document['uuid']: document['_version'] for document in documents},
            batch_size=batch_size,
        )
    except asyncio.CancelledError:
        raise
    except Exception as error:
        if len(batch) > 1:
            midpoint = len(batch) // 2
            await _reconcile_save_batch(
                driver,
                projection_kind,
                index_name,
                batch[:midpoint],
                stats,
                batch_size,
            )
            await _reconcile_save_batch(
                driver,
                projection_kind,
                index_name,
                batch[midpoint:],
                stats,
                batch_size,
            )
            return

        uuid, version = _failure_context(batch[0])
        stats.failures += 1
        logger.error(
            'Pending %s vector save remains for retry: uuid=%s generation=%s error_type=%s',
            projection_kind,
            uuid,
            version,
            type(error).__name__,
        )
        return

    if projection_kind == 'node':
        stats.nodes_saved += len(batch)
    else:
        stats.edges_saved += len(batch)


async def _delete_incident_edges(
    driver: ProjectionReconciliationDriver,
    node_deletions: list[dict[str, Any]],
    batch_size: int,
) -> None:
    """Drain incident edges before finalizing exact-generation pending node deletes."""
    from graphiti_core.driver.neptune.operations.entity_edge_ops import (
        NeptuneEntityEdgeOperations,
    )

    query = """
        UNWIND $deletions AS deletion
        MATCH (node:Entity {uuid: deletion.uuid})-[edge:RELATES_TO]-()
        WHERE coalesce(node._graphiti_vector_delete_pending, false) = true
          AND node._graphiti_projection_version = deletion.projection_version
        RETURN DISTINCT edge.uuid AS uuid
        ORDER BY uuid
        LIMIT $batch_size
    """
    edge_operations = NeptuneEntityEdgeOperations(driver)  # type: ignore[arg-type]
    while True:
        records, _, _ = await driver.execute_query(
            query,
            deletions=node_deletions,
            batch_size=batch_size,
        )
        if any(not isinstance(record.get('uuid'), str) or not record['uuid'] for record in records):
            raise ValueError('Pending node delete encountered an incident edge without a UUID')
        edge_uuids = list(dict.fromkeys(record['uuid'] for record in records))
        if not edge_uuids:
            return
        await edge_operations.delete_by_uuids(
            driver,  # type: ignore[arg-type]
            edge_uuids,
            batch_size=batch_size,
        )


async def _reconcile_delete_batch(
    driver: ProjectionReconciliationDriver,
    projection_kind: str,
    index_name: str,
    batch: list[dict[str, Any]],
    stats: ProjectionReconciliationStats,
    batch_size: int,
) -> None:
    try:
        identities = [_projection_identity(record) for record in batch]
        versions = dict(identities)
        deletions = [{'uuid': uuid, 'projection_version': version} for uuid, version in identities]
        if projection_kind == 'node':
            await _delete_incident_edges(driver, deletions, batch_size)

        deleted = await driver.delete_from_aoss_async(
            index_name,
            uuids=[uuid for uuid, _ in identities],
            versions=versions,
        )
        if deleted != len(identities):
            raise RuntimeError(f'OpenSearch acknowledged {deleted}/{len(identities)} tombstones')

        if projection_kind == 'node':
            finalize_query = """
                UNWIND $deletions AS deletion
                MATCH (projection:Entity {uuid: deletion.uuid})
                WHERE coalesce(projection._graphiti_vector_delete_pending, false) = true
                  AND projection._graphiti_projection_version = deletion.projection_version
                DETACH DELETE projection
            """
        else:
            finalize_query = """
                UNWIND $deletions AS deletion
                MATCH ()-[projection:RELATES_TO {uuid: deletion.uuid}]->()
                WHERE coalesce(projection._graphiti_vector_delete_pending, false) = true
                  AND projection._graphiti_projection_version = deletion.projection_version
                DELETE projection
            """
        await driver.execute_query(finalize_query, deletions=deletions)
    except asyncio.CancelledError:
        raise
    except Exception as error:
        if len(batch) > 1:
            midpoint = len(batch) // 2
            await _reconcile_delete_batch(
                driver,
                projection_kind,
                index_name,
                batch[:midpoint],
                stats,
                batch_size,
            )
            await _reconcile_delete_batch(
                driver,
                projection_kind,
                index_name,
                batch[midpoint:],
                stats,
                batch_size,
            )
            return

        uuid, version = _failure_context(batch[0])
        stats.failures += 1
        logger.error(
            'Pending %s vector delete remains for retry: uuid=%s generation=%s error_type=%s',
            projection_kind,
            uuid,
            version,
            type(error).__name__,
        )
        return

    if projection_kind == 'node':
        stats.nodes_deleted += len(batch)
    else:
        stats.edges_deleted += len(batch)


async def reconcile_pending_projections(
    driver: ProjectionReconciliationDriver,
    *,
    group_id: str | None = None,
    batch_size: int = DEFAULT_RECONCILIATION_BATCH_SIZE,
) -> ProjectionReconciliationStats:
    """Run one idempotent sweep over durable save and delete markers."""
    validate_batch_size(batch_size)
    stats = ProjectionReconciliationStats()

    # Resume explicit edge deletions first. Node repair can create additional edge work while
    # draining incident relationships, and any failure keeps the owning node pending for retry.
    async for batch in _iter_pending_batches(
        driver,
        _PENDING_EDGE_DELETE_QUERY,
        group_id=group_id,
        batch_size=batch_size,
    ):
        await _reconcile_delete_batch(
            driver,
            'edge',
            'edge_fact_embedding',
            batch,
            stats,
            batch_size,
        )

    async for batch in _iter_pending_batches(
        driver,
        _PENDING_NODE_DELETE_QUERY,
        group_id=group_id,
        batch_size=batch_size,
    ):
        await _reconcile_delete_batch(
            driver,
            'node',
            'node_name_embedding',
            batch,
            stats,
            batch_size,
        )

    # Rollback may disable new projections while delete cleanup must remain active. Do not clear
    # durable save markers without writing them; a later exact backfill owns that disabled state.
    if getattr(driver, 'vector_projection_enabled', True) is not False:
        for projection_kind, index_name, query in (
            ('node', 'node_name_embedding', _PENDING_NODE_SAVE_QUERY),
            ('edge', 'edge_fact_embedding', _PENDING_EDGE_SAVE_QUERY),
        ):
            async for batch in _iter_pending_batches(
                driver,
                query,
                group_id=group_id,
                batch_size=batch_size,
            ):
                await _reconcile_save_batch(
                    driver,
                    projection_kind,
                    index_name,
                    batch,
                    stats,
                    batch_size,
                )

    return stats


async def run_pending_projection_reconciler(
    driver: ProjectionReconciliationDriver,
    interval_seconds: float,
    batch_size: int = DEFAULT_RECONCILIATION_BATCH_SIZE,
) -> None:
    """Continuously repair durable projection markers, starting with an immediate sweep."""
    if interval_seconds <= 0:
        raise ValueError('interval_seconds must be greater than zero')
    validate_batch_size(batch_size)
    while True:
        try:
            stats = await reconcile_pending_projections(driver, batch_size=batch_size)
            if stats.reconciled or stats.failures:
                logger.info(
                    'Vector projection reconciliation: saved=%d nodes/%d edges, '
                    'deleted=%d nodes/%d edges, failures=%d',
                    stats.nodes_saved,
                    stats.edges_saved,
                    stats.nodes_deleted,
                    stats.edges_deleted,
                    stats.failures,
                )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.error(
                'Pending vector projection sweep failed; durable markers remain for retry '
                '(error_type=%s)',
                type(error).__name__,
            )
        await asyncio.sleep(interval_seconds)
