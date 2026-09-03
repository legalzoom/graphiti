"""Backfill entity name and fact embeddings into the Neptune vector indexes.

Streams entity nodes and entity edges from Neptune in batches
ordered by uuid, and bulk-indexes the embeddings that already exist on them
into the `node_name_embedding` and `edge_fact_embedding` OpenSearch indexes
that graphiti_core.search.search_utils.node_similarity_search and
edge_similarity_search query. Entities with no stored embedding are skipped.

Re-running is safe: each write upserts by uuid. Because cursor pagination cannot
safely race ingestion or deletion, the CLI requires an explicit acknowledgement
that both are quiesced and runs a second, idempotent reconciliation pass before
declaring the vector indexes ready for reads.

Usage (run inside the cluster, where NEPTUNE_HOST/AOSS_HOST resolve):

    python -m graph_service.backfill_embeddings --all-groups --reset-vector-indices \
        --batch-size 500 \
        --acknowledge-ingestion-and-deletion-quiesced
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import sys
from collections.abc import AsyncIterator
from typing import Any, Protocol

from graphiti_core.driver.neptune.projection_versions import clear_projection_sync_pending
from graphiti_core.driver.neptune.vector_reconciliation import (
    ProjectionReconciliationDriver,
    reconcile_pending_projections,
    run_pending_projection_reconciler,
)

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 500


class BackfillError(RuntimeError):
    """Raised when a batch fails to index completely."""


class NeptuneQuerySource(Protocol):
    """The subset of NeptuneDriver the backfill needs to read from Neptune."""

    async def execute_query(
        self, cypher_query_: str, /, **kwargs: Any
    ) -> tuple[list[dict[str, Any]], None, None]: ...


class AossWriteSink(Protocol):
    """The subset of NeptuneDriver the backfill needs to write to AOSS."""

    embedding_dim: int

    def save_vector_to_aoss(self, name: str, data: list[dict[str, Any]]) -> int: ...

    async def save_vector_to_aoss_async(self, name: str, data: list[dict[str, Any]]) -> int: ...


class BackfillDriver(ProjectionReconciliationDriver, Protocol):
    """The complete Neptune surface used by a backfill or repair pass."""

    def save_vector_to_aoss(self, name: str, data: list[dict[str, Any]]) -> int: ...


_NODE_BATCH_QUERY = """
    MATCH (n:Entity)
    WHERE {group_filter}
        {embedding_filter}
        {pending_filter}
        AND coalesce(n._graphiti_vector_delete_pending, false) = false
    {cursor_filter}
    RETURN n.uuid AS uuid, n.group_id AS group_id,
        {embedding_expression} AS embedding,
        coalesce(n._graphiti_projection_version, 0) AS projection_version
    ORDER BY n.uuid DESC
    LIMIT $batch_size
"""

_EDGE_BATCH_QUERY = """
    MATCH (:Entity)-[e:RELATES_TO]->(:Entity)
    WHERE {group_filter}
        {embedding_filter}
        {pending_filter}
        AND coalesce(e._graphiti_vector_delete_pending, false) = false
    {cursor_filter}
    RETURN e.uuid AS uuid, e.group_id AS group_id,
        {embedding_expression} AS embedding,
        coalesce(e._graphiti_projection_version, 0) AS projection_version
    ORDER BY e.uuid DESC
    LIMIT $batch_size
"""


async def _iter_batches(
    driver: NeptuneQuerySource,
    query_template: str,
    cursor_field: str,
    group_id: str | None,
    batch_size: int,
    pending_only: bool,
) -> AsyncIterator[list[dict[str, Any]]]:
    _validate_batch_size(batch_size)
    cursor: str | None = None
    while True:
        cursor_filter = f'AND {cursor_field} < $cursor' if cursor is not None else ''
        subject = 'n' if cursor_field == 'n.uuid' else 'e'
        embedding_field = 'name_embedding' if subject == 'n' else 'fact_embedding'
        group_filter = f'{subject}.group_id = $group_id' if group_id is not None else 'true'
        embedding_filter = (
            ''
            if pending_only
            else f'AND {subject}.{embedding_field} IS NOT NULL '
            f"AND {subject}.{embedding_field} <> ''"
        )
        pending_filter = (
            f'AND {subject}._graphiti_vector_sync_pending = '
            f'coalesce({subject}._graphiti_projection_version, 0)'
            if pending_only
            else ''
        )
        embedding_expression = (
            f'CASE WHEN {subject}.{embedding_field} IS NULL '
            f"OR {subject}.{embedding_field} = '' THEN null ELSE "
            f'[x IN split({subject}.{embedding_field}, ",") | toFloat(x)] END'
        )
        query = query_template.format(
            cursor_filter=cursor_filter,
            group_filter=group_filter,
            embedding_filter=embedding_filter,
            pending_filter=pending_filter,
            embedding_expression=embedding_expression,
        )
        params: dict[str, Any] = {'batch_size': batch_size}
        if group_id is not None:
            params['group_id'] = group_id
        if cursor is not None:
            params['cursor'] = cursor

        records, _, _ = await driver.execute_query(query, routing_='r', **params)
        if not records:
            return

        yield records

        cursor = records[-1]['uuid']
        if len(records) < batch_size:
            return


def iter_node_batches(
    driver: NeptuneQuerySource,
    group_id: str | None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    pending_only: bool = False,
) -> AsyncIterator[list[dict[str, Any]]]:
    """Stream (uuid, group_id, embedding) batches for entity nodes with a name_embedding."""
    return _iter_batches(
        driver,
        _NODE_BATCH_QUERY,
        'n.uuid',
        group_id,
        batch_size,
        pending_only,
    )


def iter_edge_batches(
    driver: NeptuneQuerySource,
    group_id: str | None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    pending_only: bool = False,
) -> AsyncIterator[list[dict[str, Any]]]:
    """Stream (uuid, group_id, embedding) batches for entity edges with a fact_embedding."""
    return _iter_batches(
        driver,
        _EDGE_BATCH_QUERY,
        'e.uuid',
        group_id,
        batch_size,
        pending_only,
    )


def _build_documents(
    driver: AossWriteSink,
    index_name: str,
    batch: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate graph records and build bounded OpenSearch projection documents."""
    expected_dim = driver.embedding_dim
    if not isinstance(expected_dim, int) or isinstance(expected_dim, bool) or expected_dim <= 0:
        raise BackfillError(
            f'Invalid configured embedding dimension {expected_dim!r}; expected a positive integer.'
        )

    docs: list[dict[str, Any]] = []
    for record in batch:
        embedding = record.get('embedding')
        record_uuid = record.get('uuid', '<missing uuid>')
        document: dict[str, Any] = {
            'uuid': record['uuid'],
            'group_id': record['group_id'],
            '_version': record['projection_version'],
        }
        if embedding is None:
            docs.append(document)
            continue
        if not isinstance(embedding, list):
            raise BackfillError(
                f"Record {record_uuid!r} for '{index_name}' has a malformed embedding; "
                'expected a list of finite numbers. Fix or regenerate the Neptune embedding, '
                'then re-run the idempotent backfill.'
            )
        if len(embedding) != expected_dim:
            raise BackfillError(
                f"Record {record_uuid!r} for '{index_name}' has embedding dimension "
                f'{len(embedding)}; expected {expected_dim}. Fix or regenerate the Neptune '
                'embedding, then re-run the idempotent backfill.'
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            for value in embedding
        ):
            raise BackfillError(
                f"Record {record_uuid!r} for '{index_name}' has a malformed embedding; "
                'expected a list of finite numbers. Fix or regenerate the Neptune embedding, '
                'then re-run the idempotent backfill.'
            )

        document['embedding'] = embedding
        docs.append(document)

    return docs


def index_batch(driver: AossWriteSink, index_name: str, batch: list[dict[str, Any]]) -> int:
    """Bulk-index one batch into a vector index. Raises BackfillError if any document fails."""
    if not batch:
        return 0

    docs = _build_documents(driver, index_name, batch)
    success = driver.save_vector_to_aoss(index_name, docs)
    if success != len(docs):
        raise BackfillError(
            f"Bulk index into '{index_name}' indexed {success}/{len(docs)} documents. "
            'Check OpenSearch logs and re-run; the write is idempotent.'
        )
    return success


async def index_batch_async(
    driver: AossWriteSink,
    index_name: str,
    batch: list[dict[str, Any]],
) -> int:
    """Validate a batch, then write it without blocking the event loop."""
    if not batch:
        return 0

    documents = _build_documents(driver, index_name, batch)
    success = await driver.save_vector_to_aoss_async(index_name, documents)
    if success != len(documents):
        raise BackfillError(
            f"Bulk index into '{index_name}' indexed {success}/{len(documents)} "
            'documents. Check OpenSearch logs and re-run; the write is idempotent.'
        )
    return success


class BackfillStats:
    def __init__(self) -> None:
        self.nodes_indexed = 0
        self.edges_indexed = 0
        self.nodes_deleted = 0
        self.edges_deleted = 0
        self.failures = 0


async def run_backfill(
    driver: BackfillDriver,
    group_id: str | None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    pending_only: bool = False,
) -> BackfillStats:
    """Backfill vector projections for one group, or every group when group_id is None.

    `driver` must implement both NeptuneQuerySource and AossWriteSink, as a
    real NeptuneDriver does.
    """
    _validate_batch_size(batch_size)
    stats = BackfillStats()

    if pending_only:
        reconciliation = await reconcile_pending_projections(
            driver,
            group_id=group_id,
            batch_size=batch_size,
        )
        stats.nodes_indexed = reconciliation.nodes_saved
        stats.edges_indexed = reconciliation.edges_saved
        stats.nodes_deleted = reconciliation.nodes_deleted
        stats.edges_deleted = reconciliation.edges_deleted
        stats.failures = reconciliation.failures
        return stats

    async for batch in iter_node_batches(driver, group_id, batch_size, pending_only):
        indexed = await index_batch_async(driver, 'node_name_embedding', batch)
        await clear_projection_sync_pending(
            driver,  # type: ignore[arg-type]
            'node',
            {record['uuid']: record['projection_version'] for record in batch},
            batch_size=batch_size,
        )
        stats.nodes_indexed += indexed
        logger.info('Indexed %d entity node embeddings (total %d)', indexed, stats.nodes_indexed)

    async for batch in iter_edge_batches(driver, group_id, batch_size, pending_only):
        indexed = await index_batch_async(driver, 'edge_fact_embedding', batch)
        await clear_projection_sync_pending(
            driver,  # type: ignore[arg-type]
            'edge',
            {record['uuid']: record['projection_version'] for record in batch},
            batch_size=batch_size,
        )
        stats.edges_indexed += indexed
        logger.info('Indexed %d entity edge embeddings (total %d)', indexed, stats.edges_indexed)

    return stats


run_pending_reconciler = run_pending_projection_reconciler


def _build_driver(
    host: str,
    aoss_host: str,
    port: int,
    aoss_port: int,
    vector_aoss_host: str | None,
    vector_aoss_port: int | None,
):
    from graphiti_core.driver.neptune_driver import NeptuneDriver

    return NeptuneDriver(
        host=host,
        aoss_host=aoss_host,
        port=port,
        aoss_port=aoss_port,
        vector_aoss_host=vector_aoss_host,
        vector_aoss_port=vector_aoss_port,
        vector_projection_enabled=True,
        vector_search_enabled=False,
    )


def _validate_batch_size(batch_size: int) -> None:
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError(f'batch_size must be a positive integer; got {batch_size!r}')


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'{value!r} is not an integer') from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f'{value!r} must be a positive integer')
    return parsed


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument('--group-id', help='backfill one group (does not authorize global reads)')
    scope.add_argument(
        '--all-groups',
        action='store_true',
        help='backfill every group; required for the initial driver-wide rollout',
    )
    parser.add_argument(
        '--batch-size',
        type=_positive_int,
        default=DEFAULT_BATCH_SIZE,
        help=f'records fetched from Neptune per batch (default {DEFAULT_BATCH_SIZE})',
    )
    parser.add_argument(
        '--reset-vector-indices',
        action='store_true',
        help=(
            'delete and recreate both vector indexes before backfill; required for initial '
            'rollout to remove stale AOSS-only documents and allowed only with --all-groups'
        ),
    )
    parser.add_argument(
        '--reset-group-vector-documents',
        action='store_true',
        help=(
            'hard-delete vector documents in --group-id before rebuilding that quiesced scope; '
            'required for an exact single-group repair'
        ),
    )
    parser.add_argument(
        '--pending-only',
        action='store_true',
        help=(
            'reconcile only graph generations left pending by a failed vector write; safe with '
            'concurrent ingestion and does not authorize initial rollout'
        ),
    )
    parser.add_argument(
        '--acknowledge-ingestion-and-deletion-quiesced',
        action='store_true',
        help=(
            'required safety acknowledgement that ingestion and deletion in the selected scope '
            'are stopped, all in-flight operations have completed, and the scope will remain '
            'quiesced until reset and both backfill passes finish'
        ),
    )
    args = parser.parse_args(argv)
    if args.reset_vector_indices and not args.all_groups:
        parser.error('--reset-vector-indices requires --all-groups')
    if args.reset_group_vector_documents and args.group_id is None:
        parser.error('--reset-group-vector-documents requires --group-id')
    if args.pending_only:
        if args.reset_vector_indices or args.reset_group_vector_documents:
            parser.error('--pending-only cannot be combined with reset options')
        return args
    if not args.acknowledge_ingestion_and_deletion_quiesced:
        parser.error('exact backfill requires --acknowledge-ingestion-and-deletion-quiesced')
    if args.all_groups and not args.reset_vector_indices:
        parser.error('--all-groups exact backfill requires --reset-vector-indices')
    if args.group_id is not None and not args.reset_group_vector_documents:
        parser.error('--group-id exact repair requires --reset-group-vector-documents')
    return args


async def _main_async(argv: list[str] | None) -> int:
    from graph_service.config import get_settings

    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)

    settings = get_settings()
    if settings.db_backend != 'neptune':
        logger.error('DB_BACKEND is %r; the backfill only applies to neptune', settings.db_backend)
        return 1
    if not settings.neptune_host or not settings.aoss_host:
        logger.error('NEPTUNE_HOST and AOSS_HOST are required to run the backfill')
        return 1

    driver = _build_driver(
        host=settings.neptune_host,
        aoss_host=settings.aoss_host,
        port=settings.neptune_port or 8182,
        aoss_port=settings.aoss_port or 443,
        vector_aoss_host=settings.vector_aoss_host,
        vector_aoss_port=settings.vector_aoss_port,
    )
    selected_group_id = None if args.all_groups else args.group_id
    scope_description = 'all groups' if args.all_groups else f'group_id={args.group_id}'
    try:
        if args.pending_only:
            await driver.create_vector_aoss_indices(wait_for_propagation=True)
            stats = await run_backfill(
                driver,
                selected_group_id,
                args.batch_size,
                pending_only=True,
            )
            logger.info(
                'Pending vector reconciliation complete for %s: saved=%d nodes/%d edges, '
                'deleted=%d nodes/%d edges, failures=%d',
                scope_description,
                stats.nodes_indexed,
                stats.edges_indexed,
                stats.nodes_deleted,
                stats.edges_deleted,
                stats.failures,
            )
            return 1 if stats.failures else 0
        if args.reset_vector_indices:
            logger.info('Resetting both vector indexes before the all-groups initial backfill')
            await driver.delete_vector_aoss_indices()
        await driver.create_vector_aoss_indices(wait_for_propagation=True)
        if args.reset_group_vector_documents:
            logger.info(
                'Purging stale vector documents before exact repair for group_id=%s',
                args.group_id,
            )
            await driver.purge_vector_aoss_group_documents_async(args.group_id)
        logger.info(
            'Starting initial backfill for %s; ingestion and deletion must remain quiesced',
            scope_description,
        )
        initial_stats = await run_backfill(driver, selected_group_id, args.batch_size)
        logger.info(
            'Initial pass complete for %s: %d node embeddings, %d edge embeddings. '
            'Starting final idempotent reconciliation pass.',
            scope_description,
            initial_stats.nodes_indexed,
            initial_stats.edges_indexed,
        )
        reconciliation_stats = await run_backfill(driver, selected_group_id, args.batch_size)
    except Exception as error:
        logger.error(
            'Backfill or final reconciliation failed; keep vector reads disabled and re-run '
            'both passes while ingestion and deletion remain quiesced (error_type=%s)',
            type(error).__name__,
        )
        return 1
    finally:
        await driver.close()

    completion_guidance = (
        'The all-groups scope is complete; driver-wide vector reads may now be enabled.'
        if args.all_groups
        else (
            'This single group is reconciled. Do not enable driver-wide vector reads unless '
            'every group has been backfilled.'
        )
    )
    logger.info(
        'Backfill and final reconciliation complete for %s: initial=%d nodes/%d edges, '
        'reconciliation=%d nodes/%d edges. %s',
        scope_description,
        initial_stats.nodes_indexed,
        initial_stats.edges_indexed,
        reconciliation_stats.nodes_indexed,
        reconciliation_stats.edges_indexed,
        completion_guidance,
    )
    return 0


def main(argv: list[str] | None = None) -> None:
    sys.exit(asyncio.run(_main_async(argv)))


if __name__ == '__main__':
    main()
