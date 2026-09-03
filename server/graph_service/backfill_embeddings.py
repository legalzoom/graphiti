"""Backfill entity name and fact embeddings into the Neptune vector indexes.

Streams entity nodes and entity edges for one group_id from Neptune in batches
ordered by uuid, and bulk-indexes the embeddings that already exist on them
into the `node_name_embedding` and `edge_fact_embedding` OpenSearch indexes
that graphiti_core.search.search_utils.node_similarity_search and
edge_similarity_search query. Entities with no stored embedding are skipped.

Re-running is safe: each write upserts by uuid.

Usage (run inside the cluster, where NEPTUNE_HOST/AOSS_HOST resolve):

    python -m graph_service.backfill_embeddings --group-id opr --batch-size 500
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import AsyncIterator
from typing import Any, Protocol

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

    def save_to_aoss(self, name: str, data: list[dict[str, Any]]) -> int: ...


_NODE_BATCH_QUERY = """
    MATCH (n:Entity)
    WHERE n.group_id = $group_id AND n.name_embedding IS NOT NULL
    {cursor_filter}
    RETURN n.uuid AS uuid, n.group_id AS group_id,
        [x IN split(n.name_embedding, ",") | toFloat(x)] AS embedding
    ORDER BY n.uuid DESC
    LIMIT $batch_size
"""

_EDGE_BATCH_QUERY = """
    MATCH (:Entity)-[e:RELATES_TO]->(:Entity)
    WHERE e.group_id = $group_id AND e.fact_embedding IS NOT NULL
    {cursor_filter}
    RETURN e.uuid AS uuid, e.group_id AS group_id,
        [x IN split(e.fact_embedding, ",") | toFloat(x)] AS embedding
    ORDER BY e.uuid DESC
    LIMIT $batch_size
"""


async def _iter_batches(
    driver: NeptuneQuerySource,
    query_template: str,
    cursor_field: str,
    group_id: str,
    batch_size: int,
) -> AsyncIterator[list[dict[str, Any]]]:
    cursor: str | None = None
    while True:
        cursor_filter = f'AND {cursor_field} < $cursor' if cursor is not None else ''
        query = query_template.format(cursor_filter=cursor_filter)
        params: dict[str, Any] = {'group_id': group_id, 'batch_size': batch_size}
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
    driver: NeptuneQuerySource, group_id: str, batch_size: int = DEFAULT_BATCH_SIZE
) -> AsyncIterator[list[dict[str, Any]]]:
    """Stream (uuid, group_id, embedding) batches for entity nodes with a name_embedding."""
    return _iter_batches(driver, _NODE_BATCH_QUERY, 'n.uuid', group_id, batch_size)


def iter_edge_batches(
    driver: NeptuneQuerySource, group_id: str, batch_size: int = DEFAULT_BATCH_SIZE
) -> AsyncIterator[list[dict[str, Any]]]:
    """Stream (uuid, group_id, embedding) batches for entity edges with a fact_embedding."""
    return _iter_batches(driver, _EDGE_BATCH_QUERY, 'e.uuid', group_id, batch_size)


def index_batch(driver: AossWriteSink, index_name: str, batch: list[dict[str, Any]]) -> int:
    """Bulk-index one batch into a vector index. Raises BackfillError if any document fails."""
    if not batch:
        return 0

    docs = [
        {'uuid': record['uuid'], 'group_id': record['group_id'], 'embedding': record['embedding']}
        for record in batch
    ]
    success = driver.save_to_aoss(index_name, docs)
    if success != len(docs):
        raise BackfillError(
            f"Bulk index into '{index_name}' indexed {success}/{len(docs)} documents. "
            'Check OpenSearch logs and re-run; the write is idempotent.'
        )
    return success


class BackfillStats:
    def __init__(self) -> None:
        self.nodes_indexed = 0
        self.edges_indexed = 0


async def run_backfill(
    driver: NeptuneQuerySource | AossWriteSink,
    group_id: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> BackfillStats:
    """Backfill node_name_embedding and edge_fact_embedding for one group_id.

    `driver` must implement both NeptuneQuerySource and AossWriteSink, as a
    real NeptuneDriver does.
    """
    stats = BackfillStats()

    async for batch in iter_node_batches(driver, group_id, batch_size):  # type: ignore[arg-type]
        indexed = index_batch(driver, 'node_name_embedding', batch)  # type: ignore[arg-type]
        stats.nodes_indexed += indexed
        logger.info('Indexed %d entity node embeddings (total %d)', indexed, stats.nodes_indexed)

    async for batch in iter_edge_batches(driver, group_id, batch_size):  # type: ignore[arg-type]
        indexed = index_batch(driver, 'edge_fact_embedding', batch)  # type: ignore[arg-type]
        stats.edges_indexed += indexed
        logger.info('Indexed %d entity edge embeddings (total %d)', indexed, stats.edges_indexed)

    return stats


def _build_driver(host: str, aoss_host: str, port: int, aoss_port: int):
    from graphiti_core.driver.neptune_driver import NeptuneDriver

    return NeptuneDriver(host=host, aoss_host=aoss_host, port=port, aoss_port=aoss_port)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group-id', required=True, help='group_id to backfill')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'records fetched from Neptune per batch (default {DEFAULT_BATCH_SIZE})',
    )
    return parser.parse_args(argv)


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
    )
    try:
        await driver.create_aoss_indices()
        stats = await run_backfill(driver, args.group_id, args.batch_size)
    except BackfillError:
        logger.exception('Backfill failed')
        return 1
    finally:
        await driver.close()

    logger.info(
        'Backfill complete for group_id=%s: %d node embeddings, %d edge embeddings',
        args.group_id,
        stats.nodes_indexed,
        stats.edges_indexed,
    )
    return 0


def main(argv: list[str] | None = None) -> None:
    sys.exit(asyncio.run(_main_async(argv)))


if __name__ == '__main__':
    main()
