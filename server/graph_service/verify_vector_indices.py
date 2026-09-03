"""Check whether the AOSS collection accepts knn_vector fields.

Attempts to create (or confirm existing) the node_name_embedding and
edge_fact_embedding vector indexes and reports PASS or the exact AOSS
rejection. Does not run the embeddings backfill and does not touch the four
text indexes.

An OpenSearch Serverless collection must be of type VECTORSEARCH to host a
knn_vector field; a SEARCH type collection rejects the mapping. Run this on
the dev canary to answer that question in one pass before running
`graph_service.backfill_embeddings`.

Usage:

    python -m graph_service.verify_vector_indices
"""

from __future__ import annotations

import asyncio
import logging
import sys

logger = logging.getLogger(__name__)


async def _main_async() -> int:
    from graph_service.config import get_settings

    logging.basicConfig(level=logging.INFO)

    settings = get_settings()
    if settings.db_backend != 'neptune':
        logger.error(
            'DB_BACKEND is %r; vector index verification only applies to neptune',
            settings.db_backend,
        )
        return 1
    if not settings.neptune_host or not settings.aoss_host:
        logger.error('NEPTUNE_HOST and AOSS_HOST are required to verify vector indexes')
        return 1

    from graphiti_core.driver.neptune_driver import NeptuneDriver, VectorIndexUnsupportedError

    driver = NeptuneDriver(
        host=settings.neptune_host,
        aoss_host=settings.aoss_host,
        port=settings.neptune_port or 8182,
        aoss_port=settings.aoss_port or 443,
    )
    try:
        await driver.create_vector_aoss_indices()
    except VectorIndexUnsupportedError as e:
        logger.error('FAIL: %s', e)
        return 1
    finally:
        await driver.close()

    logger.info(
        'PASS: node_name_embedding and edge_fact_embedding accept knn_vector on host %s',
        settings.aoss_host,
    )
    return 0


def main() -> None:
    sys.exit(asyncio.run(_main_async()))


if __name__ == '__main__':
    main()
