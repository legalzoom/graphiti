"""Sequential Neo4j schema upgrades required by episode retirement."""

import asyncio

from neo4j.exceptions import ClientError

from graphiti_core.driver.query_executor import QueryExecutor
from graphiti_core.errors import GraphitiError
from graphiti_core.helpers import query_result_record_count


async def ensure_episode_uuid_uniqueness(executor: QueryExecutor) -> None:
    """Replace the historical UUID range index with a uniqueness constraint.

    ``MERGE`` cannot prevent concurrent creation of two otherwise absent nodes
    without a uniqueness constraint. Episode retirement relies on that
    serialization so a writer cannot create a second same-UUID node beside a
    tombstone. Neo4j does not allow the old standalone range index and the
    index-backed constraint on the same schema, so this upgrade is deliberately
    ordered and fail-closed on any pre-existing duplicate.
    """
    lock = getattr(executor, '_episode_uuid_schema_lock', None)
    if lock is None:
        lock = asyncio.Lock()
        vars(executor)['_episode_uuid_schema_lock'] = lock
    async with lock:
        await _ensure_episode_uuid_uniqueness(executor)


async def _ensure_episode_uuid_uniqueness(executor: QueryExecutor) -> None:
    duplicates = await executor.execute_query(
        """
        MATCH (episode:Episodic)
        WHERE episode.uuid IS NOT NULL
        WITH episode.uuid AS uuid, count(*) AS occurrences
        WHERE occurrences > 1
        RETURN uuid
        LIMIT 1
        """
    )
    if await query_result_record_count(duplicates):
        raise GraphitiError('cannot enforce episode UUID uniqueness while duplicate UUIDs exist')

    await executor.execute_query('DROP INDEX episode_uuid IF EXISTS')
    try:
        await executor.execute_query(
            'CREATE CONSTRAINT episode_uuid_unique IF NOT EXISTS '
            'FOR (episode:Episodic) REQUIRE episode.uuid IS UNIQUE'
        )
    except ClientError as exc:
        # Separate service replicas can race this one-time upgrade. Neo4j may
        # surface the equivalent-rule race despite IF NOT EXISTS; the catalog
        # verification below is still authoritative.
        if 'EquivalentSchemaRuleAlreadyExists' not in str(exc):
            raise
    constraint = await executor.execute_query(
        """
        SHOW CONSTRAINTS YIELD name
        WHERE name = 'episode_uuid_unique'
        RETURN name
        """
    )
    if await query_result_record_count(constraint) != 1:
        raise GraphitiError('episode UUID uniqueness constraint is unavailable')
