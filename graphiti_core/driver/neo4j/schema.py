"""Sequential Neo4j schema upgrades required by episode retirement."""

import asyncio
from typing import Any, cast

from neo4j.exceptions import ClientError
from typing_extensions import LiteralString

from graphiti_core.driver.query_executor import QueryExecutor
from graphiti_core.errors import GraphitiError
from graphiti_core.helpers import query_result_record_count


def _query_records(result: Any) -> list[Any]:
    if isinstance(result, tuple):
        return list(result[0])
    records = getattr(result, 'records', None)
    if records is not None:
        return list(records)
    raise TypeError(f'unsupported Neo4j query result: {type(result).__name__}')


async def delete_standalone_indexes(executor: QueryExecutor) -> None:
    """Drop rebuildable indexes without touching constraint-owned indexes."""
    result = await executor.execute_query(
        """
        SHOW INDEXES YIELD name, owningConstraint
        WHERE owningConstraint IS NULL
        RETURN name
        """
    )
    for record in _query_records(result):
        name = record['name']
        if not isinstance(name, str):
            raise TypeError('Neo4j index name must be a string')
        escaped_name = name.replace('`', '``')
        await executor.execute_query(cast(LiteralString, f'DROP INDEX `{escaped_name}` IF EXISTS'))


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
    constraints = await executor.execute_query(
        """
        SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties
        WHERE entityType = 'NODE'
          AND type IN ['UNIQUENESS', 'NODE_PROPERTY_UNIQUENESS', 'NODE_KEY']
          AND ((labelsOrTypes = ['Episodic'] AND properties = ['uuid'])
           OR (labelsOrTypes = ['OPRRetirementReceipt']
               AND properties = ['request_id']))
        RETURN name, labelsOrTypes, properties
        """
    )
    existing_schemas = {
        (tuple(record['labelsOrTypes']), tuple(record['properties']))
        for record in _query_records(constraints)
    }
    episode_constraint_exists = (('Episodic',), ('uuid',)) in existing_schemas
    receipt_constraint_exists = (
        ('OPRRetirementReceipt',),
        ('request_id',),
    ) in existing_schemas
    if episode_constraint_exists and receipt_constraint_exists:
        return

    if not episode_constraint_exists:
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
            raise GraphitiError(
                'cannot enforce episode UUID uniqueness while duplicate UUIDs exist'
            )

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
            SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties
            WHERE entityType = 'NODE'
              AND type IN ['UNIQUENESS', 'NODE_PROPERTY_UNIQUENESS', 'NODE_KEY']
              AND labelsOrTypes = ['Episodic'] AND properties = ['uuid']
            RETURN name, labelsOrTypes, properties
            """
        )
        if await query_result_record_count(constraint) != 1:
            raise GraphitiError('episode UUID uniqueness constraint is unavailable')

    if not receipt_constraint_exists:
        try:
            await executor.execute_query(
                'CREATE CONSTRAINT opr_retirement_request_id_unique IF NOT EXISTS '
                'FOR (receipt:OPRRetirementReceipt) REQUIRE receipt.request_id IS UNIQUE'
            )
        except ClientError as exc:
            if 'EquivalentSchemaRuleAlreadyExists' not in str(exc):
                raise
        receipt_constraint = await executor.execute_query(
            """
            SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties
            WHERE entityType = 'NODE'
              AND type IN ['UNIQUENESS', 'NODE_PROPERTY_UNIQUENESS', 'NODE_KEY']
              AND labelsOrTypes = ['OPRRetirementReceipt']
              AND properties = ['request_id']
            RETURN name, labelsOrTypes, properties
            """
        )
        if await query_result_record_count(receipt_constraint) != 1:
            raise GraphitiError('retirement request ID uniqueness constraint is unavailable')
