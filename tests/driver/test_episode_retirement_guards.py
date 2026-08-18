from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from neo4j.exceptions import ClientError

from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.driver.neo4j.schema import (
    delete_standalone_indexes,
    ensure_episode_uuid_uniqueness,
)
from graphiti_core.driver.query_executor import QueryExecutor
from graphiti_core.errors import GraphitiError
from graphiti_core.nodes import EpisodeType, EpisodicNode


@pytest.mark.asyncio
async def test_kuzu_episode_save_remains_compatible_with_explicit_schema():
    driver = KuzuDriver(':memory:')
    episode = EpisodicNode(
        name='kuzu compatibility',
        group_id='test',
        source=EpisodeType.text,
        source_description='test',
        content='content',
        created_at=datetime.now(timezone.utc),
        valid_at=datetime.now(timezone.utc),
    )

    try:
        await episode.save(driver)
        stored = await EpisodicNode.get_by_uuid(driver, episode.uuid)
        assert stored.uuid == episode.uuid
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_neo4j_episode_uuid_constraint_replaces_range_index_sequentially():
    execute_query = AsyncMock(
        side_effect=[
            ([], None, None),
            ([], None, None),
            ([], None, None),
            ([], None, None),
            ([{'name': 'episode_uuid_unique'}], None, None),
            ([], None, None),
            ([{'name': 'opr_retirement_request_id_unique'}], None, None),
        ]
    )
    executor = cast(
        QueryExecutor,
        SimpleNamespace(execute_query=execute_query),
    )

    await ensure_episode_uuid_uniqueness(executor)

    queries = [call.args[0] for call in execute_query.await_args_list]
    assert 'episode_uuid_unique' in queries[0]
    assert 'opr_retirement_request_id_unique' in queries[0]
    assert 'occurrences > 1' in queries[1]
    assert queries[2] == 'DROP INDEX episode_uuid IF EXISTS'
    assert queries[3].startswith('CREATE CONSTRAINT episode_uuid_unique')
    assert "name = 'episode_uuid_unique'" in queries[4]
    assert queries[5].startswith('CREATE CONSTRAINT opr_retirement_request_id_unique')
    assert "name = 'opr_retirement_request_id_unique'" in queries[6]


@pytest.mark.asyncio
async def test_neo4j_episode_uuid_constraints_fast_path_skips_scan_and_ddl():
    execute_query = AsyncMock(
        return_value=(
            [
                {'name': 'episode_uuid_unique'},
                {'name': 'opr_retirement_request_id_unique'},
            ],
            None,
            None,
        )
    )
    executor = cast(
        QueryExecutor,
        SimpleNamespace(execute_query=execute_query),
    )

    await ensure_episode_uuid_uniqueness(executor)

    execute_query.assert_awaited_once()
    assert execute_query.await_args is not None
    query = execute_query.await_args.args[0]
    assert 'SHOW CONSTRAINTS' in query
    assert 'episode_uuid_unique' in query
    assert 'opr_retirement_request_id_unique' in query


@pytest.mark.asyncio
async def test_neo4j_episode_uuid_constraint_adds_only_missing_receipt_constraint():
    execute_query = AsyncMock(
        side_effect=[
            ([{'name': 'episode_uuid_unique'}], None, None),
            ([], None, None),
            ([{'name': 'opr_retirement_request_id_unique'}], None, None),
        ]
    )
    executor = cast(
        QueryExecutor,
        SimpleNamespace(execute_query=execute_query),
    )

    await ensure_episode_uuid_uniqueness(executor)

    queries = [call.args[0] for call in execute_query.await_args_list]
    assert len(queries) == 3
    assert queries[1].startswith('CREATE CONSTRAINT opr_retirement_request_id_unique')
    assert "name = 'opr_retirement_request_id_unique'" in queries[2]


@pytest.mark.asyncio
async def test_neo4j_episode_uuid_constraint_adds_only_missing_episode_constraint():
    execute_query = AsyncMock(
        side_effect=[
            ([{'name': 'opr_retirement_request_id_unique'}], None, None),
            ([], None, None),
            ([], None, None),
            ([], None, None),
            ([{'name': 'episode_uuid_unique'}], None, None),
        ]
    )
    executor = cast(
        QueryExecutor,
        SimpleNamespace(execute_query=execute_query),
    )

    await ensure_episode_uuid_uniqueness(executor)

    queries = [call.args[0] for call in execute_query.await_args_list]
    assert len(queries) == 5
    assert 'occurrences > 1' in queries[1]
    assert queries[2] == 'DROP INDEX episode_uuid IF EXISTS'
    assert queries[3].startswith('CREATE CONSTRAINT episode_uuid_unique')
    assert "name = 'episode_uuid_unique'" in queries[4]
    assert not any(query.startswith('CREATE CONSTRAINT opr_retirement') for query in queries)


@pytest.mark.asyncio
async def test_neo4j_episode_uuid_constraint_race_still_requires_catalog_verification():
    execute_query = AsyncMock(
        side_effect=[
            ([], None, None),
            ([], None, None),
            ([], None, None),
            ClientError('EquivalentSchemaRuleAlreadyExists'),
            ([], None, None),
        ]
    )
    executor = cast(
        QueryExecutor,
        SimpleNamespace(execute_query=execute_query),
    )

    with pytest.raises(GraphitiError, match='episode UUID uniqueness constraint is unavailable'):
        await ensure_episode_uuid_uniqueness(executor)

    queries = [call.args[0] for call in execute_query.await_args_list]
    assert len(queries) == 5
    assert "name = 'episode_uuid_unique'" in queries[-1]


@pytest.mark.asyncio
async def test_neo4j_episode_uuid_constraint_rejects_existing_duplicates():
    execute_query = AsyncMock(
        side_effect=[
            ([], None, None),
            ([{'uuid': 'duplicate'}], None, None),
        ]
    )
    executor = cast(
        QueryExecutor,
        SimpleNamespace(execute_query=execute_query),
    )

    with pytest.raises(GraphitiError, match='duplicate UUIDs'):
        await ensure_episode_uuid_uniqueness(executor)

    assert execute_query.await_count == 2


@pytest.mark.asyncio
async def test_neo4j_index_rebuild_preserves_constraint_owned_indexes():
    execute_query = AsyncMock(
        side_effect=[
            ([{'name': 'episode_group_id'}, {'name': 'episode_content'}], None, None),
            ([], None, None),
            ([], None, None),
        ]
    )
    executor = cast(
        QueryExecutor,
        SimpleNamespace(execute_query=execute_query),
    )

    await delete_standalone_indexes(executor)

    queries = [call.args[0] for call in execute_query.await_args_list]
    assert 'owningConstraint IS NULL' in queries[0]
    assert queries[1:] == [
        'DROP INDEX `episode_group_id` IF EXISTS',
        'DROP INDEX `episode_content` IF EXISTS',
    ]
