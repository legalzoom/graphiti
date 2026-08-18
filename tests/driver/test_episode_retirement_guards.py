from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

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
            ([{'name': 'episode_uuid_unique'}], None, None),
        ]
    )
    executor = cast(
        QueryExecutor,
        SimpleNamespace(execute_query=execute_query),
    )

    await ensure_episode_uuid_uniqueness(executor)

    queries = [call.args[0] for call in execute_query.await_args_list]
    assert 'occurrences > 1' in queries[0]
    assert queries[1] == 'DROP INDEX episode_uuid IF EXISTS'
    assert queries[2].startswith('CREATE CONSTRAINT episode_uuid_unique')
    assert "name = 'episode_uuid_unique'" in queries[3]


@pytest.mark.asyncio
async def test_neo4j_episode_uuid_constraint_rejects_existing_duplicates():
    execute_query = AsyncMock(return_value=([{'uuid': 'duplicate'}], None, None))
    executor = cast(
        QueryExecutor,
        SimpleNamespace(execute_query=execute_query),
    )

    with pytest.raises(GraphitiError, match='duplicate UUIDs'):
        await ensure_episode_uuid_uniqueness(executor)

    execute_query.assert_awaited_once()


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
