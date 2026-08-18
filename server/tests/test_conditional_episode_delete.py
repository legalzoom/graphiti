from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from graph_service.dto import DeleteEpisodeIfMatchRequest
from graph_service.routers.ingest import delete_episode_if_matches
from graph_service.zep_graphiti import ZepGraphiti


@pytest.mark.asyncio
async def test_conditional_delete_is_one_atomic_graph_statement():
    driver = SimpleNamespace(
        execute_query=AsyncMock(return_value=([{'uuid': 'episode-id'}], None, None))
    )
    service = SimpleNamespace(driver=driver)

    deleted = await ZepGraphiti.delete_episodic_node_if_matches(
        service,
        'episode-id',
        group_id='opr',
        name='curated:test.md',
        content='stored content',
        source_description='publish',
    )

    assert deleted is True
    query = driver.execute_query.await_args.args[0]
    assert 'MATCH (episode:Episodic {uuid: $uuid, group_id: $group_id})' in query
    assert query.index('SET episode._opr_conditional_delete_lock') < query.index(
        'episode.name = $name'
    )
    assert 'REMOVE episode._opr_conditional_delete_lock' in query
    assert 'episode.name = $name' in query
    assert 'episode.content = $content' in query
    assert 'episode.source_description = $source_description' in query
    assert 'DETACH DELETE episode' in query
    assert driver.execute_query.await_args.kwargs == {
        'uuid': 'episode-id',
        'group_id': 'opr',
        'name': 'curated:test.md',
        'content': 'stored content',
        'source_description': 'publish',
    }


@pytest.mark.asyncio
async def test_conditional_delete_returns_false_on_identity_mismatch():
    driver = SimpleNamespace(execute_query=AsyncMock(return_value=([], None, None)))
    service = SimpleNamespace(driver=driver)

    deleted = await ZepGraphiti.delete_episodic_node_if_matches(
        service,
        'episode-id',
        group_id='opr',
        name='curated:test.md',
        content='changed content',
        source_description='publish',
    )

    assert deleted is False


@pytest.mark.asyncio
async def test_conditional_delete_route_fails_precondition_without_success_receipt():
    graphiti = MagicMock()
    graphiti.delete_episodic_node_if_matches = AsyncMock(return_value=False)
    request = DeleteEpisodeIfMatchRequest(
        group_id='opr',
        name='curated:test.md',
        content='stored content',
        source_description='publish',
    )

    with pytest.raises(HTTPException) as exc_info:
        await delete_episode_if_matches('episode-id', request, graphiti)

    assert exc_info.value.status_code == 412


@pytest.mark.asyncio
async def test_conditional_delete_route_returns_success_only_after_atomic_match():
    graphiti = MagicMock()
    graphiti.delete_episodic_node_if_matches = AsyncMock(return_value=True)
    request = DeleteEpisodeIfMatchRequest(
        group_id='opr',
        name='curated:test.md',
        content='stored content',
        source_description='publish',
    )

    result = await delete_episode_if_matches('episode-id', request, graphiti)

    assert result.success is True
    graphiti.delete_episodic_node_if_matches.assert_awaited_once_with(
        'episode-id',
        group_id='opr',
        name='curated:test.md',
        content='stored content',
        source_description='publish',
    )
