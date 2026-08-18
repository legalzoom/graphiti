from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.driver.falkordb.operations.episode_node_ops import FalkorEpisodeNodeOperations
from graphiti_core.driver.falkordb.operations.graph_ops import FalkorGraphMaintenanceOperations
from graphiti_core.driver.neo4j.operations.episode_node_ops import Neo4jEpisodeNodeOperations
from graphiti_core.driver.neo4j.operations.graph_ops import Neo4jGraphMaintenanceOperations
from graphiti_core.driver.neptune.operations.episode_node_ops import NeptuneEpisodeNodeOperations
from graphiti_core.driver.neptune.operations.graph_ops import NeptuneGraphMaintenanceOperations
from graphiti_core.helpers import EPISODE_AOSS_TOMBSTONE_VERSION
from graphiti_core.models.nodes.node_db_queries import (
    get_episode_node_save_bulk_query,
    get_episode_node_save_query,
)
from graphiti_core.nodes import EpisodicNode
from pydantic import SecretStr

from graph_service.config import Settings
from graph_service.dto import DeleteEpisodeIfMatchRequest
from graph_service.protocol import (
    GRAPHITI_RECONCILIATION_OPERATION_RETIRE_EPISODE,
    GRAPHITI_RECONCILIATION_PROTOCOL,
)
from graph_service.routers.ingest import delete_episode_if_matches
from graph_service.routers.retrieve import get_episodes, get_episodes_for_reconciliation
from graph_service.zep_graphiti import ZepGraphiti, _conditional_episode_identity_digest

EPISODE_UUID = '11111111-1111-4111-8111-111111111111'


def test_privileged_listing_and_retirement_tokens_must_be_distinct():
    sentinel = 'super-secret-identical-token-abc123'
    with pytest.raises(ValueError, match='tokens must be distinct') as exc_info:
        Settings.model_validate(
            {
                'openai_api_key': 'test',
                'opr_reconciliation_token': sentinel,
                'opr_retirement_token': sentinel,
            }
        )
    assert sentinel not in str(exc_info.value)


@pytest.mark.asyncio
async def test_retired_reconciliation_listing_requires_distinct_service_token():
    reconciliation_listing = AsyncMock(return_value=[])
    graphiti = cast(
        ZepGraphiti,
        SimpleNamespace(
            retrieve_episodes_for_reconciliation=reconciliation_listing,
            retrieve_episodes=AsyncMock(return_value=[]),
        ),
    )
    settings = cast(
        Settings,
        SimpleNamespace(opr_reconciliation_token=SecretStr('reconcile-secret')),
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_episodes(
            'opr',
            20,
            graphiti,
            settings,
            include_retired_for_reconciliation=True,
            x_opr_reconciliation_token='wrong',
        )
    assert exc_info.value.status_code == 403
    reconciliation_listing.assert_not_awaited()

    assert (
        await get_episodes(
            'opr',
            20,
            graphiti,
            settings,
            include_retired_for_reconciliation=True,
            x_opr_reconciliation_token='reconcile-secret',
        )
        == []
    )
    reconciliation_listing.assert_awaited_once_with('opr', 20)


@pytest.mark.asyncio
async def test_dedicated_reconciliation_listing_attests_same_response(monkeypatch):
    episode = MagicMock()
    reconciliation_listing = AsyncMock(return_value=[episode])
    graphiti = cast(
        ZepGraphiti,
        SimpleNamespace(
            retrieve_episodes_for_reconciliation=reconciliation_listing,
        ),
    )
    settings = cast(
        Settings,
        SimpleNamespace(opr_reconciliation_token=SecretStr('reconcile-secret')),
    )
    monkeypatch.setattr('graph_service.routers.retrieve.version', lambda _name: '0.29.4')

    result = await get_episodes_for_reconciliation(
        'opr',
        20,
        graphiti,
        settings,
        x_opr_reconciliation_token='reconcile-secret',
    )

    assert result == {
        'reconciliation_protocol': GRAPHITI_RECONCILIATION_PROTOCOL,
        'graphiti_core_version': '0.29.4',
        'episodes': [episode],
    }


@pytest.mark.asyncio
async def test_reconciliation_listing_is_bound_to_opr_group():
    reconciliation_listing = AsyncMock(return_value=[])
    graphiti = cast(
        ZepGraphiti,
        SimpleNamespace(retrieve_episodes_for_reconciliation=reconciliation_listing),
    )
    settings = cast(
        Settings,
        SimpleNamespace(opr_reconciliation_token=SecretStr('reconcile-secret')),
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_episodes_for_reconciliation(
            'other-group',
            20,
            graphiti,
            settings,
            x_opr_reconciliation_token='reconcile-secret',
        )

    assert exc_info.value.status_code == 403
    reconciliation_listing.assert_not_awaited()


@pytest.mark.parametrize(
    'provider',
    [GraphProvider.NEO4J, GraphProvider.FALKORDB, GraphProvider.NEPTUNE],
)
def test_episode_writers_lock_and_reject_deletion_tombstones(provider: GraphProvider):
    single_query = get_episode_node_save_query(provider)
    assert single_query.index('SET n._opr_conditional_delete_lock') < single_query.index(
        'coalesce(n.opr_deleted, false) = false'
    )

    bulk_query = get_episode_node_save_bulk_query(provider)
    assert 'MERGE (existing:Episodic {uuid: episode.uuid})' in bulk_query
    assert 'existing.opr_episode_reservation = true' in bulk_query
    assert bulk_query.index('SET existing._opr_conditional_delete_lock') < bulk_query.index(
        'coalesce(candidate.existing.opr_deleted, false) = false'
    )


def test_kuzu_writer_query_remains_compatible_with_its_explicit_schema():
    query = get_episode_node_save_query(GraphProvider.KUZU)
    assert '_opr_conditional_delete_lock' not in query
    assert 'opr_deleted' not in query
    assert 'REMOVE' not in query


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'operations',
    [Neo4jEpisodeNodeOperations(), FalkorEpisodeNodeOperations(), NeptuneEpisodeNodeOperations()],
)
async def test_ordinary_episode_deletes_preserve_retirement_tombstones(operations):
    executor = SimpleNamespace(execute_query=AsyncMock(return_value=([], None, None)))
    node = SimpleNamespace(uuid=EPISODE_UUID)

    await operations.delete(executor, node)
    query = executor.execute_query.await_args.args[0]
    assert query.index('SET n._opr_conditional_delete_lock') < query.index(
        'coalesce(n.opr_deleted, false) = false'
    )

    executor.execute_query.reset_mock()
    await operations.delete_by_group_id(executor, 'opr')
    query = executor.execute_query.await_args.args[0]
    assert query.index('SET n._opr_conditional_delete_lock') < query.index(
        'coalesce(n.opr_deleted, false) = false'
    )

    executor.execute_query.reset_mock()
    await operations.delete_by_uuids(executor, [EPISODE_UUID])
    query = executor.execute_query.await_args.args[0]
    assert query.index('SET n._opr_conditional_delete_lock') < query.index(
        'coalesce(n.opr_deleted, false) = false'
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'provider',
    [GraphProvider.NEO4J, GraphProvider.FALKORDB, GraphProvider.NEPTUNE],
)
async def test_legacy_node_delete_preserves_retirement_tombstones(provider: GraphProvider):
    driver = SimpleNamespace(
        provider=provider,
        graph_operations_interface=None,
        execute_query=AsyncMock(return_value=([], None, None)),
    )

    node = cast(EpisodicNode, SimpleNamespace(uuid=EPISODE_UUID))
    await EpisodicNode.delete(node, cast(GraphDriver, driver))

    queries = [call.args[0] for call in driver.execute_query.await_args_list]
    assert queries
    assert all(
        query.index('SET n._opr_conditional_delete_lock')
        < query.index('coalesce(n.opr_deleted, false) = false')
        for query in queries
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'operations',
    [
        Neo4jGraphMaintenanceOperations(),
        FalkorGraphMaintenanceOperations(),
        NeptuneGraphMaintenanceOperations(),
    ],
)
async def test_clear_preserves_retirement_tombstones(operations):
    executor = SimpleNamespace(execute_query=AsyncMock(return_value=([], None, None)))

    await operations.clear_data(executor)

    query = executor.execute_query.await_args.args[0]
    assert query.index('SET n._opr_conditional_delete_lock') < query.index(
        'coalesce(n.opr_deleted, false) = false'
    )


@pytest.mark.asyncio
async def test_conditional_delete_locks_compares_then_finalizes_tombstone():
    driver = SimpleNamespace(
        provider=GraphProvider.NEO4J,
        execute_query=AsyncMock(
            side_effect=[
                ([{'uuid': EPISODE_UUID, 'aoss_fenced': False}], None, None),
                ([{'uuid': EPISODE_UUID}], None, None),
            ]
        ),
    )
    service = cast(ZepGraphiti, SimpleNamespace(driver=driver))

    deleted = await ZepGraphiti.delete_episodic_node_if_matches(
        service,
        EPISODE_UUID,
        group_id='opr',
        name='curated:test.md',
        content='stored content',
        source_description='publish',
    )

    assert deleted is True
    assert driver.execute_query.await_count == 2
    initial_call, final_call = driver.execute_query.await_args_list
    query = initial_call.args[0]
    assert 'MATCH (episode:Episodic {uuid: $uuid})' in query
    assert query.index('SET episode._opr_conditional_delete_lock') < query.index(
        'episode.name = $name'
    )
    assert 'REMOVE episode._opr_conditional_delete_lock' in query
    assert 'episode.uuid = $uuid' in query
    assert 'episode.group_id = $group_id' in query
    assert 'episode.name = $name' in query
    assert 'episode.content = $content' in query
    assert 'episode.source_description = $source_description' in query
    assert 'coalesce(episode.opr_deleted, false) AS was_deleted' in query
    assert 'was_deleted = false' in query
    assert 'episode.opr_deleted_identity_digest = $identity_digest' in query
    assert 'DELETE relationship' in query
    assert 'episode.opr_deleted = true' in query
    assert 'episode.opr_aoss_fenced' in query
    assert "episode.group_id = '__opr_deleted__'" not in query
    assert initial_call.kwargs == {
        'uuid': EPISODE_UUID,
        'group_id': 'opr',
        'name': 'curated:test.md',
        'content': 'stored content',
        'source_description': 'publish',
        'identity_digest': _conditional_episode_identity_digest(
            EPISODE_UUID,
            'opr',
            'curated:test.md',
            'stored content',
            'publish',
        ),
    }
    finalize_query = final_call.args[0]
    assert 'episode.opr_deleted_identity_digest = $identity_digest' in finalize_query
    assert "episode.group_id = '__opr_deleted__'" in finalize_query
    assert "episode.content = ''" in finalize_query
    assert final_call.kwargs['entity_edges'] == []


@pytest.mark.asyncio
async def test_conditional_delete_returns_false_on_identity_mismatch():
    driver = SimpleNamespace(
        provider=GraphProvider.NEO4J,
        execute_query=AsyncMock(return_value=([], None, None)),
    )
    service = cast(ZepGraphiti, SimpleNamespace(driver=driver))

    deleted = await ZepGraphiti.delete_episodic_node_if_matches(
        service,
        EPISODE_UUID,
        group_id='opr',
        name='curated:test.md',
        content='changed content',
        source_description='publish',
    )

    assert deleted is False


@pytest.mark.asyncio
async def test_conditional_delete_rejects_malformed_uuid_before_query():
    driver = SimpleNamespace(
        provider=GraphProvider.NEO4J,
        execute_query=AsyncMock(),
    )
    service = cast(ZepGraphiti, SimpleNamespace(driver=driver))

    with pytest.raises(HTTPException) as exc_info:
        await ZepGraphiti.delete_episodic_node_if_matches(
            service,
            'not-a-uuid',
            group_id='opr',
            name='curated:test.md',
            content='stored content',
            source_description='publish',
        )

    assert exc_info.value.status_code == 422
    driver.execute_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_conditional_delete_requires_durable_neptune_search_tombstone():
    driver = SimpleNamespace(
        provider=GraphProvider.NEPTUNE,
        execute_query=AsyncMock(
            return_value=([{'uuid': EPISODE_UUID, 'aoss_fenced': False}], None, None)
        ),
        save_to_aoss=MagicMock(return_value=0),
    )
    service = cast(ZepGraphiti, SimpleNamespace(driver=driver))

    with pytest.raises(HTTPException) as exc_info:
        await ZepGraphiti.delete_episodic_node_if_matches(
            service,
            EPISODE_UUID,
            group_id='opr',
            name='curated:test.md',
            content='stored content',
            source_description='publish',
        )

    assert exc_info.value.status_code == 503
    assert driver.execute_query.await_count == 1
    driver.save_to_aoss.assert_called_once_with(
        'episode_content',
        [
            {
                'uuid': EPISODE_UUID,
                'content': '',
                'source': '',
                'source_description': 'opr_conditional_delete',
                'group_id': '__opr_deleted__',
                '_version': EPISODE_AOSS_TOMBSTONE_VERSION,
            }
        ],
    )


@pytest.mark.asyncio
async def test_conditional_delete_retries_pending_neptune_search_fence_then_scrubs():
    driver = SimpleNamespace(
        provider=GraphProvider.NEPTUNE,
        execute_query=AsyncMock(
            side_effect=[
                ([{'uuid': EPISODE_UUID, 'aoss_fenced': False}], None, None),
                ([{'uuid': EPISODE_UUID}], None, None),
            ]
        ),
        save_to_aoss=MagicMock(return_value=1),
    )
    service = cast(ZepGraphiti, SimpleNamespace(driver=driver))

    assert await ZepGraphiti.delete_episodic_node_if_matches(
        service,
        EPISODE_UUID,
        group_id='opr',
        name='curated:test.md',
        content='stored content',
        source_description='publish',
    )

    assert driver.execute_query.await_count == 2
    assert driver.execute_query.await_args_list[1].kwargs['entity_edges'] == ''


@pytest.mark.asyncio
async def test_conditional_delete_rejects_kuzu_before_query():
    driver = SimpleNamespace(provider=GraphProvider.KUZU, execute_query=AsyncMock())
    service = cast(ZepGraphiti, SimpleNamespace(driver=driver))

    with pytest.raises(HTTPException) as exc_info:
        await ZepGraphiti.delete_episodic_node_if_matches(
            service,
            EPISODE_UUID,
            group_id='opr',
            name='curated:test.md',
            content='stored content',
            source_description='publish',
        )

    assert exc_info.value.status_code == 501
    driver.execute_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_conditional_delete_routes_falkor_to_requested_group_graph():
    group_driver = SimpleNamespace(
        execute_query=AsyncMock(
            side_effect=[
                ([{'uuid': EPISODE_UUID, 'aoss_fenced': False}], None, None),
                ([{'uuid': EPISODE_UUID}], None, None),
            ]
        )
    )
    driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB,
        with_database=MagicMock(return_value=group_driver),
        execute_query=AsyncMock(),
    )
    service = cast(ZepGraphiti, SimpleNamespace(driver=driver))

    assert await ZepGraphiti.delete_episodic_node_if_matches(
        service,
        EPISODE_UUID,
        group_id='opr',
        name='curated:test.md',
        content='stored content',
        source_description='publish',
    )

    driver.with_database.assert_called_once_with('opr')
    driver.execute_query.assert_not_awaited()
    assert group_driver.execute_query.await_count == 2


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
    settings = cast(
        Settings,
        SimpleNamespace(opr_retirement_token=SecretStr('retire-secret')),
    )

    with pytest.raises(HTTPException) as exc_info:
        await delete_episode_if_matches(
            'episode-id',
            request,
            graphiti,
            settings,
            x_opr_retirement_token='retire-secret',
            x_opr_reconciliation_operation=(GRAPHITI_RECONCILIATION_OPERATION_RETIRE_EPISODE),
        )

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
    settings = cast(
        Settings,
        SimpleNamespace(opr_retirement_token=SecretStr('retire-secret')),
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr('graph_service.routers.ingest.version', lambda _name: '0.29.4')
        result = await delete_episode_if_matches(
            'episode-id',
            request,
            graphiti,
            settings,
            x_opr_retirement_token='retire-secret',
            x_opr_reconciliation_operation=(GRAPHITI_RECONCILIATION_OPERATION_RETIRE_EPISODE),
        )

    assert result == {
        'message': 'Episode conditionally deleted',
        'success': True,
        'reconciliation_protocol': GRAPHITI_RECONCILIATION_PROTOCOL,
        'graphiti_core_version': '0.29.4',
    }
    graphiti.delete_episodic_node_if_matches.assert_awaited_once_with(
        'episode-id',
        group_id='opr',
        name='curated:test.md',
        content='stored content',
        source_description='publish',
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('token', 'operation'),
    [
        (None, GRAPHITI_RECONCILIATION_OPERATION_RETIRE_EPISODE),
        ('wrong', GRAPHITI_RECONCILIATION_OPERATION_RETIRE_EPISODE),
        ('retire-secret', None),
        ('retire-secret', 'list_episodes'),
        ('reconcile-secret', GRAPHITI_RECONCILIATION_OPERATION_RETIRE_EPISODE),
    ],
)
async def test_conditional_delete_requires_token_and_exact_operation_scope(
    token,
    operation,
):
    graphiti = MagicMock()
    graphiti.delete_episodic_node_if_matches = AsyncMock(return_value=True)
    request = DeleteEpisodeIfMatchRequest(
        group_id='opr',
        name='curated:test.md',
        content='stored content',
        source_description='publish',
    )
    settings = cast(
        Settings,
        SimpleNamespace(opr_retirement_token=SecretStr('retire-secret')),
    )

    with pytest.raises(HTTPException) as exc_info:
        await delete_episode_if_matches(
            'episode-id',
            request,
            graphiti,
            settings,
            x_opr_retirement_token=token,
            x_opr_reconciliation_operation=operation,
        )

    assert exc_info.value.status_code == 403
    graphiti.delete_episodic_node_if_matches.assert_not_awaited()


@pytest.mark.asyncio
async def test_conditional_delete_is_bound_to_opr_group():
    graphiti = MagicMock()
    graphiti.delete_episodic_node_if_matches = AsyncMock(return_value=True)
    request = DeleteEpisodeIfMatchRequest(
        group_id='other-group',
        name='curated:test.md',
        content='stored content',
        source_description='publish',
    )
    settings = cast(
        Settings,
        SimpleNamespace(opr_retirement_token=SecretStr('retire-secret')),
    )

    with pytest.raises(HTTPException) as exc_info:
        await delete_episode_if_matches(
            'episode-id',
            request,
            graphiti,
            settings,
            x_opr_retirement_token='retire-secret',
            x_opr_reconciliation_operation=(GRAPHITI_RECONCILIATION_OPERATION_RETIRE_EPISODE),
        )

    assert exc_info.value.status_code == 403
    graphiti.delete_episodic_node_if_matches.assert_not_awaited()
