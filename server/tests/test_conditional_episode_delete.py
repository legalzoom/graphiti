import hashlib
import inspect
import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

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
from graphiti_core.nodes import EpisodeType, EpisodicNode
from pydantic import SecretStr
from starlette.responses import JSONResponse

from graph_service.config import Settings
from graph_service.dto import DeleteEpisodeIfMatchRequest
from graph_service.protocol import (
    GRAPHITI_RECONCILIATION_OPERATION_RETIRE_EPISODE,
    GRAPHITI_RECONCILIATION_PROTOCOL,
)
from graph_service.routers.ingest import (
    delete_episode_if_matches,
    get_episode_retirement_status,
)
from graph_service.routers.retrieve import get_episodes, get_episodes_for_reconciliation
from graph_service.zep_graphiti import ZepGraphiti, _conditional_episode_identity_digest

EPISODE_UUID = '11111111-1111-4111-8111-111111111111'
RETIREMENT_REQUEST_ID = '22222222-2222-4222-8222-222222222222'
WRITER_FLEET_EPOCH = 'writer-fleet-epoch-secret-0123456789abcdef'
WRITER_FLEET_EPOCH_SHA256 = hashlib.sha256(WRITER_FLEET_EPOCH.encode()).hexdigest()


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


def test_legacy_episode_listing_cannot_request_privileged_reconciliation_data():
    assert 'include_retired_for_reconciliation' not in inspect.signature(get_episodes).parameters


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
        SimpleNamespace(
            opr_reconciliation_token=SecretStr('reconcile-secret'),
            opr_writer_fleet_epoch=SecretStr(WRITER_FLEET_EPOCH),
        ),
    )
    monkeypatch.setattr('graph_service.routers.retrieve.version', lambda _name: '0.29.4')

    result = await get_episodes_for_reconciliation(
        'opr',
        20,
        graphiti,
        settings,
        x_opr_reconciliation_token='reconcile-secret',
        x_opr_writer_fleet_epoch=WRITER_FLEET_EPOCH,
    )

    assert result == {
        'reconciliation_protocol': GRAPHITI_RECONCILIATION_PROTOCOL,
        'graphiti_core_version': '0.29.4',
        'writer_fleet_epoch_sha256': WRITER_FLEET_EPOCH_SHA256,
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
        SimpleNamespace(
            opr_reconciliation_token=SecretStr('reconcile-secret'),
            opr_writer_fleet_epoch=SecretStr(WRITER_FLEET_EPOCH),
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_episodes_for_reconciliation(
            'other-group',
            20,
            graphiti,
            settings,
            x_opr_reconciliation_token='reconcile-secret',
            x_opr_writer_fleet_epoch=WRITER_FLEET_EPOCH,
        )

    assert exc_info.value.status_code == 403
    reconciliation_listing.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('writer_fleet_epoch', [None, '', 'wrong-epoch'])
async def test_reconciliation_listing_requires_exact_writer_fleet_epoch(
    writer_fleet_epoch: str | None,
):
    reconciliation_listing = AsyncMock(return_value=[])
    graphiti = cast(
        ZepGraphiti,
        SimpleNamespace(retrieve_episodes_for_reconciliation=reconciliation_listing),
    )
    settings = cast(
        Settings,
        SimpleNamespace(
            opr_reconciliation_token=SecretStr('reconcile-secret'),
            opr_writer_fleet_epoch=SecretStr(WRITER_FLEET_EPOCH),
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_episodes_for_reconciliation(
            'opr',
            20,
            graphiti,
            settings,
            x_opr_reconciliation_token='reconcile-secret',
            x_opr_writer_fleet_epoch=writer_fleet_epoch,
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
    assert single_query.index('SET n._opr_conditional_delete_lock') < single_query.index(
        'n.group_id IS NULL OR n.group_id = $group_id'
    )

    bulk_query = get_episode_node_save_bulk_query(provider)
    assert 'MERGE (existing:Episodic {uuid: episode.uuid})' in bulk_query
    assert 'existing.opr_episode_reservation = true' in bulk_query
    assert bulk_query.index('SET existing._opr_conditional_delete_lock') < bulk_query.index(
        'coalesce(candidate.existing.opr_deleted, false) = false'
    )
    assert bulk_query.index('SET existing._opr_conditional_delete_lock') < bulk_query.index(
        'candidate.existing.group_id = candidate.episode.group_id'
    )


def test_kuzu_writer_query_remains_compatible_with_its_explicit_schema():
    query = get_episode_node_save_query(GraphProvider.KUZU)
    assert '_opr_conditional_delete_lock' not in query
    assert 'opr_deleted' not in query
    assert 'REMOVE' not in query
    assert 'n.group_id IS NULL OR n.group_id = $group_id' in query


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
                (
                    [
                        {
                            'outcome': 'retired',
                            'applied': True,
                            'aoss_fenced': False,
                        }
                    ],
                    None,
                    None,
                ),
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
        source=EpisodeType.message.value,
        source_description='publish',
        retirement_request_id=RETIREMENT_REQUEST_ID,
    )

    assert deleted is True
    assert driver.execute_query.await_count == 2
    initial_call, final_call = driver.execute_query.await_args_list
    query = initial_call.args[0]
    assert 'MATCH (episode:Episodic {uuid: $uuid})' in query
    assert query.index('SET episode._opr_conditional_delete_lock') < query.index(
        'episode.name = $name'
    )
    assert 'SET episode._opr_conditional_delete_lock = NULL' in query
    assert 'MERGE (receipt:OPRRetirementReceipt' in query
    assert 'receipt.protocol = $receipt_protocol' in query
    assert "THEN 'not_applied'" in query
    assert 'episode.uuid = $uuid' in query
    assert 'episode.group_id = $group_id' in query
    assert 'episode.name = $name' in query
    assert 'episode.content = $content' in query
    assert 'episode.source = $source' in query
    assert 'episode.source_description = $source_description' in query
    assert 'coalesce(episode.opr_deleted, false) AS was_deleted' in query
    assert 'coalesce(episode.opr_deleted, false) = false' in query
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
        'source': EpisodeType.message.value,
        'source_description': 'publish',
        'identity_digest': _conditional_episode_identity_digest(
            EPISODE_UUID,
            'opr',
            'curated:test.md',
            'stored content',
            EpisodeType.message.value,
            'publish',
        ),
        'retirement_request_id': RETIREMENT_REQUEST_ID,
        'receipt_protocol': GRAPHITI_RECONCILIATION_PROTOCOL,
    }
    finalize_query = final_call.args[0]
    assert 'MATCH (receipt:OPRRetirementReceipt' in finalize_query
    assert 'receipt.protocol = $receipt_protocol' in finalize_query
    assert 'episode.opr_deleted_identity_digest = $identity_digest' in finalize_query
    assert "episode.group_id = '__opr_deleted__'" in finalize_query
    assert "episode.content = ''" in finalize_query
    assert final_call.kwargs['entity_edges'] == []


@pytest.mark.asyncio
async def test_conditional_delete_returns_false_on_identity_mismatch():
    driver = SimpleNamespace(
        provider=GraphProvider.NEO4J,
        execute_query=AsyncMock(
            return_value=(
                [
                    {
                        'outcome': 'not_applied',
                        'applied': False,
                        'aoss_fenced': False,
                    }
                ],
                None,
                None,
            )
        ),
    )
    service = cast(ZepGraphiti, SimpleNamespace(driver=driver))

    deleted = await ZepGraphiti.delete_episodic_node_if_matches(
        service,
        EPISODE_UUID,
        group_id='opr',
        name='curated:test.md',
        content='changed content',
        source=EpisodeType.message.value,
        source_description='publish',
        retirement_request_id=RETIREMENT_REQUEST_ID,
    )

    assert deleted is False
    query = driver.execute_query.await_args.args[0]
    assert 'MERGE (receipt:OPRRetirementReceipt' in query
    assert "WHEN receipt.outcome = 'pending' THEN 'not_applied'" in query


def test_conditional_delete_identity_digest_binds_source_type():
    message_digest = _conditional_episode_identity_digest(
        EPISODE_UUID,
        'opr',
        'curated:test.md',
        'stored content',
        EpisodeType.message.value,
        'publish',
    )
    text_digest = _conditional_episode_identity_digest(
        EPISODE_UUID,
        'opr',
        'curated:test.md',
        'stored content',
        EpisodeType.text.value,
        'publish',
    )

    assert message_digest != text_digest


def test_conditional_delete_identity_digest_rejects_legacy_domain():
    content_digest = hashlib.sha256(b'stored content').hexdigest()
    canonical = json.dumps(
        [
            EPISODE_UUID,
            'opr',
            'curated:test.md',
            content_digest,
            EpisodeType.message.value,
            'publish',
        ],
        ensure_ascii=False,
        separators=(',', ':'),
    )
    legacy_digest = hashlib.sha256(
        f'opr:conditional-episode-delete:v1\0{canonical}'.encode()
    ).hexdigest()

    assert (
        _conditional_episode_identity_digest(
            EPISODE_UUID,
            'opr',
            'curated:test.md',
            'stored content',
            EpisodeType.message.value,
            'publish',
        )
        != legacy_digest
    )


def test_conditional_delete_request_allows_exact_empty_source_description():
    request = DeleteEpisodeIfMatchRequest(
        group_id='opr',
        name='curated:test.md',
        content='stored content',
        source=EpisodeType.message,
        source_description='',
        retirement_request_id=UUID(RETIREMENT_REQUEST_ID),
    )

    assert request.source_description == ''


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
            source=EpisodeType.message.value,
            source_description='publish',
            retirement_request_id=RETIREMENT_REQUEST_ID,
        )

    assert exc_info.value.status_code == 422
    driver.execute_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_conditional_delete_requires_durable_neptune_search_tombstone():
    driver = SimpleNamespace(
        provider=GraphProvider.NEPTUNE,
        execute_query=AsyncMock(
            side_effect=[
                ([{'bound': True, 'outcome': 'pending'}], None, None),
                (
                    [{'outcome': 'retired', 'applied': True, 'aoss_fenced': False}],
                    None,
                    None,
                ),
            ]
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
            source=EpisodeType.message.value,
            source_description='publish',
            retirement_request_id=RETIREMENT_REQUEST_ID,
        )

    assert exc_info.value.status_code == 503
    assert driver.execute_query.await_count == 2
    admission_query = driver.execute_query.await_args_list[0].args[0]
    mutation_query = driver.execute_query.await_args_list[1].args[0]
    assert 'FOREACH' not in admission_query + mutation_query
    assert '{`~id`: $receipt_node_id}' in admission_query
    assert driver.execute_query.await_args_list[0].kwargs['receipt_node_id'] == (
        f'opr-retirement-receipt:{RETIREMENT_REQUEST_ID}'
    )
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
                ([{'bound': True, 'outcome': 'pending'}], None, None),
                (
                    [
                        {
                            'outcome': 'retired',
                            'applied': True,
                            'aoss_fenced': False,
                        }
                    ],
                    None,
                    None,
                ),
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
        source=EpisodeType.message.value,
        source_description='publish',
        retirement_request_id=RETIREMENT_REQUEST_ID,
    )

    assert driver.execute_query.await_count == 3
    assert driver.execute_query.await_args_list[2].kwargs['entity_edges'] == ''


@pytest.mark.asyncio
async def test_neptune_persists_not_applied_receipt_without_foreach():
    driver = SimpleNamespace(
        provider=GraphProvider.NEPTUNE,
        execute_query=AsyncMock(
            side_effect=[
                ([{'bound': True, 'outcome': 'pending'}], None, None),
                ([], None, None),
                ([{'outcome': 'not_applied'}], None, None),
            ]
        ),
    )
    service = cast(ZepGraphiti, SimpleNamespace(driver=driver))

    assert (
        await ZepGraphiti.delete_episodic_node_if_matches(
            service,
            EPISODE_UUID,
            group_id='opr',
            name='curated:test.md',
            content='changed content',
            source=EpisodeType.message.value,
            source_description='publish',
            retirement_request_id=RETIREMENT_REQUEST_ID,
        )
        is False
    )
    assert driver.execute_query.await_count == 3
    assert all('FOREACH' not in call.args[0] for call in driver.execute_query.await_args_list)
    assert all(
        call.kwargs['receipt_node_id'] == f'opr-retirement-receipt:{RETIREMENT_REQUEST_ID}'
        for call in driver.execute_query.await_args_list
    )


@pytest.mark.asyncio
async def test_neptune_receipt_binding_conflict_is_not_not_applied():
    driver = SimpleNamespace(
        provider=GraphProvider.NEPTUNE,
        execute_query=AsyncMock(
            return_value=([{'bound': False, 'outcome': 'pending'}], None, None)
        ),
    )
    service = cast(ZepGraphiti, SimpleNamespace(driver=driver))

    assert (
        await ZepGraphiti.delete_episodic_node_if_matches(
            service,
            EPISODE_UUID,
            group_id='opr',
            name='curated:test.md',
            content='stored content',
            source=EpisodeType.message.value,
            source_description='publish',
            retirement_request_id=RETIREMENT_REQUEST_ID,
        )
        is None
    )
    assert driver.execute_query.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', [GraphProvider.NEO4J, GraphProvider.NEPTUNE])
@pytest.mark.parametrize(
    'legacy_protocol',
    [None, 'opr.graphiti.reconciliation/v3', 'opr.graphiti.reconciliation/v4'],
    ids=['missing-protocol', 'v3-protocol', 'v4-protocol'],
)
async def test_conditional_delete_rejects_legacy_receipt_protocol(
    provider: GraphProvider,
    legacy_protocol: str | None,
):
    async def execute_query(_query: str, **params):
        assert legacy_protocol != params['receipt_protocol']
        existing_receipt_result = (
            [{'bound': False, 'outcome': 'retired'}] if provider == GraphProvider.NEPTUNE else []
        )
        return existing_receipt_result, None, None

    driver = SimpleNamespace(
        provider=provider,
        execute_query=AsyncMock(side_effect=execute_query),
    )
    service = cast(ZepGraphiti, SimpleNamespace(driver=driver))

    assert (
        await ZepGraphiti.delete_episodic_node_if_matches(
            service,
            EPISODE_UUID,
            group_id='opr',
            name='curated:test.md',
            content='stored content',
            source=EpisodeType.message.value,
            source_description='publish',
            retirement_request_id=RETIREMENT_REQUEST_ID,
        )
        is None
    )

    assert driver.execute_query.await_count == 1
    query = driver.execute_query.await_args.args[0]
    assert 'receipt.protocol = $receipt_protocol' in query
    assert driver.execute_query.await_args.kwargs['receipt_protocol'] == (
        GRAPHITI_RECONCILIATION_PROTOCOL
    )


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
            source=EpisodeType.message.value,
            source_description='publish',
            retirement_request_id=RETIREMENT_REQUEST_ID,
        )

    assert exc_info.value.status_code == 501
    driver.execute_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_conditional_delete_rejects_falkor_without_receipt_uniqueness():
    driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB,
        execute_query=AsyncMock(),
    )
    service = cast(ZepGraphiti, SimpleNamespace(driver=driver))

    with pytest.raises(HTTPException) as exc_info:
        await ZepGraphiti.delete_episodic_node_if_matches(
            service,
            EPISODE_UUID,
            group_id='opr',
            name='curated:test.md',
            content='stored content',
            source=EpisodeType.message.value,
            source_description='publish',
            retirement_request_id=RETIREMENT_REQUEST_ID,
        )

    assert exc_info.value.status_code == 501
    driver.execute_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_conditional_delete_route_fails_precondition_without_success_receipt():
    graphiti = MagicMock()
    graphiti.delete_episodic_node_if_matches = AsyncMock(return_value=False)
    request = DeleteEpisodeIfMatchRequest(
        group_id='opr',
        name='curated:test.md',
        content='stored content',
        source=EpisodeType.message,
        source_description='publish',
        retirement_request_id=UUID(RETIREMENT_REQUEST_ID),
    )
    settings = cast(
        Settings,
        SimpleNamespace(
            opr_retirement_token=SecretStr('retire-secret'),
            opr_writer_fleet_epoch=SecretStr(WRITER_FLEET_EPOCH),
        ),
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr('graph_service.routers.ingest.version', lambda _name: '0.29.4')
        response = cast(
            JSONResponse,
            await delete_episode_if_matches(
                'episode-id',
                request,
                graphiti,
                settings,
                x_opr_retirement_token='retire-secret',
                x_opr_writer_fleet_epoch=WRITER_FLEET_EPOCH,
                x_opr_reconciliation_operation=(GRAPHITI_RECONCILIATION_OPERATION_RETIRE_EPISODE),
            ),
        )

    assert response.status_code == 412
    assert json.loads(bytes(response.body)) == {
        'message': 'Episode identity precondition failed',
        'success': False,
        'outcome': 'not_applied',
        'reconciliation_protocol': GRAPHITI_RECONCILIATION_PROTOCOL,
        'graphiti_core_version': '0.29.4',
        'writer_fleet_epoch_sha256': WRITER_FLEET_EPOCH_SHA256,
        'retirement_request_id': RETIREMENT_REQUEST_ID,
    }


@pytest.mark.asyncio
async def test_conditional_delete_route_rejects_receipt_binding_conflict():
    graphiti = MagicMock()
    graphiti.delete_episodic_node_if_matches = AsyncMock(return_value=None)
    request = DeleteEpisodeIfMatchRequest(
        group_id='opr',
        name='curated:test.md',
        content='stored content',
        source=EpisodeType.message,
        source_description='publish',
        retirement_request_id=UUID(RETIREMENT_REQUEST_ID),
    )
    settings = cast(
        Settings,
        SimpleNamespace(
            opr_retirement_token=SecretStr('retire-secret'),
            opr_writer_fleet_epoch=SecretStr(WRITER_FLEET_EPOCH),
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        await delete_episode_if_matches(
            EPISODE_UUID,
            request,
            graphiti,
            settings,
            x_opr_retirement_token='retire-secret',
            x_opr_writer_fleet_epoch=WRITER_FLEET_EPOCH,
            x_opr_reconciliation_operation=(GRAPHITI_RECONCILIATION_OPERATION_RETIRE_EPISODE),
        )

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_conditional_delete_route_returns_success_only_after_atomic_match():
    graphiti = MagicMock()
    graphiti.delete_episodic_node_if_matches = AsyncMock(return_value=True)
    request = DeleteEpisodeIfMatchRequest(
        group_id='opr',
        name='curated:test.md',
        content='stored content',
        source=EpisodeType.message,
        source_description='publish',
        retirement_request_id=UUID(RETIREMENT_REQUEST_ID),
    )
    settings = cast(
        Settings,
        SimpleNamespace(
            opr_retirement_token=SecretStr('retire-secret'),
            opr_writer_fleet_epoch=SecretStr(WRITER_FLEET_EPOCH),
        ),
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr('graph_service.routers.ingest.version', lambda _name: '0.29.4')
        result = await delete_episode_if_matches(
            'episode-id',
            request,
            graphiti,
            settings,
            x_opr_retirement_token='retire-secret',
            x_opr_writer_fleet_epoch=WRITER_FLEET_EPOCH,
            x_opr_reconciliation_operation=(GRAPHITI_RECONCILIATION_OPERATION_RETIRE_EPISODE),
        )

    assert result == {
        'message': 'Episode conditionally deleted',
        'success': True,
        'outcome': 'retired',
        'reconciliation_protocol': GRAPHITI_RECONCILIATION_PROTOCOL,
        'graphiti_core_version': '0.29.4',
        'writer_fleet_epoch_sha256': WRITER_FLEET_EPOCH_SHA256,
        'retirement_request_id': RETIREMENT_REQUEST_ID,
    }
    graphiti.delete_episodic_node_if_matches.assert_awaited_once_with(
        'episode-id',
        group_id='opr',
        name='curated:test.md',
        content='stored content',
        source=EpisodeType.message.value,
        source_description='publish',
        retirement_request_id=RETIREMENT_REQUEST_ID,
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
        source=EpisodeType.message,
        source_description='publish',
        retirement_request_id=UUID(RETIREMENT_REQUEST_ID),
    )
    settings = cast(
        Settings,
        SimpleNamespace(
            opr_retirement_token=SecretStr('retire-secret'),
            opr_writer_fleet_epoch=SecretStr(WRITER_FLEET_EPOCH),
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        await delete_episode_if_matches(
            'episode-id',
            request,
            graphiti,
            settings,
            x_opr_retirement_token=token,
            x_opr_writer_fleet_epoch=WRITER_FLEET_EPOCH,
            x_opr_reconciliation_operation=operation,
        )

    assert exc_info.value.status_code == 403
    graphiti.delete_episodic_node_if_matches.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('writer_fleet_epoch', [None, '', 'wrong-epoch'])
async def test_conditional_delete_requires_exact_writer_fleet_epoch(
    writer_fleet_epoch: str | None,
):
    graphiti = MagicMock()
    graphiti.delete_episodic_node_if_matches = AsyncMock(return_value=True)
    request = DeleteEpisodeIfMatchRequest(
        group_id='opr',
        name='curated:test.md',
        content='stored content',
        source=EpisodeType.message,
        source_description='publish',
        retirement_request_id=UUID(RETIREMENT_REQUEST_ID),
    )
    settings = cast(
        Settings,
        SimpleNamespace(
            opr_retirement_token=SecretStr('retire-secret'),
            opr_writer_fleet_epoch=SecretStr(WRITER_FLEET_EPOCH),
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        await delete_episode_if_matches(
            EPISODE_UUID,
            request,
            graphiti,
            settings,
            x_opr_retirement_token='retire-secret',
            x_opr_writer_fleet_epoch=writer_fleet_epoch,
            x_opr_reconciliation_operation=(GRAPHITI_RECONCILIATION_OPERATION_RETIRE_EPISODE),
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
        source=EpisodeType.message,
        source_description='publish',
        retirement_request_id=UUID(RETIREMENT_REQUEST_ID),
    )
    settings = cast(
        Settings,
        SimpleNamespace(
            opr_retirement_token=SecretStr('retire-secret'),
            opr_writer_fleet_epoch=SecretStr(WRITER_FLEET_EPOCH),
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        await delete_episode_if_matches(
            'episode-id',
            request,
            graphiti,
            settings,
            x_opr_retirement_token='retire-secret',
            x_opr_writer_fleet_epoch=WRITER_FLEET_EPOCH,
            x_opr_reconciliation_operation=(GRAPHITI_RECONCILIATION_OPERATION_RETIRE_EPISODE),
        )

    assert exc_info.value.status_code == 403
    graphiti.delete_episodic_node_if_matches.assert_not_awaited()


@pytest.mark.asyncio
async def test_retirement_status_is_bound_to_matching_durable_request():
    driver = SimpleNamespace(
        provider=GraphProvider.NEO4J,
        execute_query=AsyncMock(
            return_value=([{'outcome': 'retired', 'durable': True}], None, None)
        ),
    )
    service = cast(ZepGraphiti, SimpleNamespace(driver=driver))

    assert (
        await ZepGraphiti.episode_retirement_outcome(
            service,
            EPISODE_UUID,
            group_id='opr',
            retirement_request_id=RETIREMENT_REQUEST_ID,
        )
        == 'retired'
    )

    query = driver.execute_query.await_args.args[0]
    assert query.index('SET episode._opr_conditional_delete_lock') < query.index(
        'episode.opr_retirement_request_id = $retirement_request_id'
    )
    assert 'coalesce(episode.opr_aoss_fenced, false) = true' in query
    assert driver.execute_query.await_args.kwargs == {
        'uuid': EPISODE_UUID,
        'group_id': 'opr',
        'retirement_request_id': RETIREMENT_REQUEST_ID,
        'receipt_protocol': GRAPHITI_RECONCILIATION_PROTOCOL,
    }


@pytest.mark.asyncio
async def test_neptune_retirement_status_uses_unique_receipt_id_without_foreach():
    driver = SimpleNamespace(
        provider=GraphProvider.NEPTUNE,
        execute_query=AsyncMock(
            return_value=(
                [{'bound': True, 'outcome': 'retired', 'durable': True}],
                None,
                None,
            ),
        ),
    )
    service = cast(ZepGraphiti, SimpleNamespace(driver=driver))

    assert (
        await ZepGraphiti.episode_retirement_outcome(
            service,
            EPISODE_UUID,
            group_id='opr',
            retirement_request_id=RETIREMENT_REQUEST_ID,
        )
        == 'retired'
    )
    assert driver.execute_query.await_count == 1
    query = driver.execute_query.await_args.args[0]
    assert 'FOREACH' not in query
    assert (
        driver.execute_query.await_args.kwargs['receipt_node_id']
        == f'opr-retirement-receipt:{RETIREMENT_REQUEST_ID}'
    )
    assert query.index('SET receipt._opr_conditional_delete_lock') < query.index(
        'OPTIONAL MATCH (episode:Episodic {uuid: $uuid})'
    )
    assert query.index('OPTIONAL MATCH (episode:Episodic {uuid: $uuid})') < query.index(
        'SET decision_lock._opr_conditional_delete_lock'
    )
    assert query.index('SET decision_lock._opr_conditional_delete_lock') < query.index(
        'SET receipt.outcome = CASE'
    )


@pytest.mark.asyncio
async def test_neptune_retirement_status_leaves_pending_when_episode_is_present():
    driver = SimpleNamespace(
        provider=GraphProvider.NEPTUNE,
        execute_query=AsyncMock(
            return_value=(
                [{'bound': True, 'outcome': 'pending', 'durable': False}],
                None,
                None,
            )
        ),
    )
    service = cast(ZepGraphiti, SimpleNamespace(driver=driver))

    assert (
        await ZepGraphiti.episode_retirement_outcome(
            service,
            EPISODE_UUID,
            group_id='opr',
            retirement_request_id=RETIREMENT_REQUEST_ID,
        )
        is None
    )
    query = driver.execute_query.await_args.args[0]
    assert "receipt.outcome = 'pending' AND episode IS NULL" in query


@pytest.mark.asyncio
async def test_neptune_retirement_status_cancels_pending_only_when_episode_is_absent():
    driver = SimpleNamespace(
        provider=GraphProvider.NEPTUNE,
        execute_query=AsyncMock(
            return_value=(
                [{'bound': True, 'outcome': 'not_applied', 'durable': False}],
                None,
                None,
            )
        ),
    )
    service = cast(ZepGraphiti, SimpleNamespace(driver=driver))

    assert (
        await ZepGraphiti.episode_retirement_outcome(
            service,
            EPISODE_UUID,
            group_id='opr',
            retirement_request_id=RETIREMENT_REQUEST_ID,
        )
        == 'not_applied'
    )
    query = driver.execute_query.await_args.args[0]
    assert "WHEN receipt.outcome = 'pending' AND episode IS NULL THEN 'not_applied'" in query
    assert query.index('SET receipt._opr_conditional_delete_lock') < query.index(
        "WHEN receipt.outcome = 'pending' AND episode IS NULL THEN 'not_applied'"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', [GraphProvider.NEO4J, GraphProvider.NEPTUNE])
@pytest.mark.parametrize(
    'legacy_protocol',
    [None, 'opr.graphiti.reconciliation/v3', 'opr.graphiti.reconciliation/v4'],
    ids=['missing-protocol', 'v3-protocol', 'v4-protocol'],
)
async def test_retirement_status_rejects_legacy_receipt_protocol_before_resolution(
    provider: GraphProvider,
    legacy_protocol: str | None,
):
    async def execute_query(_query: str, **params):
        assert legacy_protocol != params['receipt_protocol']
        return [], None, None

    driver = SimpleNamespace(
        provider=provider,
        execute_query=AsyncMock(side_effect=execute_query),
    )
    service = cast(ZepGraphiti, SimpleNamespace(driver=driver))

    assert (
        await ZepGraphiti.episode_retirement_outcome(
            service,
            EPISODE_UUID,
            group_id='opr',
            retirement_request_id=RETIREMENT_REQUEST_ID,
        )
        is None
    )

    assert driver.execute_query.await_count == 1
    query = driver.execute_query.await_args.args[0]
    assert 'receipt.protocol = $receipt_protocol' in query
    first_effect_after_binding = 'OPTIONAL MATCH (episode:Episodic {uuid: $uuid})'
    assert query.index('receipt.protocol = $receipt_protocol') < query.index(
        first_effect_after_binding
    )
    assert driver.execute_query.await_args.kwargs['receipt_protocol'] == (
        GRAPHITI_RECONCILIATION_PROTOCOL
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', [GraphProvider.KUZU, GraphProvider.FALKORDB])
async def test_retirement_status_rejects_backend_without_receipt_uniqueness(
    provider: GraphProvider,
):
    driver = SimpleNamespace(provider=provider, execute_query=AsyncMock())
    service = cast(ZepGraphiti, SimpleNamespace(driver=driver))

    with pytest.raises(HTTPException) as exc_info:
        await ZepGraphiti.episode_retirement_outcome(
            service,
            EPISODE_UUID,
            group_id='opr',
            retirement_request_id=RETIREMENT_REQUEST_ID,
        )

    assert exc_info.value.status_code == 501
    driver.execute_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_retirement_status_route_returns_request_bound_receipt():
    graphiti = MagicMock()
    graphiti.episode_retirement_outcome = AsyncMock(return_value='retired')
    settings = cast(
        Settings,
        SimpleNamespace(
            opr_retirement_token=SecretStr('retire-secret'),
            opr_writer_fleet_epoch=SecretStr(WRITER_FLEET_EPOCH),
        ),
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr('graph_service.routers.ingest.version', lambda _name: '0.29.4')
        result = await get_episode_retirement_status(
            EPISODE_UUID,
            RETIREMENT_REQUEST_ID,
            'opr',
            graphiti,
            settings,
            x_opr_retirement_token='retire-secret',
            x_opr_writer_fleet_epoch=WRITER_FLEET_EPOCH,
            x_opr_reconciliation_operation=(GRAPHITI_RECONCILIATION_OPERATION_RETIRE_EPISODE),
        )

    assert result == {
        'message': 'Episode retirement outcome is durable',
        'success': True,
        'outcome': 'retired',
        'reconciliation_protocol': GRAPHITI_RECONCILIATION_PROTOCOL,
        'graphiti_core_version': '0.29.4',
        'writer_fleet_epoch_sha256': WRITER_FLEET_EPOCH_SHA256,
        'retirement_request_id': RETIREMENT_REQUEST_ID,
    }
    graphiti.episode_retirement_outcome.assert_awaited_once_with(
        EPISODE_UUID,
        group_id='opr',
        retirement_request_id=RETIREMENT_REQUEST_ID,
    )


@pytest.mark.asyncio
async def test_retirement_status_route_returns_durable_not_applied_receipt():
    graphiti = MagicMock()
    graphiti.episode_retirement_outcome = AsyncMock(return_value='not_applied')
    settings = cast(
        Settings,
        SimpleNamespace(
            opr_retirement_token=SecretStr('retire-secret'),
            opr_writer_fleet_epoch=SecretStr(WRITER_FLEET_EPOCH),
        ),
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr('graph_service.routers.ingest.version', lambda _name: '0.29.4')
        result = await get_episode_retirement_status(
            EPISODE_UUID,
            RETIREMENT_REQUEST_ID,
            'opr',
            graphiti,
            settings,
            x_opr_retirement_token='retire-secret',
            x_opr_writer_fleet_epoch=WRITER_FLEET_EPOCH,
            x_opr_reconciliation_operation=(GRAPHITI_RECONCILIATION_OPERATION_RETIRE_EPISODE),
        )

    assert result['success'] is False
    assert result['outcome'] == 'not_applied'
    assert result['writer_fleet_epoch_sha256'] == WRITER_FLEET_EPOCH_SHA256
    assert result['retirement_request_id'] == RETIREMENT_REQUEST_ID
