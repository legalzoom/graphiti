from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException, Request
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.errors import NodeGroupMismatchError
from pydantic import SecretStr

from graph_service.auth import GraphitiAuthorizer
from graph_service.config import OprAuthMode, Settings
from graph_service.dto import (
    AddEntityNodeRequest,
    AddMessagesRequest,
    GetMemoryRequest,
    Message,
    Result,
    SearchQuery,
)
from graph_service.protocol import (
    bearer_token_matches,
    is_http_token68,
    reconciliation_token_matches,
)
from graph_service.routers import ingest
from graph_service.routers.ingest import (
    add_entity_node,
    add_messages,
    clear,
    delete_entity_edge,
    delete_episode,
    delete_group,
)
from graph_service.routers.retrieve import get_episodes, get_memory, search
from graph_service.zep_graphiti import GRAPHITI_CLIENT_STATE_ATTR, ZepGraphiti


def _http_request(graphiti: ZepGraphiti | None = None) -> Request:
    """A stand-in for the Starlette request `add_messages` now takes.

    `add_messages` needs the application only to resolve the shared Graphiti
    client from app state when a queued job actually runs. These tests never
    run a queued job, so an empty state is enough unless a client is passed.
    """
    state = SimpleNamespace()
    if graphiti is not None:
        setattr(state, GRAPHITI_CLIENT_STATE_ATTR, graphiti)
    return cast(Request, SimpleNamespace(app=SimpleNamespace(state=state)))


def _settings(
    *,
    opr_read_token: str = 'opr-read-secret',
    opr_write_token: str = 'opr-write-secret',
    admin_token: str = 'admin-secret',
    reconciliation_token: str = 'reconcile-secret',
    retirement_token: str = 'retire-secret',
    writer_fleet_epoch: str = 'writer-fleet-epoch-secret-0123456789abcdef',
    clear_enabled: bool = False,
) -> Settings:
    return cast(
        Settings,
        SimpleNamespace(
            opr_read_token=SecretStr(opr_read_token),
            opr_write_token=SecretStr(opr_write_token),
            graphiti_admin_token=SecretStr(admin_token),
            opr_reconciliation_token=SecretStr(reconciliation_token),
            opr_retirement_token=SecretStr(retirement_token),
            opr_writer_fleet_epoch=SecretStr(writer_fleet_epoch),
            graphiti_admin_clear_enabled=clear_enabled,
            opr_auth_mode=OprAuthMode.STATIC,
        ),
    )


def _authorizer(settings: Settings | None = None) -> GraphitiAuthorizer:
    return GraphitiAuthorizer(settings or _settings())


def _bearer(token: str) -> str:
    return f'Bearer {token}'


def _required_auth_values(**overrides) -> dict:
    values = {
        'openai_api_key': 'test',
        'ingest_queue_maxsize': 1000,
        'opr_auth_required': True,
        'opr_read_token': 'read-' + ('a' * 32),
        'opr_write_token': 'write-' + ('b' * 32),
        'opr_reconciliation_token': 'reconcile-' + ('c' * 32),
        'opr_retirement_token': 'retire-' + ('d' * 32),
        'opr_writer_fleet_epoch': 'epoch-' + ('e' * 32),
        'graphiti_admin_token': 'admin-' + ('f' * 32),
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    ('left', 'right'),
    [
        ('opr_read_token', 'opr_write_token'),
        ('opr_read_token', 'opr_reconciliation_token'),
        ('opr_read_token', 'opr_retirement_token'),
        ('opr_read_token', 'graphiti_admin_token'),
        ('opr_read_token', 'opr_writer_fleet_epoch'),
        ('opr_write_token', 'opr_reconciliation_token'),
        ('opr_write_token', 'opr_retirement_token'),
        ('opr_write_token', 'graphiti_admin_token'),
        ('opr_write_token', 'opr_writer_fleet_epoch'),
        ('opr_reconciliation_token', 'opr_retirement_token'),
        ('opr_reconciliation_token', 'graphiti_admin_token'),
        ('opr_reconciliation_token', 'opr_writer_fleet_epoch'),
        ('opr_retirement_token', 'graphiti_admin_token'),
        ('opr_retirement_token', 'opr_writer_fleet_epoch'),
        ('graphiti_admin_token', 'opr_writer_fleet_epoch'),
    ],
)
def test_all_configured_privileged_tokens_must_be_distinct(left: str, right: str):
    sentinel = 'same-secret-that-must-not-appear'
    values = {
        'openai_api_key': 'test',
        'ingest_queue_maxsize': 1000,
        left: sentinel,
        right: sentinel,
    }

    with pytest.raises(ValueError, match='privileged tokens must be distinct') as exc_info:
        Settings.model_validate(values)

    assert sentinel not in str(exc_info.value)


def test_writer_fleet_epoch_requires_32_byte_minimum_without_leaking_input():
    sentinel = 'short-fleet-epoch-secret'
    with pytest.raises(ValueError, match='at least 32 bytes') as exc_info:
        Settings.model_validate(
            {
                'openai_api_key': 'test',
                'ingest_queue_maxsize': 1000,
                'opr_writer_fleet_epoch': sentinel,
            }
        )
    assert sentinel not in str(exc_info.value)


def test_non_opr_deployment_does_not_require_opr_credentials():
    settings = Settings.model_validate(
        {
            'openai_api_key': 'test',
            'ingest_queue_maxsize': 1000,
        }
    )

    assert settings.opr_auth_required is False
    assert settings.opr_read_token.get_secret_value() == ''
    assert settings.opr_write_token.get_secret_value() == ''


@pytest.mark.parametrize(
    'missing_field',
    [
        'opr_read_token',
        'opr_write_token',
        'opr_reconciliation_token',
        'opr_retirement_token',
        'opr_writer_fleet_epoch',
        'graphiti_admin_token',
    ],
)
def test_required_opr_auth_rejects_each_missing_privileged_value(missing_field: str):
    values = _required_auth_values()
    values[missing_field] = ''

    with pytest.raises(ValueError, match='requires non-empty privileged values'):
        Settings.model_validate(values)


@pytest.mark.parametrize(
    'short_field',
    [
        'opr_read_token',
        'opr_write_token',
        'opr_reconciliation_token',
        'opr_retirement_token',
        'opr_writer_fleet_epoch',
        'graphiti_admin_token',
    ],
)
def test_required_opr_auth_rejects_each_short_privileged_value_without_leaking_it(
    short_field: str,
):
    sentinel = 'short-secret'
    values = _required_auth_values(**{short_field: sentinel})

    with pytest.raises(ValueError, match='at least 32 bytes') as exc_info:
        Settings.model_validate(values)

    assert sentinel not in str(exc_info.value)


def test_required_opr_auth_accepts_six_distinct_http_safe_values():
    settings = Settings.model_validate(_required_auth_values())

    assert settings.opr_auth_required is True


@pytest.mark.parametrize(
    'invalid_value',
    [
        'x' * 16 + ' ' + 'y' * 16,
        '\u00e9' * 16,
        'x' * 32 + '!',
        'x' * 32 + '\n',
    ],
)
def test_required_opr_auth_rejects_values_that_cannot_round_trip_in_http_headers(
    invalid_value: str,
):
    with pytest.raises(ValueError, match='HTTP token68-compatible ASCII'):
        Settings.model_validate(_required_auth_values(opr_read_token=invalid_value))


@pytest.mark.parametrize(
    'field',
    [
        'opr_read_token',
        'opr_write_token',
        'opr_reconciliation_token',
        'opr_retirement_token',
        'opr_writer_fleet_epoch',
        'graphiti_admin_token',
    ],
)
def test_required_opr_auth_rejects_internal_whitespace_for_every_credential(field: str):
    invalid_value = 'x' * 16 + ' ' + 'y' * 16

    with pytest.raises(ValueError, match='HTTP token68-compatible ASCII') as exc_info:
        Settings.model_validate(_required_auth_values(**{field: invalid_value}))

    assert invalid_value not in str(exc_info.value)


def test_required_opr_auth_credentials_round_trip_through_protocol_matchers():
    settings = Settings.model_validate(_required_auth_values())
    read = settings.opr_read_token.get_secret_value()
    write = settings.opr_write_token.get_secret_value()
    admin = settings.graphiti_admin_token.get_secret_value()
    reconciliation = settings.opr_reconciliation_token.get_secret_value()
    retirement = settings.opr_retirement_token.get_secret_value()
    fleet_epoch = settings.opr_writer_fleet_epoch.get_secret_value()

    assert all(
        is_http_token68(value)
        for value in (read, write, admin, reconciliation, retirement, fleet_epoch)
    )
    assert bearer_token_matches(read, f'Bearer {read}')
    assert bearer_token_matches(write, f'Bearer {write}')
    assert bearer_token_matches(admin, f'Bearer {admin}')
    assert reconciliation_token_matches(reconciliation, reconciliation)
    assert reconciliation_token_matches(retirement, retirement)
    assert reconciliation_token_matches(fleet_epoch, fleet_epoch)


def test_required_opr_auth_rejects_equal_ordinary_credentials_without_leaking_them():
    sentinel = 'same-ordinary-secret-' + ('x' * 32)

    with pytest.raises(ValueError, match='privileged tokens must be distinct') as exc_info:
        Settings.model_validate(
            _required_auth_values(opr_read_token=sentinel, opr_write_token=sentinel)
        )

    assert sentinel not in str(exc_info.value)


@pytest.mark.parametrize(
    'authorization',
    [None, '', 'secret', 'Basic secret', 'Bearer', 'Bearer ', 'Bearer  secret', 'Bearer secret '],
)
def test_bearer_parser_rejects_missing_raw_or_malformed_credentials(
    authorization: str | None,
):
    assert bearer_token_matches('secret', authorization) is False


def test_bearer_parser_accepts_only_the_exact_configured_secret():
    assert bearer_token_matches('secret', 'bearer secret') is True
    assert bearer_token_matches('secret', 'Bearer secret') is True
    assert bearer_token_matches('secret', 'Bearer wrong') is False
    assert bearer_token_matches('', 'Bearer secret') is False


def test_token_comparisons_fail_closed_for_non_ascii_bearer_secrets_without_raising():
    secret = '\u00e9' * 16

    assert bearer_token_matches(secret, f'Bearer {secret}') is False
    assert bearer_token_matches(secret, f'Bearer {secret}x') is False
    assert reconciliation_token_matches(secret, secret) is True
    assert reconciliation_token_matches(secret, secret + 'x') is False


@pytest.mark.asyncio
async def test_opr_message_write_rejects_missing_or_read_bearer_before_enqueue(monkeypatch):
    monkeypatch.setattr(
        ingest,
        'async_worker',
        SimpleNamespace(accepting=True, draining=False, depth=0, capacity=1000),
    )
    request = AddMessagesRequest(group_id='opr', messages=[])
    graphiti = cast(ZepGraphiti, SimpleNamespace())
    settings = _settings()

    with pytest.raises(HTTPException) as exc_info:
        await add_messages(
            request, _http_request(graphiti), graphiti, settings, _authorizer(settings)
        )
    assert exc_info.value.status_code == 403

    with pytest.raises(HTTPException) as exc_info:
        await add_messages(
            request,
            _http_request(graphiti),
            graphiti,
            settings,
            _authorizer(settings),
            authorization=_bearer('opr-read-secret'),
        )
    assert exc_info.value.status_code == 403

    result = await add_messages(
        request,
        _http_request(graphiti),
        graphiti,
        settings,
        _authorizer(settings),
        authorization=_bearer('opr-write-secret'),
    )
    assert isinstance(result, Result)
    assert result.success is True


@pytest.mark.asyncio
async def test_message_write_rejects_known_cross_group_uuid_before_enqueue(monkeypatch):
    monkeypatch.setattr(
        ingest,
        'async_worker',
        SimpleNamespace(accepting=True, draining=False, depth=0, capacity=1000),
    )
    assert_episode_uuid_group = AsyncMock(side_effect=NodeGroupMismatchError())
    graphiti = cast(
        ZepGraphiti,
        SimpleNamespace(assert_episode_uuid_group=assert_episode_uuid_group),
    )
    request = AddMessagesRequest(
        group_id='other',
        messages=[
            Message(
                uuid='opr-owned-episode',
                name='attempted overwrite',
                content='content',
                role_type='user',
                role=None,
            )
        ],
    )

    with pytest.raises(HTTPException) as exc_info:
        settings = _settings()
        await add_messages(
            request, _http_request(graphiti), graphiti, settings, _authorizer(settings)
        )

    assert exc_info.value.status_code == 409
    assert_episode_uuid_group.assert_awaited_once_with('opr-owned-episode', 'other')


@pytest.mark.asyncio
async def test_episode_uuid_group_preflight_keeps_configured_neo4j_database(monkeypatch):
    with_database = Mock()
    driver = SimpleNamespace(
        provider=GraphProvider.NEO4J,
        with_database=with_database,
    )
    get_by_uuid = AsyncMock(return_value=SimpleNamespace(group_id='opr'))
    monkeypatch.setattr(
        'graph_service.zep_graphiti.EpisodicNode.get_by_uuid',
        get_by_uuid,
    )
    graphiti = cast(ZepGraphiti, SimpleNamespace(driver=driver))

    await ZepGraphiti.assert_episode_uuid_group(graphiti, 'episode-id', 'opr')

    with_database.assert_not_called()
    get_by_uuid.assert_awaited_once_with(driver, 'episode-id')


@pytest.mark.asyncio
async def test_episode_uuid_group_preflight_selects_falkordb_group(monkeypatch):
    group_driver = object()
    with_database = Mock(return_value=group_driver)
    driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB,
        with_database=with_database,
    )
    get_by_uuid = AsyncMock(return_value=SimpleNamespace(group_id='opr'))
    monkeypatch.setattr(
        'graph_service.zep_graphiti.EpisodicNode.get_by_uuid',
        get_by_uuid,
    )
    graphiti = cast(ZepGraphiti, SimpleNamespace(driver=driver))

    await ZepGraphiti.assert_episode_uuid_group(graphiti, 'episode-id', 'opr')

    with_database.assert_called_once_with('opr')
    get_by_uuid.assert_awaited_once_with(group_driver, 'episode-id')


@pytest.mark.asyncio
async def test_opr_write_requires_write_bearer_but_non_opr_write_remains_compatible():
    save_entity_node = AsyncMock(return_value={'uuid': 'entity'})
    graphiti = cast(ZepGraphiti, SimpleNamespace(save_entity_node=save_entity_node))
    settings = _settings()
    opr_request = AddEntityNodeRequest(
        uuid='entity', group_id='opr', name='name', summary='summary'
    )

    with pytest.raises(HTTPException) as exc_info:
        await add_entity_node(opr_request, graphiti, settings, _authorizer(settings))
    assert exc_info.value.status_code == 403
    with pytest.raises(HTTPException) as exc_info:
        await add_entity_node(
            opr_request,
            graphiti,
            settings,
            _authorizer(settings),
            authorization=_bearer('opr-read-secret'),
        )
    assert exc_info.value.status_code == 403
    save_entity_node.assert_not_awaited()

    await add_entity_node(
        opr_request,
        graphiti,
        settings,
        _authorizer(settings),
        authorization=_bearer('opr-write-secret'),
    )
    save_entity_node.assert_awaited_once()

    save_entity_node.reset_mock()
    await add_entity_node(
        AddEntityNodeRequest(
            uuid='other-entity',
            group_id='other',
            name='name',
            summary='summary',
        ),
        graphiti,
        settings,
        _authorizer(settings),
    )
    save_entity_node.assert_awaited_once()


@pytest.mark.asyncio
async def test_entity_write_maps_atomic_group_mismatch_to_conflict():
    save_entity_node = AsyncMock(side_effect=NodeGroupMismatchError())
    graphiti = cast(ZepGraphiti, SimpleNamespace(save_entity_node=save_entity_node))
    request = AddEntityNodeRequest(
        uuid='opr-owned-entity',
        group_id='other',
        name='attempted overwrite',
        summary='summary',
    )

    with pytest.raises(HTTPException) as exc_info:
        settings = _settings()
        await add_entity_node(request, graphiti, settings, _authorizer(settings))

    assert exc_info.value.status_code == 409
    save_entity_node.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_requires_opr_bearer_for_explicit_or_unrestricted_group_access():
    graphiti_search = AsyncMock(return_value=[])
    graphiti = cast(ZepGraphiti, SimpleNamespace(search=graphiti_search))
    settings = _settings()

    for group_ids in (None, [], [''], ['other', 'opr']):
        with pytest.raises(HTTPException) as exc_info:
            await search(
                SearchQuery(query='query', group_ids=group_ids),
                graphiti,
                settings,
                _authorizer(settings),
            )
        assert exc_info.value.status_code == 403
    with pytest.raises(HTTPException) as exc_info:
        await search(
            SearchQuery(query='query', group_ids=['opr']),
            graphiti,
            settings,
            _authorizer(settings),
            authorization=_bearer('opr-write-secret'),
        )
    assert exc_info.value.status_code == 403
    graphiti_search.assert_not_awaited()

    await search(
        SearchQuery(query='query', group_ids=['opr']),
        graphiti,
        settings,
        _authorizer(settings),
        authorization=_bearer('opr-read-secret'),
    )
    graphiti_search.assert_awaited_once()

    graphiti_search.reset_mock()
    await search(
        SearchQuery(query='query', group_ids=['other']),
        graphiti,
        settings,
        _authorizer(settings),
    )
    graphiti_search.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_memory_forwards_center_node_uuid_to_node_distance_search():
    graphiti_search = AsyncMock(return_value=[])
    graphiti = cast(ZepGraphiti, SimpleNamespace(search=graphiti_search))
    request = GetMemoryRequest(
        group_id='opr',
        max_facts=7,
        center_node_uuid='repo-entity-node',
        messages=[
            Message(
                content='review context query',
                role_type='user',
                role=None,
            )
        ],
    )

    settings = _settings()
    response = await get_memory(
        request,
        graphiti,
        settings,
        _authorizer(settings),
        authorization=_bearer('opr-read-secret'),
    )

    assert response.facts == []
    graphiti_search.assert_awaited_once_with(
        group_ids=['opr'],
        query='user(): review context query\n',
        center_node_uuid='repo-entity-node',
        num_results=7,
    )


@pytest.mark.asyncio
async def test_ordinary_opr_listing_uses_read_bearer():
    ordinary = AsyncMock(return_value=[])
    graphiti = cast(
        ZepGraphiti,
        SimpleNamespace(
            retrieve_episodes=ordinary,
        ),
    )
    settings = _settings()

    with pytest.raises(HTTPException) as exc_info:
        await get_episodes('opr', 10, graphiti, settings, _authorizer(settings))
    assert exc_info.value.status_code == 403
    with pytest.raises(HTTPException) as exc_info:
        await get_episodes(
            'opr',
            10,
            graphiti,
            settings,
            _authorizer(settings),
            authorization=_bearer('wrong'),
        )
    assert exc_info.value.status_code == 403
    ordinary.assert_not_awaited()

    await get_episodes(
        'opr',
        10,
        graphiti,
        settings,
        _authorizer(settings),
        authorization=_bearer('opr-read-secret'),
    )
    ordinary.assert_awaited_once()


@pytest.mark.asyncio
async def test_legacy_destructive_routes_require_admin_and_group_delete_forbids_opr():
    delete_edge = AsyncMock()
    delete_graph_group = AsyncMock()
    graphiti = cast(
        ZepGraphiti,
        SimpleNamespace(delete_entity_edge=delete_edge, delete_group=delete_graph_group),
    )
    settings = _settings()

    with pytest.raises(HTTPException) as exc_info:
        await delete_entity_edge('edge', graphiti, settings, _authorizer(settings))
    assert exc_info.value.status_code == 403
    with pytest.raises(HTTPException) as exc_info:
        await delete_entity_edge(
            'edge',
            graphiti,
            settings,
            _authorizer(settings),
            authorization=_bearer('opr-read-secret'),
        )
    assert exc_info.value.status_code == 403
    delete_edge.assert_not_awaited()

    await delete_entity_edge(
        'edge',
        graphiti,
        settings,
        _authorizer(settings),
        authorization=_bearer('admin-secret'),
    )
    delete_edge.assert_awaited_once_with('edge')

    with pytest.raises(HTTPException) as exc_info:
        await delete_group(
            'opr',
            graphiti,
            settings,
            _authorizer(settings),
            authorization=_bearer('admin-secret'),
        )
    assert exc_info.value.status_code == 403
    delete_graph_group.assert_not_awaited()

    await delete_group(
        'other',
        graphiti,
        settings,
        _authorizer(settings),
        authorization=_bearer('admin-secret'),
    )
    delete_graph_group.assert_awaited_once_with('other')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'provider',
    [
        GraphProvider.NEO4J,
        GraphProvider.NEPTUNE,
    ],
)
async def test_legacy_episode_delete_atomically_excludes_opr_group(provider: GraphProvider):
    execute_query = AsyncMock(return_value=([{'uuid': 'episode'}], None, None))
    driver = SimpleNamespace(provider=provider, execute_query=execute_query)
    graphiti = cast(ZepGraphiti, SimpleNamespace(driver=driver))

    settings = _settings()
    await delete_episode(
        'episode',
        graphiti,
        settings,
        _authorizer(settings),
        authorization=_bearer('admin-secret'),
    )

    query_call = execute_query.await_args
    assert query_call is not None
    query = query_call.args[0]
    assert query.index('SET episode._opr_conditional_delete_lock') < query.index(
        'episode.group_id <> $opr_group_id'
    )
    assert 'coalesce(episode.opr_deleted, false) = false' in query
    assert 'DETACH DELETE episode' in query
    assert query_call.kwargs == {'uuid': 'episode', 'opr_group_id': 'opr'}


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', [GraphProvider.KUZU, GraphProvider.FALKORDB])
async def test_legacy_episode_delete_fails_closed_on_backend_without_safe_lock_query(
    provider: GraphProvider,
):
    execute_query = AsyncMock()
    driver = SimpleNamespace(provider=provider, execute_query=execute_query)
    graphiti = cast(ZepGraphiti, SimpleNamespace(driver=driver))

    with pytest.raises(HTTPException) as exc_info:
        settings = _settings()
        await delete_episode(
            'episode',
            graphiti,
            settings,
            _authorizer(settings),
            authorization=_bearer('admin-secret'),
        )

    assert exc_info.value.status_code == 501
    execute_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_episode_delete_fails_closed_when_atomic_group_guard_does_not_delete():
    execute_query = AsyncMock(return_value=([], None, None))
    driver = SimpleNamespace(provider=GraphProvider.NEO4J, execute_query=execute_query)
    graphiti = cast(ZepGraphiti, SimpleNamespace(driver=driver))

    with pytest.raises(HTTPException) as exc_info:
        settings = _settings()
        await delete_episode(
            'episode',
            graphiti,
            settings,
            _authorizer(settings),
            authorization=_bearer('admin-secret'),
        )

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_global_clear_requires_admin_and_explicit_enable(monkeypatch):
    clear_data = AsyncMock()
    monkeypatch.setattr('graph_service.routers.ingest.clear_data', clear_data)
    build_indices = AsyncMock()
    graphiti = cast(
        ZepGraphiti,
        SimpleNamespace(driver=SimpleNamespace(), build_indices_and_constraints=build_indices),
    )

    with pytest.raises(HTTPException) as exc_info:
        settings = _settings(clear_enabled=False)
        await clear(
            graphiti,
            settings,
            _authorizer(settings),
            authorization=_bearer('admin-secret'),
        )
    assert exc_info.value.status_code == 403
    clear_data.assert_not_awaited()

    with pytest.raises(HTTPException) as exc_info:
        settings = _settings(clear_enabled=True)
        await clear(
            graphiti,
            settings,
            _authorizer(settings),
            authorization=_bearer('opr-read-secret'),
        )
    assert exc_info.value.status_code == 403
    clear_data.assert_not_awaited()

    settings = _settings(clear_enabled=True)
    await clear(
        graphiti,
        settings,
        _authorizer(settings),
        authorization=_bearer('admin-secret'),
    )
    clear_data.assert_awaited_once_with(graphiti.driver)
    build_indices.assert_awaited_once_with()
