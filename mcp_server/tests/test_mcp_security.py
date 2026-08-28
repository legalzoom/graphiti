"""Focused tests for the production MCP HTTP security contract."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, call

import pytest
from dotenv import dotenv_values
from graphiti_core import Graphiti
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.errors import GroupIdValidationError, NodeNotFoundError
from graphiti_core.nodes import EntityNode, EpisodicNode, SagaNode
from graphiti_core.search.search_config import SearchResults
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken
from mcp.server.fastmcp import FastMCP
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

import graphiti_mcp_server
from security import (
    DEFAULT_LOCAL_ORIGINS,
    McpAuthorizationError,
    McpSecuritySettings,
    StaticTokenVerifier,
    ToolScope,
    authorize_tool_access,
)
from services import graphiti_scope

READ_TOKEN = 'read_8Vp3xQa7N2mZ5Ls9J4tY6Rc1W0kHfBuE'
WRITE_TOKEN = 'write_2Md8pXv4Kq7Tn1Zs5Jc9Rw3Yh6La0FgB'
ADMIN_TOKEN = 'admin_7Hs2Va9Qm4Kx8Zp1Nc5Rt3Yw6Ld0BjFe'


def secure_environment(**overrides: str) -> dict[str, str]:
    environment = {
        'GRAPHITI_MCP_SECURITY_REQUIRED': 'true',
        'GRAPHITI_MCP_HTTP_AUTH_ENABLED': 'true',
        'GRAPHITI_MCP_ALLOWED_HOSTS': 'graphiti.dev.svc:8000,graphiti.dev.example.com',
        'GRAPHITI_MCP_ALLOWED_ORIGINS': 'https://console.dev.example.com',
        'GRAPHITI_MCP_ALLOWED_GROUPS': 'team-a,team-b',
        'GRAPHITI_MCP_READ_TOKEN': READ_TOKEN,
        'GRAPHITI_MCP_WRITE_TOKEN': WRITE_TOKEN,
        'GRAPHITI_MCP_ADMIN_TOKEN': ADMIN_TOKEN,
    }
    environment.update(overrides)
    return environment


def settings_from(overrides: Mapping[str, str] | None = None) -> McpSecuritySettings:
    return McpSecuritySettings.from_env(secure_environment(**dict(overrides or {})))


def initialize_payload() -> dict[str, object]:
    return {
        'jsonrpc': '2.0',
        'id': 1,
        'method': 'initialize',
        'params': {
            'protocolVersion': '2025-06-18',
            'capabilities': {},
            'clientInfo': {'name': 'security-test', 'version': '1'},
        },
    }


def mcp_headers(
    *,
    host: str = 'graphiti.dev.svc:8000',
    token: str | None = READ_TOKEN,
    origin: str | None = None,
) -> dict[str, str]:
    headers = {
        'host': host,
        'content-type': 'application/json',
        'accept': 'application/json, text/event-stream',
    }
    if token is not None:
        headers['authorization'] = f'Bearer {token}'
    if origin is not None:
        headers['origin'] = origin
    return headers


def build_test_server(settings: McpSecuritySettings) -> FastMCP:
    server = FastMCP(
        'security-test',
        token_verifier=StaticTokenVerifier(settings),
        auth=settings.auth_settings(),
        transport_security=settings.transport_security_settings(),
    )

    @server.custom_route('/health', methods=['GET'])
    async def health(_request):
        return JSONResponse({'status': 'healthy'})

    return server


def registered_tool(name: str):
    tool = graphiti_mcp_server.mcp._tool_manager.get_tool(name)
    assert tool is not None
    return tool.fn


@pytest.mark.asyncio
async def test_group_scoped_graphiti_copy_does_not_rebind_shared_client(monkeypatch) -> None:
    scoped_driver = SimpleNamespace(name='team-a-driver')
    base_driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB,
        clone=Mock(return_value=scoped_driver),
    )
    scoped_clients = SimpleNamespace(driver=scoped_driver)
    base_clients = SimpleNamespace(model_copy=Mock(return_value=scoped_clients))
    base_nodes = object()
    base_edges = object()
    embedder = object()
    client = cast(Any, object.__new__(Graphiti))
    client.driver = base_driver
    client.clients = base_clients
    client.nodes = base_nodes
    client.edges = base_edges
    client.embedder = embedder
    scoped_nodes = object()
    scoped_edges = object()
    node_namespace = Mock(return_value=scoped_nodes)
    edge_namespace = Mock(return_value=scoped_edges)
    monkeypatch.setattr(graphiti_scope, 'NodeNamespace', node_namespace)
    monkeypatch.setattr(graphiti_scope, 'EdgeNamespace', edge_namespace)

    scoped = await graphiti_scope.graphiti_for_group(client, 'team-a')
    second_scoped = await graphiti_scope.graphiti_for_group(client, 'team-a')

    assert scoped is not client
    assert second_scoped is not scoped
    assert scoped.driver is scoped_driver
    assert second_scoped.driver is scoped_driver
    assert scoped.clients is scoped_clients
    assert scoped.nodes is scoped_nodes
    assert scoped.edges is scoped_edges
    assert client.driver is base_driver
    assert client.clients is base_clients
    assert client.nodes is base_nodes
    assert client.edges is base_edges
    base_driver.clone.assert_called_once_with(database='team-a')
    assert base_clients.model_copy.call_args_list == [
        call(update={'driver': scoped_driver}),
        call(update={'driver': scoped_driver}),
    ]
    assert node_namespace.call_args_list == [
        call(scoped_driver, embedder),
        call(scoped_driver, embedder),
    ]
    assert edge_namespace.call_args_list == [
        call(scoped_driver, embedder),
        call(scoped_driver, embedder),
    ]


@pytest.mark.asyncio
async def test_group_driver_initialization_is_single_flight() -> None:
    initialization_gate = asyncio.Event()
    initialization_task = asyncio.create_task(initialization_gate.wait())
    scoped_driver = SimpleNamespace(_init_task=initialization_task)
    clone_called = asyncio.Event()

    def clone(*, database: str) -> SimpleNamespace:
        assert database == 'team-a'
        clone_called.set()
        return scoped_driver

    base_driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB,
        clone=Mock(side_effect=clone),
    )
    client = cast(Graphiti, SimpleNamespace(driver=base_driver))

    first = asyncio.create_task(graphiti_scope.driver_for_group(client, 'team-a'))
    second = asyncio.create_task(graphiti_scope.driver_for_group(client, 'team-a'))
    await asyncio.wait_for(clone_called.wait(), timeout=1)

    base_driver.clone.assert_called_once_with(database='team-a')
    assert not first.done()
    assert not second.done()

    initialization_gate.set()
    assert await first is scoped_driver
    assert await second is scoped_driver


@pytest.mark.asyncio
async def test_group_driver_initialization_is_serialized_across_falkor_groups() -> None:
    gates = {'team-a': asyncio.Event(), 'team-b': asyncio.Event()}
    first_clone_called = asyncio.Event()
    second_clone_called = asyncio.Event()
    cloned_groups: list[str] = []

    def clone(*, database: str) -> SimpleNamespace:
        cloned_groups.append(database)
        (first_clone_called if len(cloned_groups) == 1 else second_clone_called).set()
        return SimpleNamespace(_init_task=asyncio.create_task(gates[database].wait()))

    clone_driver = Mock(side_effect=clone)
    base_driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB,
        clone=clone_driver,
    )
    client = cast(Graphiti, SimpleNamespace(driver=base_driver))

    first = asyncio.create_task(graphiti_scope.driver_for_group(client, 'team-a'))
    second = asyncio.create_task(graphiti_scope.driver_for_group(client, 'team-b'))
    await asyncio.wait_for(first_clone_called.wait(), timeout=1)

    assert clone_driver.call_count == 1
    initialized_group = cloned_groups[0]
    gates[initialized_group].set()
    await asyncio.wait_for(second_clone_called.wait(), timeout=1)

    assert clone_driver.call_count == 2
    gates[({'team-a', 'team-b'} - {initialized_group}).pop()].set()
    await asyncio.gather(first, second)


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_leave_failed_group_initialization_cached() -> None:
    failure_gate = asyncio.Event()
    initialization_failed = asyncio.Event()
    first_clone_called = asyncio.Event()

    async def fail_initialization() -> None:
        await failure_gate.wait()
        initialization_failed.set()
        raise RuntimeError('index initialization failed')

    failed_driver = SimpleNamespace(_init_task=asyncio.create_task(fail_initialization()))
    recovered_driver = SimpleNamespace(_init_task=None)

    drivers = iter([failed_driver, recovered_driver])

    def clone(*, database: str) -> SimpleNamespace:
        assert database == 'team-a'
        first_clone_called.set()
        return next(drivers)

    clone_driver = Mock(side_effect=clone)
    base_driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB,
        clone=clone_driver,
    )
    client = cast(Graphiti, SimpleNamespace(driver=base_driver))

    waiter = asyncio.create_task(graphiti_scope.driver_for_group(client, 'team-a'))
    await asyncio.wait_for(first_clone_called.wait(), timeout=1)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    failure_gate.set()
    await asyncio.wait_for(initialization_failed.wait(), timeout=1)
    cache = cast(Any, client)._graphiti_mcp_group_driver_cache
    for _ in range(10):
        if 'team-a' not in cache.tasks:
            break
        await asyncio.sleep(0)

    assert 'team-a' not in cache.tasks
    assert await graphiti_scope.driver_for_group(client, 'team-a') is recovered_driver
    assert clone_driver.call_count == 2


@pytest.mark.asyncio
async def test_group_driver_does_not_clone_shared_database_backends() -> None:
    base_driver = SimpleNamespace(provider=GraphProvider.NEO4J, clone=Mock())
    client = cast(Graphiti, SimpleNamespace(driver=base_driver))

    assert await graphiti_scope.driver_for_group(client, 'team-a') is base_driver
    base_driver.clone.assert_not_called()


@pytest.mark.asyncio
async def test_falkor_logical_default_preserves_configured_base_database() -> None:
    initialization_gate = asyncio.Event()
    base_driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB,
        default_group_id='_',
        _database='custom-graph',
        _init_task=asyncio.create_task(initialization_gate.wait()),
        clone=Mock(),
    )
    client = cast(Graphiti, SimpleNamespace(driver=base_driver))

    request = asyncio.create_task(graphiti_scope.driver_for_group(client, '_'))
    await asyncio.sleep(0)
    assert not request.done()

    initialization_gate.set()
    assert await request is base_driver
    base_driver.clone.assert_not_called()


@pytest.mark.asyncio
async def test_group_driver_rejects_invalid_group_before_backend_side_effects() -> None:
    base_driver = SimpleNamespace(provider=GraphProvider.FALKORDB, clone=Mock())
    client = cast(Graphiti, SimpleNamespace(driver=base_driver))

    with pytest.raises(GroupIdValidationError):
        await graphiti_scope.driver_for_group(client, 'team/a')

    base_driver.clone.assert_not_called()


@pytest.mark.asyncio
async def test_center_node_lookup_searches_requested_falkor_graphs_not_ambient_driver(
    monkeypatch,
) -> None:
    ambient_driver = SimpleNamespace(provider=GraphProvider.FALKORDB)
    client = SimpleNamespace(driver=ambient_driver)
    team_a_driver = object()
    team_b_driver = object()
    group_driver = AsyncMock(side_effect=[team_a_driver, team_b_driver])
    get_node = AsyncMock(
        side_effect=[NodeNotFoundError('center-uuid'), SimpleNamespace(group_id='team-b')]
    )
    monkeypatch.setattr(graphiti_mcp_server, 'driver_for_group', group_driver)
    monkeypatch.setattr(EntityNode, 'get_by_uuid', get_node)

    result = await graphiti_mcp_server._entity_node_for_requested_groups(
        cast(Graphiti, client),
        'center-uuid',
        ['team-a', 'team-a', 'team-b'],
    )

    assert result.group_id == 'team-b'
    assert group_driver.await_args_list == [call(client, 'team-a'), call(client, 'team-b')]
    assert get_node.await_args_list == [
        call(team_a_driver, 'center-uuid'),
        call(team_b_driver, 'center-uuid'),
    ]


@pytest.mark.asyncio
async def test_get_episodes_routes_falkor_groups_and_applies_one_global_limit(
    monkeypatch,
) -> None:
    settings = settings_from()
    token = await StaticTokenVerifier(settings).verify_token(READ_TOKEN)
    assert token is not None
    base_client = cast(
        Graphiti,
        SimpleNamespace(driver=SimpleNamespace(provider=GraphProvider.FALKORDB)),
    )
    team_a_driver = object()
    team_b_driver = object()
    group_driver = AsyncMock(side_effect=[team_a_driver, team_b_driver])

    def episode(uuid: str, group_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            uuid=uuid,
            name=uuid,
            content='content',
            created_at=None,
            source='text',
            source_description='test',
            group_id=group_id,
        )

    get_episodes = AsyncMock(
        side_effect=[
            [episode('003', 'team-a'), episode('001', 'team-a')],
            [episode('004', 'team-b'), episode('002', 'team-b')],
        ]
    )
    monkeypatch.setattr(graphiti_mcp_server, 'MCP_SECURITY', settings)
    monkeypatch.setattr(graphiti_mcp_server, 'active_transport', 'http')
    monkeypatch.setattr(
        graphiti_mcp_server,
        'graphiti_service',
        SimpleNamespace(get_client=AsyncMock(return_value=base_client)),
    )
    monkeypatch.setattr(graphiti_mcp_server, 'driver_for_group', group_driver)
    monkeypatch.setattr(EpisodicNode, 'get_by_group_ids', get_episodes)

    token_context = auth_context_var.set(AuthenticatedUser(token))
    try:
        result = await registered_tool('get_episodes')(
            group_ids=['team-a', 'team-b'], max_episodes=3
        )
    finally:
        auth_context_var.reset(token_context)

    assert [item['uuid'] for item in result['episodes']] == ['004', '003', '002']
    assert group_driver.await_args_list == [
        call(base_client, 'team-a'),
        call(base_client, 'team-b'),
    ]
    assert get_episodes.await_args_list == [
        call(team_a_driver, ['team-a'], limit=3),
        call(team_b_driver, ['team-b'], limit=3),
    ]


@pytest.mark.asyncio
async def test_falkor_search_and_communities_use_initialized_scoped_clients(
    monkeypatch,
) -> None:
    settings = settings_from()
    token = await StaticTokenVerifier(settings).verify_token(WRITE_TOKEN)
    assert token is not None
    scoped_clients = {
        group_id: SimpleNamespace(
            search_=AsyncMock(return_value=SearchResults()),
            search=AsyncMock(return_value=[]),
            build_communities=AsyncMock(return_value=([], [])),
        )
        for group_id in ('team-a', 'team-b')
    }
    base_client = SimpleNamespace(
        driver=SimpleNamespace(provider=GraphProvider.FALKORDB),
        search_=AsyncMock(),
        search=AsyncMock(),
        build_communities=AsyncMock(),
    )
    scope_client = AsyncMock(side_effect=lambda _client, group_id: scoped_clients[group_id])
    monkeypatch.setattr(graphiti_mcp_server, 'MCP_SECURITY', settings)
    monkeypatch.setattr(graphiti_mcp_server, 'active_transport', 'http')
    monkeypatch.setattr(
        graphiti_mcp_server,
        'graphiti_service',
        SimpleNamespace(get_client=AsyncMock(return_value=base_client)),
    )
    monkeypatch.setattr(
        graphiti_mcp_server,
        'config',
        SimpleNamespace(graphiti=SimpleNamespace(group_id='team-a')),
        raising=False,
    )
    monkeypatch.setattr(graphiti_mcp_server, 'graphiti_for_group', scope_client)

    token_context = auth_context_var.set(AuthenticatedUser(token))
    try:
        await registered_tool('search_nodes')(
            query='query', group_ids=['team-a', 'team-b', 'team-a']
        )
        await registered_tool('search_memory_facts')(query='query', group_ids=['team-a', 'team-b'])
        await registered_tool('build_communities')(group_ids=['team-a', 'team-b'])
    finally:
        auth_context_var.reset(token_context)

    assert (
        scope_client.await_args_list
        == [
            call(base_client, 'team-a'),
            call(base_client, 'team-b'),
        ]
        * 3
    )
    for group_id, scoped_client in scoped_clients.items():
        scoped_client.search_.assert_awaited_once()
        assert scoped_client.search_.await_args.kwargs['group_ids'] == [group_id]
        scoped_client.search.assert_awaited_once()
        assert scoped_client.search.await_args.kwargs['group_ids'] == [group_id]
        scoped_client.build_communities.assert_awaited_once_with(group_ids=[group_id])
    base_client.search_.assert_not_awaited()
    base_client.search.assert_not_awaited()
    base_client.build_communities.assert_not_awaited()


def test_http_transport_rejects_missing_and_wrong_tokens() -> None:
    server = build_test_server(settings_from())

    with TestClient(server.streamable_http_app()) as client:
        missing = client.post('/mcp', headers=mcp_headers(token=None), json=initialize_payload())
        wrong = client.post(
            '/mcp', headers=mcp_headers(token='wrong-token'), json=initialize_payload()
        )

    assert missing.status_code == 401
    assert wrong.status_code == 401


def test_exact_host_and_origin_policy_accepts_only_configured_values() -> None:
    server = build_test_server(settings_from())

    with TestClient(server.streamable_http_app()) as client:
        accepted = client.post('/mcp', headers=mcp_headers(), json=initialize_payload())
        wrong_host = client.post(
            '/mcp',
            headers=mcp_headers(host='graphiti.other.svc:8000'),
            json=initialize_payload(),
        )
        wrong_origin = client.post(
            '/mcp',
            headers=mcp_headers(origin='https://untrusted.example.com'),
            json=initialize_payload(),
        )

    assert accepted.status_code == 200
    assert wrong_host.status_code == 421
    assert wrong_origin.status_code == 403


def test_default_local_transport_policy_accepts_a_custom_port() -> None:
    settings = McpSecuritySettings.from_env({})
    server = FastMCP(
        'local-port-test',
        transport_security=settings.transport_security_settings(),
    )

    with TestClient(server.streamable_http_app()) as client:
        response = client.post(
            '/mcp',
            headers=mcp_headers(host='localhost:8123', token=None),
            json=initialize_payload(),
        )

    assert response.status_code == 200


def test_example_env_preserves_local_browser_origin_defaults() -> None:
    example_path = Path(__file__).parent.parent / '.env.example'
    example_environment = {
        key: value for key, value in dotenv_values(example_path).items() if value is not None
    }

    settings = McpSecuritySettings.from_env(example_environment)

    assert settings.allowed_origins == DEFAULT_LOCAL_ORIGINS
    assert settings.allowed_origins_explicit is False


def test_health_route_remains_unauthenticated() -> None:
    server = build_test_server(settings_from())

    with TestClient(server.streamable_http_app()) as client:
        response = client.get('/health')

    assert response.status_code == 200
    assert response.json() == {'status': 'healthy'}


@pytest.mark.asyncio
async def test_scope_separation_and_role_inheritance() -> None:
    settings = settings_from()
    verifier = StaticTokenVerifier(settings)
    read = await verifier.verify_token(READ_TOKEN)
    write = await verifier.verify_token(WRITE_TOKEN)
    admin = await verifier.verify_token(ADMIN_TOKEN)
    assert read is not None and write is not None and admin is not None

    authorize_tool_access(
        settings,
        required_scope=ToolScope.READ,
        requested_groups=['team-a'],
        access_token=read,
    )
    with pytest.raises(McpAuthorizationError, match='graphiti:write scope is required'):
        authorize_tool_access(
            settings,
            required_scope=ToolScope.WRITE,
            requested_groups=['team-a'],
            access_token=read,
        )

    authorize_tool_access(
        settings,
        required_scope=ToolScope.WRITE,
        requested_groups=['team-a'],
        access_token=write,
    )
    with pytest.raises(McpAuthorizationError, match='graphiti:admin scope is required'):
        authorize_tool_access(
            settings,
            required_scope=ToolScope.ADMIN,
            requested_groups=['team-a'],
            access_token=write,
        )

    authorize_tool_access(
        settings,
        required_scope=ToolScope.ADMIN,
        requested_groups=['team-b'],
        access_token=admin,
    )
    assert settings.auth_settings().required_scopes == []


@pytest.mark.asyncio
async def test_group_binding_denies_unconfigured_and_missing_groups() -> None:
    settings = settings_from()
    token = await StaticTokenVerifier(settings).verify_token(READ_TOKEN)
    assert token is not None

    authorize_tool_access(
        settings,
        required_scope=ToolScope.READ,
        requested_groups=['team-a', 'team-b'],
        access_token=token,
    )
    with pytest.raises(McpAuthorizationError, match='not authorized'):
        authorize_tool_access(
            settings,
            required_scope=ToolScope.READ,
            requested_groups=['other-team'],
            access_token=token,
        )
    with pytest.raises(McpAuthorizationError, match='explicit authorized'):
        authorize_tool_access(
            settings,
            required_scope=ToolScope.READ,
            requested_groups=[],
            access_token=token,
        )
    with pytest.raises(McpAuthorizationError, match='every requested'):
        authorize_tool_access(
            settings,
            required_scope=ToolScope.READ,
            requested_groups=['team-a', ''],
            access_token=token,
        )

    token_without_binding = AccessToken(
        token='[redacted]',
        client_id='test',
        scopes=[ToolScope.READ.value],
        claims={},
    )
    with pytest.raises(McpAuthorizationError, match='no valid Graphiti group binding'):
        authorize_tool_access(
            settings,
            required_scope=ToolScope.READ,
            requested_groups=['team-a'],
            access_token=token_without_binding,
        )


@pytest.mark.asyncio
async def test_destructive_tools_default_deny_and_require_admin_when_enabled() -> None:
    disabled = settings_from()
    disabled_admin = await StaticTokenVerifier(disabled).verify_token(ADMIN_TOKEN)
    assert disabled_admin is not None
    with pytest.raises(McpAuthorizationError, match='destructive MCP tools are disabled'):
        authorize_tool_access(
            disabled,
            required_scope=ToolScope.ADMIN,
            requested_groups=['team-a'],
            destructive=True,
            access_token=disabled_admin,
        )

    enabled = settings_from({'GRAPHITI_MCP_DESTRUCTIVE_TOOLS_ENABLED': 'true'})
    enabled_write = await StaticTokenVerifier(enabled).verify_token(WRITE_TOKEN)
    enabled_admin = await StaticTokenVerifier(enabled).verify_token(ADMIN_TOKEN)
    assert enabled_write is not None and enabled_admin is not None
    with pytest.raises(McpAuthorizationError, match='graphiti:admin scope is required'):
        authorize_tool_access(
            enabled,
            required_scope=ToolScope.ADMIN,
            requested_groups=['team-a'],
            destructive=True,
            access_token=enabled_write,
        )
    authorize_tool_access(
        enabled,
        required_scope=ToolScope.ADMIN,
        requested_groups=['team-a'],
        destructive=True,
        access_token=enabled_admin,
    )


def test_security_required_fails_closed_on_incomplete_configuration() -> None:
    with pytest.raises(ValueError, match='HTTP_AUTH_ENABLED=true'):
        McpSecuritySettings.from_env(
            {
                'GRAPHITI_MCP_SECURITY_REQUIRED': 'true',
                'GRAPHITI_MCP_ALLOWED_HOSTS': 'graphiti.dev.svc:8000',
                'GRAPHITI_MCP_ALLOWED_GROUPS': 'team-a',
            }
        )
    environment_without_hosts = secure_environment()
    environment_without_hosts.pop('GRAPHITI_MCP_ALLOWED_HOSTS')
    with pytest.raises(ValueError, match='explicit GRAPHITI_MCP_ALLOWED_HOSTS'):
        McpSecuritySettings.from_env(environment_without_hosts)
    environment_without_origins = secure_environment()
    environment_without_origins.pop('GRAPHITI_MCP_ALLOWED_ORIGINS')
    with pytest.raises(ValueError, match='explicit GRAPHITI_MCP_ALLOWED_ORIGINS'):
        McpSecuritySettings.from_env(environment_without_origins)
    server_only = settings_from({'GRAPHITI_MCP_ALLOWED_ORIGINS': ''})
    assert server_only.allowed_origins == ()
    with pytest.raises(ValueError, match='streamable HTTP transport'):
        settings_from().validate_runtime(transport='stdio', destroy_graph=False)
    with pytest.raises(ValueError, match='destroy-graph is disabled'):
        settings_from().validate_runtime(transport='http', destroy_graph=True)


def test_exact_transport_values_reject_wildcards() -> None:
    with pytest.raises(ValueError, match='wildcards'):
        settings_from({'GRAPHITI_MCP_ALLOWED_HOSTS': 'graphiti.dev.svc:*'})
    with pytest.raises(ValueError, match='wildcards'):
        settings_from({'GRAPHITI_MCP_ALLOWED_HOSTS': 'localhost:*'})
    with pytest.raises(ValueError, match='wildcards'):
        settings_from({'GRAPHITI_MCP_ALLOWED_ORIGINS': 'https://*.example.com'})
    with pytest.raises(ValueError, match='wildcards'):
        settings_from({'GRAPHITI_MCP_ALLOWED_ORIGINS': 'http://localhost:*'})
    with pytest.raises(ValueError, match='exact http'):
        settings_from({'GRAPHITI_MCP_ALLOWED_ORIGINS': 'https://console.\tdev.example.com'})


@pytest.mark.parametrize('group_id', ['team a', 't\u00e9am-a', 'team/a', 'team.a'])
def test_allowed_groups_must_match_graphiti_group_id_grammar(group_id: str) -> None:
    with pytest.raises(ValueError, match='ASCII letters'):
        settings_from({'GRAPHITI_MCP_ALLOWED_GROUPS': group_id})


def test_tokens_are_strong_distinct_and_redacted() -> None:
    settings = settings_from()
    rendered = repr(settings)
    assert READ_TOKEN not in rendered
    assert WRITE_TOKEN not in rendered
    assert ADMIN_TOKEN not in rendered

    with pytest.raises(ValueError) as weak_error:
        settings_from({'GRAPHITI_MCP_READ_TOKEN': 'short-secret'})
    assert 'short-secret' not in str(weak_error.value)

    with pytest.raises(ValueError) as duplicate_error:
        settings_from({'GRAPHITI_MCP_WRITE_TOKEN': READ_TOKEN})
    assert READ_TOKEN not in str(duplicate_error.value)

    verifier = StaticTokenVerifier(settings)
    verifier_rendered = repr(verifier)
    assert READ_TOKEN not in verifier_rendered
    assert WRITE_TOKEN not in verifier_rendered
    assert ADMIN_TOKEN not in verifier_rendered


@pytest.mark.parametrize(
    'invalid_token',
    [
        READ_TOKEN + '\u00e9',
        'x' * 16 + ' ' + 'y' * 16,
        'x' * 32 + '!',
        'x' * 32 + '\n',
    ],
)
def test_tokens_reject_values_that_cannot_round_trip_as_http_bearer_credentials(
    invalid_token: str,
) -> None:
    with pytest.raises(ValueError, match='HTTP token68-compatible ASCII') as exc_info:
        settings_from({'GRAPHITI_MCP_READ_TOKEN': invalid_token})

    assert invalid_token not in str(exc_info.value)


def test_base64_token68_credentials_round_trip_through_http_authentication() -> None:
    token = 'AbCdEf0123456789+/AbCdEf0123456789+/=='
    settings = settings_from({'GRAPHITI_MCP_READ_TOKEN': token})

    with TestClient(build_test_server(settings).streamable_http_app()) as client:
        response = client.post('/mcp', headers=mcp_headers(token=token), json=initialize_payload())

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_registered_wrappers_fail_closed_and_enforce_read_write_admin_scopes(
    monkeypatch,
) -> None:
    settings = settings_from({'GRAPHITI_MCP_DESTRUCTIVE_TOOLS_ENABLED': 'true'})
    verifier = StaticTokenVerifier(settings)
    read = await verifier.verify_token(READ_TOKEN)
    write = await verifier.verify_token(WRITE_TOKEN)
    admin = await verifier.verify_token(ADMIN_TOKEN)
    assert read is not None and write is not None and admin is not None

    monkeypatch.setattr(graphiti_mcp_server, 'MCP_SECURITY', settings)
    # This intentionally remains unresolved: auth-enabled startup must deny,
    # not bypass, tools before initialize_server records the transport.
    monkeypatch.setattr(graphiti_mcp_server, 'active_transport', None)
    monkeypatch.setattr(graphiti_mcp_server, 'graphiti_service', None)
    monkeypatch.setattr(graphiti_mcp_server, 'queue_service', None)

    token_context = auth_context_var.set(AuthenticatedUser(read))
    try:
        status = await registered_tool('get_status')()
        assert status['status'] == 'error'
        with pytest.raises(McpAuthorizationError, match='graphiti:write scope is required'):
            await registered_tool('add_memory')(
                name='denied',
                episode_body='denied',
                group_id='team-a',
            )
    finally:
        auth_context_var.reset(token_context)

    token_context = auth_context_var.set(AuthenticatedUser(write))
    try:
        accepted_write = await registered_tool('add_memory')(
            name='accepted-by-wrapper',
            episode_body='service remains intentionally uninitialized',
            group_id='team-a',
        )
        assert accepted_write['error'] == 'Services not initialized'
        with pytest.raises(McpAuthorizationError, match='graphiti:admin scope is required'):
            await registered_tool('delete_episode')(uuid='denied', group_id='team-a')
    finally:
        auth_context_var.reset(token_context)

    token_context = auth_context_var.set(AuthenticatedUser(admin))
    try:
        accepted_admin = await registered_tool('delete_episode')(
            uuid='accepted-by-wrapper', group_id='team-a'
        )
        assert accepted_admin['error'] == 'Graphiti service not initialized'
    finally:
        auth_context_var.reset(token_context)


@pytest.mark.asyncio
async def test_registered_wrapper_binds_an_omitted_group_to_the_configured_default(
    monkeypatch,
) -> None:
    settings = settings_from()
    token = await StaticTokenVerifier(settings).verify_token(READ_TOKEN)
    assert token is not None
    monkeypatch.setattr(graphiti_mcp_server, 'MCP_SECURITY', settings)
    monkeypatch.setattr(graphiti_mcp_server, 'active_transport', 'http')
    monkeypatch.setattr(graphiti_mcp_server, 'graphiti_service', None)
    monkeypatch.setattr(
        graphiti_mcp_server,
        'config',
        SimpleNamespace(graphiti=SimpleNamespace(group_id='not-authorized')),
        raising=False,
    )

    token_context = auth_context_var.set(AuthenticatedUser(token))
    try:
        with pytest.raises(McpAuthorizationError, match='not authorized'):
            await registered_tool('get_episodes')()
    finally:
        auth_context_var.reset(token_context)


@pytest.mark.asyncio
async def test_internal_resource_authorization_errors_are_not_returned_as_tool_data(
    monkeypatch,
) -> None:
    settings = settings_from()
    token = await StaticTokenVerifier(settings).verify_token(READ_TOKEN)
    assert token is not None
    monkeypatch.setattr(graphiti_mcp_server, 'MCP_SECURITY', settings)
    monkeypatch.setattr(graphiti_mcp_server, 'active_transport', 'http')
    monkeypatch.setattr(
        graphiti_mcp_server,
        'graphiti_service',
        SimpleNamespace(
            get_client=AsyncMock(
                return_value=SimpleNamespace(driver=SimpleNamespace(provider=GraphProvider.NEO4J))
            )
        ),
    )
    monkeypatch.setattr(
        EpisodicNode,
        'get_by_group_ids',
        AsyncMock(return_value=[SimpleNamespace(group_id='team-a')]),
    )
    monkeypatch.setattr(
        graphiti_mcp_server,
        '_authorize_returned_groups',
        Mock(side_effect=McpAuthorizationError('resource escaped its authorized group')),
    )

    token_context = auth_context_var.set(AuthenticatedUser(token))
    try:
        with pytest.raises(McpAuthorizationError, match='escaped its authorized group'):
            await registered_tool('get_episodes')(group_ids=['team-a'])
    finally:
        auth_context_var.reset(token_context)


@pytest.mark.asyncio
async def test_stdio_uuid_tools_preserve_requested_group_integrity(monkeypatch) -> None:
    settings = settings_from({'GRAPHITI_MCP_DESTRUCTIVE_TOOLS_ENABLED': 'true'})
    scoped_driver = object()
    edge = SimpleNamespace(group_id='team-b', delete=AsyncMock())
    scoped_client = SimpleNamespace(
        driver=scoped_driver,
        remove_episode=AsyncMock(),
        get_nodes_and_edges_by_episode=AsyncMock(),
        add_triplet=AsyncMock(),
    )
    queue = SimpleNamespace(add_episode=AsyncMock())
    base_client = SimpleNamespace(driver=SimpleNamespace(provider=GraphProvider.NEO4J))
    service = SimpleNamespace(
        get_client=AsyncMock(return_value=base_client),
        entity_types={},
        edge_types={},
        edge_type_map={},
    )
    monkeypatch.setattr(graphiti_mcp_server, 'MCP_SECURITY', settings)
    monkeypatch.setattr(graphiti_mcp_server, 'active_transport', 'stdio')
    monkeypatch.setattr(graphiti_mcp_server, 'graphiti_service', service)
    monkeypatch.setattr(graphiti_mcp_server, 'queue_service', queue)
    monkeypatch.setattr(
        graphiti_mcp_server,
        'config',
        SimpleNamespace(graphiti=SimpleNamespace(group_id='team-a')),
        raising=False,
    )
    monkeypatch.setattr(
        graphiti_mcp_server,
        'graphiti_for_group',
        AsyncMock(return_value=scoped_client),
    )
    monkeypatch.setattr(
        graphiti_mcp_server,
        'driver_for_group',
        AsyncMock(return_value=scoped_driver),
    )
    monkeypatch.setattr(EntityEdge, 'get_by_uuid', AsyncMock(return_value=edge))
    monkeypatch.setattr(
        EpisodicNode,
        'get_by_uuid',
        AsyncMock(return_value=SimpleNamespace(group_id='team-b')),
    )
    monkeypatch.setattr(
        EntityNode,
        'get_by_uuid',
        AsyncMock(return_value=SimpleNamespace(group_id='team-b')),
    )

    responses = [
        await registered_tool('get_entity_edge')(uuid='edge', group_id='team-a'),
        await registered_tool('delete_entity_edge')(uuid='edge', group_id='team-a'),
        await registered_tool('delete_episode')(uuid='episode', group_id='team-a'),
        await registered_tool('get_episode_entities')(episode_uuids=['episode'], group_id='team-a'),
        await registered_tool('add_triplet')(
            source_node_name='source',
            edge_name='relates',
            fact='source relates to target',
            target_node_name='target',
            group_id='team-a',
            source_node_uuid='node',
        ),
        await registered_tool('add_memory')(
            name='episode',
            episode_body='body',
            group_id='team-a',
            uuid='episode',
        ),
    ]

    assert all(
        'must belong to the requested Graphiti group' in response['error'] for response in responses
    )
    edge.delete.assert_not_awaited()
    scoped_client.remove_episode.assert_not_awaited()
    scoped_client.get_nodes_and_edges_by_episode.assert_not_awaited()
    scoped_client.add_triplet.assert_not_awaited()
    queue.add_episode.assert_not_awaited()


@pytest.mark.asyncio
async def test_add_memory_rejects_cross_group_episode_uuid_before_enqueue(monkeypatch) -> None:
    settings = settings_from()
    token = await StaticTokenVerifier(settings).verify_token(WRITE_TOKEN)
    assert token is not None
    queue = SimpleNamespace(add_episode=AsyncMock())
    lookup_driver = object()
    base_driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB,
        clone=Mock(return_value=lookup_driver),
    )
    service = SimpleNamespace(
        get_client=AsyncMock(return_value=SimpleNamespace(driver=base_driver)),
        entity_types={},
        edge_types={},
        edge_type_map={},
    )
    monkeypatch.setattr(graphiti_mcp_server, 'MCP_SECURITY', settings)
    monkeypatch.setattr(graphiti_mcp_server, 'active_transport', 'http')
    monkeypatch.setattr(graphiti_mcp_server, 'graphiti_service', service)
    monkeypatch.setattr(graphiti_mcp_server, 'queue_service', queue)
    monkeypatch.setattr(
        graphiti_mcp_server,
        'config',
        SimpleNamespace(graphiti=SimpleNamespace(group_id='team-a')),
        raising=False,
    )
    get_episode = AsyncMock(return_value=SimpleNamespace(group_id='team-b'))
    monkeypatch.setattr(
        EpisodicNode,
        'get_by_uuid',
        get_episode,
    )

    token_context = auth_context_var.set(AuthenticatedUser(token))
    try:
        with pytest.raises(McpAuthorizationError, match='episode UUID must belong'):
            await registered_tool('add_memory')(
                name='collision',
                episode_body='must not be queued',
                group_id='team-a',
                uuid='existing-team-b-uuid',
            )
    finally:
        auth_context_var.reset(token_context)

    queue.add_episode.assert_not_awaited()
    base_driver.clone.assert_called_once_with(database='team-a')
    get_episode.assert_awaited_once_with(lookup_driver, 'existing-team-b-uuid')


@pytest.mark.asyncio
async def test_add_memory_allows_new_caller_supplied_episode_uuid(monkeypatch) -> None:
    settings = settings_from()
    token = await StaticTokenVerifier(settings).verify_token(WRITE_TOKEN)
    assert token is not None
    queue = SimpleNamespace(add_episode=AsyncMock(return_value=1))
    lookup_driver = object()
    base_driver = SimpleNamespace(
        provider=GraphProvider.FALKORDB,
        clone=Mock(return_value=lookup_driver),
    )
    service = SimpleNamespace(
        get_client=AsyncMock(return_value=SimpleNamespace(driver=base_driver)),
        entity_types={},
        edge_types={},
        edge_type_map={},
    )
    monkeypatch.setattr(graphiti_mcp_server, 'MCP_SECURITY', settings)
    monkeypatch.setattr(graphiti_mcp_server, 'active_transport', 'http')
    monkeypatch.setattr(graphiti_mcp_server, 'graphiti_service', service)
    monkeypatch.setattr(graphiti_mcp_server, 'queue_service', queue)
    monkeypatch.setattr(
        graphiti_mcp_server,
        'config',
        SimpleNamespace(graphiti=SimpleNamespace(group_id='team-a')),
        raising=False,
    )
    get_episode = AsyncMock(
        side_effect=[
            NodeNotFoundError('new-uuid'),
            SimpleNamespace(group_id='team-a'),
            SimpleNamespace(group_id='team-a'),
        ]
    )
    monkeypatch.setattr(
        EpisodicNode,
        'get_by_uuid',
        get_episode,
    )

    token_context = auth_context_var.set(AuthenticatedUser(token))
    try:
        result = await registered_tool('add_memory')(
            name='new episode',
            episode_body='safe to queue',
            group_id='team-a',
            uuid='new-uuid',
            previous_episode_uuids=['previous-uuid'],
            saga_previous_episode_uuid='saga-previous-uuid',
        )
    finally:
        auth_context_var.reset(token_context)

    assert result['message'] == "Episode 'new episode' queued for processing in group 'team-a'"
    queue.add_episode.assert_awaited_once()
    base_driver.clone.assert_called_once_with(database='team-a')
    assert get_episode.await_args_list == [
        call(lookup_driver, 'new-uuid'),
        call(lookup_driver, 'previous-uuid'),
        call(lookup_driver, 'saga-previous-uuid'),
    ]


@pytest.mark.asyncio
async def test_group_bound_tools_route_lookup_and_mutation_through_same_scoped_client(
    monkeypatch,
) -> None:
    settings = settings_from({'GRAPHITI_MCP_DESTRUCTIVE_TOOLS_ENABLED': 'true'})
    token = await StaticTokenVerifier(settings).verify_token(ADMIN_TOKEN)
    assert token is not None
    base_client = SimpleNamespace(driver=SimpleNamespace(provider=GraphProvider.FALKORDB))
    scoped_driver = object()
    scoped_client = SimpleNamespace(
        driver=scoped_driver,
        add_triplet=AsyncMock(return_value=SimpleNamespace(nodes=[], edges=[])),
        remove_episode=AsyncMock(),
        get_nodes_and_edges_by_episode=AsyncMock(return_value=SimpleNamespace(nodes=[], edges=[])),
        summarize_saga=AsyncMock(
            return_value=SimpleNamespace(
                uuid='saga-uuid',
                name='saga-name',
                summary='summary',
                group_id='team-a',
            )
        ),
    )
    scope_client = AsyncMock(return_value=scoped_client)
    group_driver = AsyncMock(return_value=scoped_driver)
    service = SimpleNamespace(get_client=AsyncMock(return_value=base_client))
    edge = SimpleNamespace(group_id='team-a', delete=AsyncMock())
    get_edge = AsyncMock(return_value=edge)
    get_episode = AsyncMock(return_value=SimpleNamespace(group_id='team-a'))
    get_sagas = AsyncMock(
        return_value=[SimpleNamespace(uuid='saga-uuid', name='saga-name', group_id='team-a')]
    )
    clear = AsyncMock()
    monkeypatch.setattr(graphiti_mcp_server, 'MCP_SECURITY', settings)
    monkeypatch.setattr(graphiti_mcp_server, 'active_transport', 'http')
    monkeypatch.setattr(graphiti_mcp_server, 'graphiti_service', service)
    monkeypatch.setattr(
        graphiti_mcp_server,
        'config',
        SimpleNamespace(graphiti=SimpleNamespace(group_id='team-a')),
        raising=False,
    )
    monkeypatch.setattr(graphiti_mcp_server, 'graphiti_for_group', scope_client)
    monkeypatch.setattr(graphiti_mcp_server, 'driver_for_group', group_driver)
    monkeypatch.setattr(EntityEdge, 'get_by_uuid', get_edge)
    monkeypatch.setattr(EpisodicNode, 'get_by_uuid', get_episode)
    monkeypatch.setattr(SagaNode, 'get_by_group_ids', get_sagas)
    monkeypatch.setattr(graphiti_mcp_server, 'format_fact_result', Mock(return_value={'uuid': 'e'}))
    monkeypatch.setattr(graphiti_mcp_server, 'clear_data', clear)

    token_context = auth_context_var.set(AuthenticatedUser(token))
    try:
        await registered_tool('add_triplet')(
            source_node_name='source',
            edge_name='relates',
            fact='source relates to target',
            target_node_name='target',
            group_id='team-a',
        )
        await registered_tool('get_entity_edge')(uuid='edge-uuid', group_id='team-a')
        await registered_tool('delete_entity_edge')(uuid='edge-uuid', group_id='team-a')
        await registered_tool('delete_episode')(uuid='episode-uuid', group_id='team-a')
        await registered_tool('get_episode_entities')(
            episode_uuids=['episode-1', 'episode-2'],
            group_id='team-a',
        )
        await registered_tool('summarize_saga')(saga_name='saga-name', group_id='team-a')
        with pytest.raises(McpAuthorizationError, match='every requested'):
            await registered_tool('clear_graph')(group_ids=['team-a', ''])
        await registered_tool('clear_graph')(group_ids=['team-a'])
    finally:
        auth_context_var.reset(token_context)

    assert scope_client.await_args_list == [call(base_client, 'team-a')] * 6
    assert get_edge.await_args_list == [
        call(scoped_driver, 'edge-uuid'),
        call(scoped_driver, 'edge-uuid'),
    ]
    edge.delete.assert_awaited_once_with(scoped_driver)
    assert get_episode.await_args_list == [
        call(scoped_driver, 'episode-uuid'),
        call(scoped_driver, 'episode-1'),
        call(scoped_driver, 'episode-2'),
    ]
    scoped_client.remove_episode.assert_awaited_once_with('episode-uuid')
    scoped_client.get_nodes_and_edges_by_episode.assert_awaited_once_with(
        ['episode-1', 'episode-2']
    )
    get_sagas.assert_awaited_once_with(scoped_driver, ['team-a'])
    scoped_client.summarize_saga.assert_awaited_once_with('saga-uuid')
    scoped_client.add_triplet.assert_awaited_once()
    group_driver.assert_awaited_once_with(base_client, 'team-a')
    clear.assert_awaited_once_with(scoped_driver, group_ids=['team-a'])


@pytest.mark.asyncio
async def test_clear_graph_preserves_one_transaction_for_shared_database_backends(
    monkeypatch,
) -> None:
    settings = settings_from({'GRAPHITI_MCP_DESTRUCTIVE_TOOLS_ENABLED': 'true'})
    token = await StaticTokenVerifier(settings).verify_token(ADMIN_TOKEN)
    assert token is not None
    base_driver = SimpleNamespace(provider=GraphProvider.NEO4J)
    base_client = SimpleNamespace(driver=base_driver)
    clear = AsyncMock()
    group_driver = AsyncMock()
    monkeypatch.setattr(graphiti_mcp_server, 'MCP_SECURITY', settings)
    monkeypatch.setattr(graphiti_mcp_server, 'active_transport', 'http')
    monkeypatch.setattr(
        graphiti_mcp_server,
        'graphiti_service',
        SimpleNamespace(get_client=AsyncMock(return_value=base_client)),
    )
    monkeypatch.setattr(
        graphiti_mcp_server,
        'config',
        SimpleNamespace(graphiti=SimpleNamespace(group_id='team-a')),
        raising=False,
    )
    monkeypatch.setattr(graphiti_mcp_server, 'driver_for_group', group_driver)
    monkeypatch.setattr(graphiti_mcp_server, 'clear_data', clear)

    token_context = auth_context_var.set(AuthenticatedUser(token))
    try:
        await registered_tool('clear_graph')(group_ids=['team-a', 'team-b'])
    finally:
        auth_context_var.reset(token_context)

    clear.assert_awaited_once_with(base_driver, group_ids=['team-a', 'team-b'])
    group_driver.assert_not_awaited()


def test_secured_tool_omits_destructive_registration_in_required_mode(monkeypatch) -> None:
    settings = settings_from()
    isolated_server = FastMCP('destructive-registration-test')
    monkeypatch.setattr(graphiti_mcp_server, 'MCP_SECURITY', settings)
    monkeypatch.setattr(graphiti_mcp_server, 'mcp', isolated_server)

    @graphiti_mcp_server.secured_tool(ToolScope.ADMIN, destructive=True)
    async def dangerous_test_tool() -> None:
        return None

    assert isolated_server._tool_manager.get_tool('dangerous_test_tool') is None


def test_required_mode_protocol_hides_destructive_tools_and_denies_wrong_scope() -> None:
    """Exercise the real registered FastMCP server in an isolated import."""
    script = textwrap.dedent(
        """
        import json

        from starlette.testclient import TestClient

        import graphiti_mcp_server as graphiti_mcp


        def response_body(response):
            return next(
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith('data: ')
            )


        headers = {
            'authorization': 'Bearer read_8Vp3xQa7N2mZ5Ls9J4tY6Rc1W0kHfBuE',
            'host': 'graphiti.dev.svc:8000',
            'content-type': 'application/json',
            'accept': 'application/json, text/event-stream',
        }
        initialize = {
            'jsonrpc': '2.0',
            'id': 1,
            'method': 'initialize',
            'params': {
                'protocolVersion': '2025-06-18',
                'capabilities': {},
                'clientInfo': {'name': 'security-test', 'version': '1'},
            },
        }

        with TestClient(graphiti_mcp.mcp.streamable_http_app()) as client:
            initialized = client.post('/mcp', headers=headers, json=initialize)
            assert initialized.status_code == 200
            session_headers = {
                **headers,
                'mcp-session-id': initialized.headers['mcp-session-id'],
            }
            acknowledged = client.post(
                '/mcp',
                headers=session_headers,
                json={'jsonrpc': '2.0', 'method': 'notifications/initialized'},
            )
            assert acknowledged.status_code == 202

            listed = response_body(
                client.post(
                    '/mcp',
                    headers=session_headers,
                    json={'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list', 'params': {}},
                )
            )
            names = {tool['name'] for tool in listed['result']['tools']}
            assert not {'clear_graph', 'delete_episode', 'delete_entity_edge'} & names

            denied = response_body(
                client.post(
                    '/mcp',
                    headers=session_headers,
                    json={
                        'jsonrpc': '2.0',
                        'id': 3,
                        'method': 'tools/call',
                        'params': {
                            'name': 'add_memory',
                            'arguments': {
                                'name': 'denied',
                                'episode_body': 'denied',
                                'group_id': 'team-a',
                            },
                        },
                    },
                )
            )
            assert denied['result']['isError'] is True
            assert 'graphiti:write scope is required' in denied['result']['content'][0]['text']
        """
    )
    environment = os.environ.copy()
    environment.update(secure_environment(GRAPHITI_MCP_DESTRUCTIVE_TOOLS_ENABLED='false'))
    source_path = Path(__file__).parent.parent / 'src'
    environment['PYTHONPATH'] = os.pathsep.join(
        [str(source_path), environment.get('PYTHONPATH', '')]
    ).rstrip(os.pathsep)

    completed = subprocess.run(
        [sys.executable, '-c', script],
        cwd=Path(__file__).parent.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
