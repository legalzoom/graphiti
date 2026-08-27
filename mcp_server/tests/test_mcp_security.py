"""Focused tests for the production MCP HTTP security contract."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from graphiti_core.nodes import EpisodicNode
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken
from mcp.server.fastmcp import FastMCP
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

import graphiti_mcp_server
from security import (
    McpAuthorizationError,
    McpSecuritySettings,
    StaticTokenVerifier,
    ToolScope,
    authorize_tool_access,
)

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


def test_non_ascii_tokens_are_compared_as_utf8_without_type_errors() -> None:
    unicode_token = READ_TOKEN + '\u00e9'
    settings = settings_from({'GRAPHITI_MCP_READ_TOKEN': unicode_token})

    assert settings.read_token == unicode_token

    with pytest.raises(ValueError, match='pairwise distinct'):
        settings_from(
            {
                'GRAPHITI_MCP_READ_TOKEN': unicode_token,
                'GRAPHITI_MCP_WRITE_TOKEN': unicode_token,
            }
        )


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
            await registered_tool('delete_episode')(uuid='denied')
    finally:
        auth_context_var.reset(token_context)

    token_context = auth_context_var.set(AuthenticatedUser(admin))
    try:
        accepted_admin = await registered_tool('delete_episode')(uuid='accepted-by-wrapper')
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
        SimpleNamespace(get_client=AsyncMock(return_value=SimpleNamespace(driver=object()))),
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
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
