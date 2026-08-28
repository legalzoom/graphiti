"""Security policy for the HTTP Graphiti MCP transport.

The MCP SDK deliberately treats transport security (Host/Origin validation) and
authentication as separate controls.  This module keeps that distinction
explicit: an allowed Host is never considered caller authentication.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urlsplit

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import AnyHttpUrl

MIN_TOKEN_BYTES = 32
_HTTP_TOKEN68 = re.compile(r'[A-Za-z0-9._~+/\-]+=*', re.ASCII)
_GRAPHITI_GROUP_ID = re.compile(r'[A-Za-z0-9_-]+', re.ASCII)

DEFAULT_LOCAL_HOSTS = (
    'localhost:*',
    '127.0.0.1:*',
    '[::1]:*',
)
DEFAULT_LOCAL_ORIGINS = (
    'http://localhost:*',
    'http://127.0.0.1:*',
    'http://[::1]:*',
)


class ToolScope(str, Enum):
    """Scopes understood by the static MCP token verifier."""

    READ = 'graphiti:read'
    WRITE = 'graphiti:write'
    ADMIN = 'graphiti:admin'


class McpAuthorizationError(PermissionError):
    """Raised when an authenticated MCP caller is not authorized for an operation."""


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    raise ValueError('boolean security settings must be true/false')


def _parse_csv(value: str | None, *, default: Sequence[str] = ()) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    return tuple(item.strip() for item in value.split(',') if item.strip())


def _validate_exact_hosts(hosts: Sequence[str]) -> None:
    if not hosts:
        raise ValueError('GRAPHITI_MCP_ALLOWED_HOSTS must contain at least one exact Host value')
    for host in hosts:
        try:
            parsed = urlsplit(f'//{host}')
            _ = parsed.port
        except ValueError:
            parsed = None
        if (
            '*' in host
            or '://' in host
            or '/' in host
            or any(char.isspace() for char in host)
            or parsed is None
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                'GRAPHITI_MCP_ALLOWED_HOSTS entries must be exact host[:port] values; '
                'wildcards and URLs are not allowed'
            )


def _validate_exact_origins(origins: Sequence[str]) -> None:
    for origin in origins:
        try:
            parsed = urlsplit(origin)
            _ = parsed.port
        except ValueError:
            parsed = None
        if (
            '*' in origin
            or any(char.isspace() for char in origin)
            or parsed is None
            or parsed.scheme not in {'http', 'https'}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                'GRAPHITI_MCP_ALLOWED_ORIGINS entries must be exact http(s) origins; '
                'wildcards, credentials, paths, queries, and fragments are not allowed'
            )


def _validate_token(name: str, token: str | None) -> None:
    if token is None or not token:
        raise ValueError(f'{name} must be configured')
    if _HTTP_TOKEN68.fullmatch(token) is None:
        raise ValueError(f'{name} must use HTTP token68-compatible ASCII')
    if len(token.encode('utf-8')) < MIN_TOKEN_BYTES:
        raise ValueError(f'{name} must be at least {MIN_TOKEN_BYTES} bytes')
    normalized = token.lower()
    if len(set(token)) < 12 or any(
        marker in normalized
        for marker in ('changeme', 'change-me', 'replace-with', 'placeholder', 'your-token')
    ):
        raise ValueError(f'{name} must not use a placeholder value')


@dataclass(frozen=True)
class McpSecuritySettings:
    """Environment-backed MCP security settings.

    Token fields are excluded from ``repr`` so validation and startup logging can
    safely include this object without disclosing credentials.
    """

    security_required: bool = False
    http_auth_enabled: bool = False
    destructive_tools_enabled: bool = False
    allowed_hosts: tuple[str, ...] = DEFAULT_LOCAL_HOSTS
    allowed_origins: tuple[str, ...] = DEFAULT_LOCAL_ORIGINS
    allowed_groups: tuple[str, ...] = ()
    read_token: str | None = field(default=None, repr=False)
    write_token: str | None = field(default=None, repr=False)
    admin_token: str | None = field(default=None, repr=False)
    issuer_url: str = 'https://graphiti-mcp.invalid'
    resource_server_url: str | None = None
    allowed_hosts_explicit: bool = False
    allowed_origins_explicit: bool = False
    allowed_groups_explicit: bool = False

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> McpSecuritySettings:
        """Build and validate settings from an environment mapping."""
        allowed_hosts_value = environ.get('GRAPHITI_MCP_ALLOWED_HOSTS')
        allowed_origins_value = environ.get('GRAPHITI_MCP_ALLOWED_ORIGINS')
        allowed_groups_value = environ.get('GRAPHITI_MCP_ALLOWED_GROUPS')
        resource_server_url = environ.get('GRAPHITI_MCP_RESOURCE_SERVER_URL') or None
        settings = cls(
            security_required=_parse_bool(environ.get('GRAPHITI_MCP_SECURITY_REQUIRED')),
            http_auth_enabled=_parse_bool(environ.get('GRAPHITI_MCP_HTTP_AUTH_ENABLED')),
            destructive_tools_enabled=_parse_bool(
                environ.get('GRAPHITI_MCP_DESTRUCTIVE_TOOLS_ENABLED')
            ),
            allowed_hosts=_parse_csv(allowed_hosts_value, default=DEFAULT_LOCAL_HOSTS),
            allowed_origins=_parse_csv(allowed_origins_value, default=DEFAULT_LOCAL_ORIGINS),
            allowed_groups=_parse_csv(allowed_groups_value),
            read_token=environ.get('GRAPHITI_MCP_READ_TOKEN'),
            write_token=environ.get('GRAPHITI_MCP_WRITE_TOKEN'),
            admin_token=environ.get('GRAPHITI_MCP_ADMIN_TOKEN'),
            issuer_url=environ.get('GRAPHITI_MCP_ISSUER_URL', 'https://graphiti-mcp.invalid'),
            resource_server_url=resource_server_url,
            allowed_hosts_explicit=allowed_hosts_value is not None,
            allowed_origins_explicit=allowed_origins_value is not None,
            allowed_groups_explicit=allowed_groups_value is not None,
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Validate fail-closed production settings without revealing secrets."""
        # Match FastMCP's own safe localhost defaults so local `--port` overrides
        # keep working. Any explicitly configured production value remains an
        # exact match and may not contain a wildcard.
        if self.allowed_hosts_explicit or self.allowed_hosts != DEFAULT_LOCAL_HOSTS:
            _validate_exact_hosts(self.allowed_hosts)
        if self.allowed_origins_explicit or self.allowed_origins != DEFAULT_LOCAL_ORIGINS:
            _validate_exact_origins(self.allowed_origins)

        if self.security_required and not self.http_auth_enabled:
            raise ValueError(
                'GRAPHITI_MCP_SECURITY_REQUIRED requires GRAPHITI_MCP_HTTP_AUTH_ENABLED=true'
            )
        if self.security_required and not self.allowed_hosts_explicit:
            raise ValueError(
                'GRAPHITI_MCP_SECURITY_REQUIRED requires explicit GRAPHITI_MCP_ALLOWED_HOSTS'
            )
        if self.security_required and not self.allowed_origins_explicit:
            raise ValueError(
                'GRAPHITI_MCP_SECURITY_REQUIRED requires explicit '
                'GRAPHITI_MCP_ALLOWED_ORIGINS (an empty value is valid for server-only callers)'
            )
        if self.security_required and not self.allowed_groups_explicit:
            raise ValueError(
                'GRAPHITI_MCP_SECURITY_REQUIRED requires explicit GRAPHITI_MCP_ALLOWED_GROUPS'
            )

        if not self.http_auth_enabled:
            return

        _validate_token('GRAPHITI_MCP_READ_TOKEN', self.read_token)
        _validate_token('GRAPHITI_MCP_WRITE_TOKEN', self.write_token)
        _validate_token('GRAPHITI_MCP_ADMIN_TOKEN', self.admin_token)

        tokens = (self.read_token, self.write_token, self.admin_token)
        assert all(token is not None for token in tokens)  # narrowed by validation above
        for index, token in enumerate(tokens):
            for other in tokens[index + 1 :]:
                if hmac.compare_digest(
                    (token or '').encode('utf-8'),
                    (other or '').encode('utf-8'),
                ):
                    raise ValueError(
                        'Graphiti MCP read/write/admin tokens must be pairwise distinct'
                    )

        if not self.allowed_groups:
            raise ValueError('GRAPHITI_MCP_ALLOWED_GROUPS must contain at least one group')
        if any('*' in group for group in self.allowed_groups):
            raise ValueError('GRAPHITI_MCP_ALLOWED_GROUPS does not permit wildcard groups')
        if any(_GRAPHITI_GROUP_ID.fullmatch(group) is None for group in self.allowed_groups):
            raise ValueError(
                'GRAPHITI_MCP_ALLOWED_GROUPS entries may contain only ASCII letters, '
                'numbers, dashes, and underscores'
            )

        # Let Pydantic validate URL syntax now rather than after service initialization.
        self.auth_settings()

    def validate_runtime(self, *, transport: str, destroy_graph: bool) -> None:
        """Validate settings that depend on parsed CLI/YAML runtime configuration."""
        if self.security_required and transport != 'http':
            raise ValueError(
                'GRAPHITI_MCP_SECURITY_REQUIRED is supported only with the streamable HTTP transport'
            )
        if self.http_auth_enabled and destroy_graph:
            raise ValueError(
                '--destroy-graph is disabled while MCP HTTP authentication is enabled; '
                'run destructive maintenance through an explicitly authorized path'
            )

    def transport_security_settings(self) -> TransportSecuritySettings:
        """Return the exact Host/Origin policy frozen into FastMCP at construction."""
        return TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=list(self.allowed_hosts),
            allowed_origins=list(self.allowed_origins),
        )

    def auth_settings(self) -> AuthSettings:
        """Return FastMCP auth settings with no global scope requirement.

        Any valid token may initialize an MCP session; individual tools enforce
        their own read/write/admin scope.  A global scope would incorrectly deny
        valid role-specific tokens before tool authorization runs.
        """
        return AuthSettings(
            issuer_url=AnyHttpUrl(self.issuer_url),
            resource_server_url=(
                AnyHttpUrl(self.resource_server_url) if self.resource_server_url else None
            ),
            required_scopes=[],
        )


@dataclass(frozen=True)
class _TokenRecord:
    digest: bytes = field(repr=False)
    role: str
    scopes: tuple[str, ...]


class StaticTokenVerifier:
    """Constant-time verifier for the three configured static bearer tokens."""

    def __init__(self, settings: McpSecuritySettings):
        if not settings.http_auth_enabled:
            raise ValueError('static token verifier requires MCP HTTP authentication')
        settings.validate()
        assert settings.read_token is not None
        assert settings.write_token is not None
        assert settings.admin_token is not None
        self._allowed_groups = settings.allowed_groups
        self._records = (
            _TokenRecord(
                digest=self._digest(settings.read_token),
                role='read',
                scopes=(ToolScope.READ.value,),
            ),
            _TokenRecord(
                digest=self._digest(settings.write_token),
                role='write',
                scopes=(ToolScope.READ.value, ToolScope.WRITE.value),
            ),
            _TokenRecord(
                digest=self._digest(settings.admin_token),
                role='admin',
                scopes=(ToolScope.READ.value, ToolScope.WRITE.value, ToolScope.ADMIN.value),
            ),
        )

    @staticmethod
    def _digest(token: str) -> bytes:
        return hashlib.sha256(token.encode('utf-8')).digest()

    async def verify_token(self, token: str) -> AccessToken | None:
        candidate = self._digest(token)
        matched: _TokenRecord | None = None
        # Compare all records so role position does not create a timing shortcut.
        for record in self._records:
            if hmac.compare_digest(candidate, record.digest):
                matched = record
        if matched is None:
            return None
        return AccessToken(
            token='[redacted]',
            client_id=f'graphiti-mcp-static-{matched.role}',
            subject=matched.role,
            scopes=list(matched.scopes),
            claims={
                'graphiti_role': matched.role,
                'graphiti_allowed_groups': list(self._allowed_groups),
            },
        )


def authorize_tool_access(
    settings: McpSecuritySettings,
    *,
    required_scope: ToolScope,
    requested_groups: Sequence[str] | None = None,
    destructive: bool = False,
    access_token: AccessToken | None = None,
) -> None:
    """Enforce a tool's scope, configured destructive policy, and group binding."""
    if not settings.http_auth_enabled:
        return

    token = access_token if access_token is not None else get_access_token()
    if token is None:
        raise McpAuthorizationError('MCP authentication is required')
    if required_scope.value not in token.scopes:
        raise McpAuthorizationError(f'{required_scope.value} scope is required')
    if destructive and not settings.destructive_tools_enabled:
        raise McpAuthorizationError('destructive MCP tools are disabled')

    if requested_groups is None:
        return
    if not requested_groups:
        raise McpAuthorizationError('an explicit authorized Graphiti group is required')
    if any(
        not isinstance(group, str) or _GRAPHITI_GROUP_ID.fullmatch(group) is None
        for group in requested_groups
    ):
        raise McpAuthorizationError(
            'every requested Graphiti group must be a non-empty valid group ID'
        )
    groups = set(requested_groups)
    claims: dict[str, Any] = token.claims or {}
    claimed_groups = claims.get('graphiti_allowed_groups')
    if not isinstance(claimed_groups, list) or not all(
        isinstance(group, str) for group in claimed_groups
    ):
        raise McpAuthorizationError('token has no valid Graphiti group binding')
    allowed = set(claimed_groups).intersection(settings.allowed_groups)
    denied = groups.difference(allowed)
    if denied:
        raise McpAuthorizationError('requested Graphiti group is not authorized')


def authorize_resource_group(
    settings: McpSecuritySettings,
    group_id: str,
    *,
    required_scope: ToolScope,
    destructive: bool = False,
) -> None:
    """Authorize a group discovered after resolving a UUID-backed resource."""
    authorize_tool_access(
        settings,
        required_scope=required_scope,
        requested_groups=[group_id],
        destructive=destructive,
    )
