import json
import time
from collections.abc import Callable
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from jwt.algorithms import RSAAlgorithm

from graph_service.auth import GraphitiAuthorizer, Permission
from graph_service.config import OprAuthMode, Settings
from graph_service.routers.ingest import _authorize_episode_retirement
from graph_service.routers.retrieve import _authorize_reconciliation_listing

ISSUER = 'https://dev-auth.legalzoom.com/'
JWKS_URL = f'{ISSUER}jwks.json'
AUDIENCE = 'urn:apigee:target:api'
FLEET_EPOCH = 'fleet-epoch-' + ('e' * 32)

PRIVATE_KEY_1 = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PRIVATE_KEY_2 = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _public_jwk(private_key, kid: str) -> dict[str, Any]:
    value = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    value.update({'kid': kid, 'alg': 'RS256', 'use': 'sig'})
    return value


JWK_1 = _public_jwk(PRIVATE_KEY_1, 'key-1')
JWK_2 = _public_jwk(PRIVATE_KEY_2, 'key-2')


def _jwt_values(**overrides) -> dict[str, Any]:
    values: dict[str, Any] = {
        'openai_api_key': 'test',
        'ingest_queue_maxsize': 1000,
        'opr_auth_required': True,
        'opr_auth_mode': 'lz_jwt',
        'opr_writer_fleet_epoch': FLEET_EPOCH,
        'opr_jwt_issuer': ISSUER,
        'opr_jwt_jwks_url': JWKS_URL,
        'opr_jwt_audience': AUDIENCE,
        'opr_jwt_allowed_client_ids': 'opr-client-id,opr-reconciliation-client-id',
        'opr_jwt_read_scope': 'graphiti:read',
        'opr_jwt_write_scope': 'graphiti:write',
        'opr_jwt_reconciliation_scope': 'graphiti:reconcile',
        'opr_jwt_retirement_scope': 'graphiti:retire',
        'opr_jwt_admin_scope': 'graphiti:admin',
    }
    values.update(overrides)
    return values


def _settings(**overrides) -> Settings:
    return Settings.model_validate(_jwt_values(**overrides))


def _claims(scope: str | list[str], **overrides) -> dict[str, Any]:
    now = int(time.time())
    claims: dict[str, Any] = {
        'iss': ISSUER,
        'aud': [AUDIENCE],
        'sub': 'opr-client-id',
        'azp': 'opr-client-id',
        'iat': now,
        'exp': now + 300,
        'scope': scope,
        'gty': 'client-credentials',
    }
    claims.update(overrides)
    return claims


def _token(
    scope: str | list[str],
    *,
    private_key=PRIVATE_KEY_1,
    kid: str | None = 'key-1',
    algorithm: str = 'RS256',
    claims: dict[str, Any] | None = None,
) -> str:
    headers = {} if kid is None else {'kid': kid}
    key = private_key if algorithm == 'RS256' else 'symmetric-test-key-that-is-32-bytes'
    return jwt.encode(claims or _claims(scope), key, algorithm=algorithm, headers=headers)


def _client(
    handler: Callable[[httpx.Request], httpx.Response] | None = None,
) -> httpx.AsyncClient:
    if handler is None:

        def default_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={'keys': [JWK_1]}, request=request)

        handler = default_handler
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)


async def _started_authorizer(
    *,
    settings: Settings | None = None,
    client: httpx.AsyncClient | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[GraphitiAuthorizer, httpx.AsyncClient]:
    http_client = client or _client()
    authorizer = GraphitiAuthorizer(settings or _settings(), http_client=http_client, clock=clock)
    await authorizer.start()
    return authorizer, http_client


def test_static_remains_the_default_auth_mode():
    settings = Settings.model_validate({'openai_api_key': 'test', 'ingest_queue_maxsize': 1})

    assert settings.opr_auth_mode is OprAuthMode.STATIC


def test_lz_jwt_mode_requires_the_existing_required_auth_gate():
    with pytest.raises(ValueError, match='OPR_AUTH_REQUIRED=true'):
        Settings.model_validate(_jwt_values(opr_auth_required=False))


def test_lz_jwt_mode_rejects_zero_jwks_refresh_interval():
    with pytest.raises(ValueError, match='greater than or equal to 1'):
        Settings.model_validate(_jwt_values(opr_jwks_refresh_min_interval_seconds=0))


@pytest.mark.parametrize(
    'field',
    [
        'opr_writer_fleet_epoch',
        'opr_jwt_issuer',
        'opr_jwt_jwks_url',
        'opr_jwt_audience',
        'opr_jwt_allowed_client_ids',
        'opr_jwt_read_scope',
        'opr_jwt_write_scope',
        'opr_jwt_reconciliation_scope',
        'opr_jwt_retirement_scope',
        'opr_jwt_admin_scope',
    ],
)
def test_lz_jwt_mode_rejects_each_missing_required_value(field: str):
    with pytest.raises(ValueError, match='requires non-empty values'):
        Settings.model_validate(_jwt_values(**{field: ''}))


@pytest.mark.parametrize('field', ['opr_jwt_issuer', 'opr_jwt_jwks_url'])
def test_lz_jwt_mode_requires_https_trusted_endpoint_configuration(field: str):
    with pytest.raises(ValueError, match='must be an HTTPS URL'):
        Settings.model_validate(_jwt_values(**{field: 'http://auth.example.test/'}))


def test_lz_jwt_mode_rejects_ambiguous_scope_configuration():
    with pytest.raises(ValueError, match='must be distinct'):
        Settings.model_validate(_jwt_values(opr_jwt_write_scope='graphiti:read'))
    with pytest.raises(ValueError, match='RFC 6749'):
        Settings.model_validate(_jwt_values(opr_jwt_write_scope='graphiti write'))


@pytest.mark.parametrize(
    'client_ids',
    ['', 'opr-client-id,', 'opr-client-id, opr-second-client', 'opr-client-id,opr-client-id'],
)
def test_lz_jwt_mode_rejects_unsafe_or_duplicate_client_allowlist(client_ids: str):
    with pytest.raises(ValueError, match='OPR_JWT_ALLOWED_CLIENT_IDS'):
        Settings.model_validate(_jwt_values(opr_jwt_allowed_client_ids=client_ids))


def test_lz_jwt_mode_does_not_require_legacy_identity_secrets():
    settings = _settings()

    assert settings.opr_read_token.get_secret_value() == ''
    assert settings.opr_write_token.get_secret_value() == ''
    assert settings.opr_reconciliation_token.get_secret_value() == ''
    assert settings.opr_retirement_token.get_secret_value() == ''
    assert settings.graphiti_admin_token.get_secret_value() == ''
    assert settings.opr_writer_fleet_epoch.get_secret_value() == FLEET_EPOCH


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('permission', 'scope'),
    [
        (Permission.READ, 'graphiti:read'),
        (Permission.WRITE, 'graphiti:write'),
        (Permission.RECONCILE, 'graphiti:reconcile'),
        (Permission.RETIRE, 'graphiti:retire'),
        (Permission.ADMIN, 'graphiti:admin'),
    ],
)
async def test_each_permission_accepts_only_its_configured_lz_scope(permission, scope):
    authorizer, client = await _started_authorizer()
    try:
        await authorizer.require(permission, f'Bearer {_token([scope])}')
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_scope_claim_accepts_lz_collection_and_standard_space_delimited_form():
    authorizer, client = await _started_authorizer()
    try:
        await authorizer.require(
            Permission.READ,
            f'Bearer {_token(["profile", "graphiti:read"])}',
        )
        await authorizer.require(
            Permission.READ,
            f'Bearer {_token("profile graphiti:read")}',
        )
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'authorization',
    [
        None,
        '',
        'raw-token',
        'Basic token',
        'Bearer',
        'Bearer ',
        'Bearer  token',
        'Bearer not-a-jwt',
        'Bearer ' + ('x' * (16 * 1024 + 1)),
    ],
)
async def test_missing_malformed_and_oversized_bearers_return_401(authorization):
    authorizer, client = await _started_authorizer()
    try:
        with pytest.raises(HTTPException) as exc_info:
            await authorizer.require(Permission.READ, authorization)
        assert exc_info.value.status_code == 401
        assert exc_info.value.headers == {'WWW-Authenticate': 'Bearer error="invalid_token"'}
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_valid_token_without_required_scope_returns_403_and_roles_do_not_grant():
    authorizer, client = await _started_authorizer()
    claims = _claims(['graphiti:read'], roles=['lz_admin'])
    try:
        with pytest.raises(HTTPException) as exc_info:
            await authorizer.require(
                Permission.ADMIN,
                f'Bearer {_token("graphiti:read", claims=claims)}',
            )
        assert exc_info.value.status_code == 403
        assert exc_info.value.headers is not None
        assert 'scope="graphiti:admin"' in exc_info.value.headers['WWW-Authenticate']
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_other_valid_lz_client_is_rejected_even_with_the_required_scope():
    authorizer, client = await _started_authorizer()
    claims = _claims(['graphiti:read'], azp='another-valid-lz-client')
    try:
        with pytest.raises(HTTPException) as exc_info:
            await authorizer.require(
                Permission.READ,
                f'Bearer {_token("graphiti:read", claims=claims)}',
            )
        assert exc_info.value.status_code == 401
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'claims',
    [
        _claims('graphiti:read', iss='https://wrong.example/'),
        _claims('graphiti:read', aud='wrong-audience'),
        _claims('graphiti:read', exp=int(time.time()) - 300),
        _claims('graphiti:read', nbf=int(time.time()) + 300),
        _claims('graphiti:read', iat=int(time.time()) + 300),
        _claims('graphiti:read', gty='authorization-code'),
        _claims('graphiti:read', azp=['opr-client-id']),
        _claims('graphiti:read', azp={'client_id': 'opr-client-id'}),
        {key: value for key, value in _claims('graphiti:read').items() if key != 'azp'},
        {key: value for key, value in _claims('graphiti:read').items() if key != 'exp'},
        {key: value for key, value in _claims('graphiti:read').items() if key != 'sub'},
    ],
)
async def test_invalid_issuer_audience_time_and_m2m_claims_return_401(claims):
    authorizer, client = await _started_authorizer()
    try:
        with pytest.raises(HTTPException) as exc_info:
            await authorizer.require(
                Permission.READ,
                f'Bearer {_token("graphiti:read", claims=claims)}',
            )
        assert exc_info.value.status_code == 401
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_wrong_signature_algorithm_and_missing_kid_return_401():
    authorizer, client = await _started_authorizer()
    tokens = [
        _token('graphiti:read', private_key=PRIVATE_KEY_2),
        _token('graphiti:read', kid='unknown-key'),
        _token('graphiti:read', algorithm='HS256'),
        _token('graphiti:read', kid=None),
    ]
    try:
        for token in tokens:
            with pytest.raises(HTTPException) as exc_info:
                await authorizer.require(Permission.READ, f'Bearer {token}')
            assert exc_info.value.status_code == 401
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_readiness_fails_closed_after_jwks_max_stale_and_recovers():
    calls = 0
    now = [100.0]
    issuer_available = [True]

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if not issuer_available[0]:
            return httpx.Response(503, request=request)
        return httpx.Response(200, json={'keys': [JWK_1]}, request=request)

    settings = _settings(
        opr_jwks_cache_ttl_seconds=10,
        opr_jwks_max_stale_seconds=20,
        opr_jwks_refresh_min_interval_seconds=5,
    )
    client = _client(handler)
    authorizer, _ = await _started_authorizer(
        settings=settings,
        client=client,
        clock=lambda: now[0],
    )
    try:
        assert await authorizer.is_ready() is True
        assert calls == 1

        issuer_available[0] = False
        now[0] = 121.0
        assert await authorizer.is_ready() is False
        assert calls == 2

        issuer_available[0] = True
        now[0] = 126.0
        assert await authorizer.is_ready() is True
        assert calls == 3
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_jwks_is_cached_and_unknown_kid_refreshes_for_rotation():
    calls = 0
    now = [100.0]

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        keys = [JWK_1] if calls == 1 else [JWK_1, JWK_2]
        return httpx.Response(200, json={'keys': keys}, request=request)

    client = _client(handler)
    authorizer, _ = await _started_authorizer(client=client, clock=lambda: now[0])
    try:
        await authorizer.require(Permission.READ, f'Bearer {_token("graphiti:read")}')
        await authorizer.require(Permission.READ, f'Bearer {_token("graphiti:read")}')
        assert calls == 1

        now[0] += 10
        rotated = _token('graphiti:read', private_key=PRIVATE_KEY_2, kid='key-2')
        await authorizer.require(Permission.READ, f'Bearer {rotated}')
        assert calls == 2
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_unknown_kid_refreshes_are_throttled():
    calls = 0
    now = [100.0]

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={'keys': [JWK_1]}, request=request)

    settings = _settings(opr_jwks_refresh_min_interval_seconds=5)
    client = _client(handler)
    authorizer, _ = await _started_authorizer(
        settings=settings,
        client=client,
        clock=lambda: now[0],
    )
    unknown_kid_token = _token('graphiti:read', kid='unknown-key')
    try:
        for _ in range(3):
            with pytest.raises(HTTPException) as exc_info:
                await authorizer.require(Permission.READ, f'Bearer {unknown_kid_token}')
            assert exc_info.value.status_code == 401
        assert calls == 1

        now[0] += 5
        with pytest.raises(HTTPException):
            await authorizer.require(Permission.READ, f'Bearer {unknown_kid_token}')
        assert calls == 2

        with pytest.raises(HTTPException):
            await authorizer.require(Permission.READ, f'Bearer {unknown_kid_token}')
        assert calls == 2
    finally:
        await client.aclose()


class _TrackingJwksStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.chunks_yielded = 0

    async def __aiter__(self):
        for chunk in self._chunks:
            self.chunks_yielded += 1
            yield chunk


@pytest.mark.asyncio
async def test_oversized_declared_jwks_is_rejected_before_reading_the_body():
    stream = _TrackingJwksStream([b'body must not be read'])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={'Content-Length': str(256 * 1024 + 1)},
            stream=stream,
            request=request,
        )

    client = _client(handler)
    authorizer = GraphitiAuthorizer(_settings(), http_client=client)
    try:
        with pytest.raises(RuntimeError, match='cannot start JWT authorization'):
            await authorizer.start()
        assert stream.chunks_yielded == 0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_oversized_chunked_jwks_without_content_length_stops_streaming_at_limit():
    chunks = [b'x' * (64 * 1024) for _ in range(10)]
    stream = _TrackingJwksStream(chunks)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream, request=request)

    client = _client(handler)
    authorizer = GraphitiAuthorizer(_settings(), http_client=client)
    try:
        with pytest.raises(RuntimeError, match='cannot start JWT authorization'):
            await authorizer.start()
        assert stream.chunks_yielded < len(chunks)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_bounded_stale_key_works_during_outage_then_fails_closed():
    calls = 0
    now = [100.0]

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json={'keys': [JWK_1]}, request=request)
        return httpx.Response(503, request=request)

    settings = _settings(
        opr_jwks_cache_ttl_seconds=10,
        opr_jwks_max_stale_seconds=20,
        opr_jwks_refresh_min_interval_seconds=1,
    )
    client = _client(handler)
    authorizer, _ = await _started_authorizer(
        settings=settings,
        client=client,
        clock=lambda: now[0],
    )
    token = _token('graphiti:read')
    try:
        now[0] = 111
        await authorizer.require(Permission.READ, f'Bearer {token}')

        now[0] = 121
        with pytest.raises(HTTPException) as exc_info:
            await authorizer.require(Permission.READ, f'Bearer {token}')
        assert exc_info.value.status_code == 503
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_failed_refresh_cooldown_never_extends_cached_keys_past_max_stale():
    calls = 0
    now = [100.0]

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json={'keys': [JWK_1]}, request=request)
        return httpx.Response(503, request=request)

    settings = _settings(
        opr_jwks_cache_ttl_seconds=10,
        opr_jwks_max_stale_seconds=20,
        opr_jwks_refresh_min_interval_seconds=30,
    )
    client = _client(handler)
    authorizer, _ = await _started_authorizer(
        settings=settings,
        client=client,
        clock=lambda: now[0],
    )
    token = _token('graphiti:read')
    try:
        now[0] = 119.0
        await authorizer.require(Permission.READ, f'Bearer {token}')
        assert calls == 2

        now[0] = 120.0
        await authorizer.require(Permission.READ, f'Bearer {token}')
        assert calls == 2

        now[0] = 120.001
        with pytest.raises(HTTPException) as exc_info:
            await authorizer.require(Permission.READ, f'Bearer {token}')
        assert exc_info.value.status_code == 503
        assert calls == 2
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['reconciliation', 'retirement'])
async def test_jwt_privileged_paths_authenticate_before_validating_fences(path: str):
    settings = _settings()
    authorizer, client = await _started_authorizer(settings=settings)

    async def authorize(authorization: str) -> None:
        if path == 'reconciliation':
            await _authorize_reconciliation_listing(
                authorizer,
                settings,
                authorization,
                None,
                'wrong-writer-fleet-epoch',
                'wrong-group',
            )
        else:
            await _authorize_episode_retirement(
                authorizer,
                settings,
                authorization,
                None,
                'wrong-writer-fleet-epoch',
                'wrong-operation',
                'wrong-group',
            )

    scope = 'graphiti:reconcile' if path == 'reconciliation' else 'graphiti:retire'
    try:
        with pytest.raises(HTTPException) as invalid_bearer:
            await authorize('Bearer not-a-jwt')
        assert invalid_bearer.value.status_code == 401

        with pytest.raises(HTTPException) as invalid_fence:
            await authorize(f'Bearer {_token(scope)}')
        assert invalid_fence.value.status_code == 403
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'response',
    [
        httpx.Response(200, json={'not_keys': []}),
        httpx.Response(200, json={'keys': []}),
        httpx.Response(200, json={'keys': [JWK_1, JWK_1]}),
        httpx.Response(302, headers={'location': 'https://redirect.example/jwks'}),
    ],
)
async def test_malformed_empty_duplicate_and_redirected_jwks_abort_startup(response):
    def handler(request: httpx.Request) -> httpx.Response:
        response.request = request
        return response

    client = _client(handler)
    authorizer = GraphitiAuthorizer(_settings(), http_client=client)
    try:
        with pytest.raises(RuntimeError, match='cannot start JWT authorization'):
            await authorizer.start()
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_lz_jwt_mode_ignores_legacy_identity_token_instead_of_falling_back():
    legacy = 'legacy-' + ('x' * 32)
    settings = _settings(opr_read_token=legacy, opr_reconciliation_token='reconcile-' + ('y' * 32))
    authorizer, client = await _started_authorizer(settings=settings)
    try:
        with pytest.raises(HTTPException) as exc_info:
            await authorizer.require(Permission.READ, f'Bearer {legacy}')
        assert exc_info.value.status_code == 401

        with pytest.raises(HTTPException) as exc_info:
            await authorizer.require(
                Permission.RECONCILE,
                None,
                legacy_token='reconcile-' + ('y' * 32),
            )
        assert exc_info.value.status_code == 401
    finally:
        await client.aclose()
