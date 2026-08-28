"""Authentication for the deployment-protected OPR graph.

Static capability tokens remain available for rollback compatibility. JWT mode
is deliberately exclusive: once selected, no legacy identity token is consulted
and every protected capability is authorized by an LZ access-token scope.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any

import httpx
import jwt
from fastapi import Depends, FastAPI, HTTPException, Request, status

from graph_service.config import OprAuthMode, Settings
from graph_service.protocol import (
    bearer_token_matches,
    is_http_token68,
    reconciliation_token_matches,
)

logger = logging.getLogger(__name__)

AUTHORIZER_STATE_ATTR = 'graphiti_authorizer'
_MAX_BEARER_TOKEN_BYTES = 16 * 1024
_MAX_JWKS_BYTES = 256 * 1024
_MAX_JWKS_KEYS = 32
_JWKS_READ_CHUNK_BYTES = 8 * 1024
_JWKS_HTTP_TIMEOUT_SECONDS = 5.0


class Permission(str, Enum):
    READ = 'read'
    WRITE = 'write'
    RECONCILE = 'reconcile'
    RETIRE = 'retire'
    ADMIN = 'admin'


_STATIC_SECRET_ATTR = {
    Permission.READ: 'opr_read_token',
    Permission.WRITE: 'opr_write_token',
    Permission.RECONCILE: 'opr_reconciliation_token',
    Permission.RETIRE: 'opr_retirement_token',
    Permission.ADMIN: 'graphiti_admin_token',
}

_JWT_SCOPE_ATTR = {
    Permission.READ: 'opr_jwt_read_scope',
    Permission.WRITE: 'opr_jwt_write_scope',
    Permission.RECONCILE: 'opr_jwt_reconciliation_scope',
    Permission.RETIRE: 'opr_jwt_retirement_scope',
    Permission.ADMIN: 'opr_jwt_admin_scope',
}

_STATIC_FORBIDDEN_DETAIL = {
    Permission.READ: 'OPR graph read is not authorized',
    Permission.WRITE: 'OPR graph write is not authorized',
    Permission.RECONCILE: 'retired episode reconciliation is not authorized',
    Permission.RETIRE: 'conditional episode retirement is not authorized',
    Permission.ADMIN: 'Graphiti administrative access is not authorized',
}


class _InvalidAccessToken(Exception):
    pass


class _JwksUnavailable(Exception):
    pass


@dataclass(frozen=True)
class _VerificationKey:
    key: Any
    algorithm: str


def _parse_bearer(authorization: str | None) -> str:
    if not authorization:
        raise _InvalidAccessToken
    scheme, separator, token = authorization.partition(' ')
    if (
        separator != ' '
        or scheme.casefold() != 'bearer'
        or not is_http_token68(token)
        or len(token.encode('ascii')) > _MAX_BEARER_TOKEN_BYTES
    ):
        raise _InvalidAccessToken
    return token


def _token_scopes(value: Any) -> set[str]:
    """Normalize the LZ collection claim and the RFC space-delimited form."""
    if isinstance(value, str):
        scopes = value.split(' ')
        if not value or any(not scope for scope in scopes):
            raise _InvalidAccessToken
        return set(scopes)
    if (
        isinstance(value, list)
        and value
        and all(isinstance(scope, str) and scope for scope in value)
    ):
        return set(value)
    raise _InvalidAccessToken


class JwtVerifier:
    """Validate LZ access tokens against a bounded, rotating JWKS cache."""

    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._allowed_client_ids = frozenset(settings.opr_jwt_allowed_client_ids.split(','))
        self._clock = clock
        self._client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(_JWKS_HTTP_TIMEOUT_SECONDS),
            follow_redirects=False,
        )
        self._owns_client = http_client is None
        self._keys: dict[str, _VerificationKey] = {}
        self._refresh_lock = asyncio.Lock()
        self._last_refresh_attempt = float('-inf')
        self._last_successful_refresh = float('-inf')
        self._refresh_after = float('-inf')

    async def start(self) -> None:
        """Prime keys before the pod can become ready."""
        await self._refresh(startup=True, force=True)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def verify(self, token: str, required_scope: str) -> Mapping[str, Any]:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise _InvalidAccessToken from exc

        kid = header.get('kid')
        algorithm = header.get('alg')
        critical = header.get('crit')
        if (
            not isinstance(kid, str)
            or not kid
            or algorithm != self._settings.opr_jwt_algorithm
            or critical not in (None, [])
        ):
            raise _InvalidAccessToken

        verification_key = await self._key_for(kid)
        try:
            claims = self._decode(token, verification_key)
        except jwt.InvalidSignatureError:
            # A signing-key rotation may reuse a kid. Refresh at most once, with
            # the same cooldown that prevents random-kid requests from turning
            # into an unbounded outbound-request primitive.
            refreshed = await self._refresh(force=True)
            if not refreshed:
                raise _InvalidAccessToken from None
            verification_key = self._keys.get(kid)
            if verification_key is None:
                raise _InvalidAccessToken from None
            try:
                claims = self._decode(token, verification_key)
            except jwt.PyJWTError as exc:
                raise _InvalidAccessToken from exc
        except jwt.PyJWTError as exc:
            raise _InvalidAccessToken from exc

        if (
            claims.get('gty') != 'client-credentials'
            or claims.get('azp') not in self._allowed_client_ids
        ):
            raise _InvalidAccessToken
        if required_scope not in _token_scopes(claims.get('scope')):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail='Bearer token lacks the required Graphiti scope',
                headers={
                    'WWW-Authenticate': (
                        f'Bearer error="insufficient_scope", scope="{required_scope}"'
                    )
                },
            )
        return claims

    def _decode(self, token: str, verification_key: _VerificationKey) -> Mapping[str, Any]:
        if verification_key.algorithm != self._settings.opr_jwt_algorithm:
            raise jwt.InvalidAlgorithmError('JWK algorithm is not allowed')
        return jwt.decode(
            token,
            key=verification_key.key,
            algorithms=[self._settings.opr_jwt_algorithm],
            audience=self._settings.opr_jwt_audience,
            issuer=self._settings.opr_jwt_issuer,
            leeway=self._settings.opr_jwt_clock_skew_seconds,
            options={
                'require': ['iss', 'aud', 'sub', 'azp', 'exp', 'scope', 'gty'],
                'verify_exp': True,
                'verify_nbf': True,
                'verify_iat': True,
            },
        )

    async def _key_for(self, kid: str) -> _VerificationKey:
        now = self._clock()
        cache_expired = (
            bool(self._keys)
            and now - self._last_successful_refresh > self._settings.opr_jwks_max_stale_seconds
        )
        if cache_expired:
            # A failed refresh can place the next attempt beyond max-stale.
            # Respect that outbound-request cooldown, but never let it extend
            # the lifetime of cached verification keys.
            await self._refresh(force=True)
        elif now >= self._refresh_after:
            await self._refresh()

        if (
            self._keys
            and self._clock() - self._last_successful_refresh
            > self._settings.opr_jwks_max_stale_seconds
        ):
            raise _JwksUnavailable('LZ JWKS cached keys exceeded max-stale')

        key = self._keys.get(kid)
        if key is not None:
            return key

        await self._refresh(force=True)
        key = self._keys.get(kid)
        if key is None:
            raise _InvalidAccessToken
        return key

    async def _refresh(self, *, startup: bool = False, force: bool = False) -> bool:
        async with self._refresh_lock:
            now = self._clock()
            if not force and self._keys and now < self._refresh_after:
                return False
            if (
                force
                and self._keys
                and now - self._last_refresh_attempt
                < self._settings.opr_jwks_refresh_min_interval_seconds
            ):
                return False

            self._last_refresh_attempt = now
            try:
                keys = await self._fetch_keys()
            except (httpx.HTTPError, ValueError, jwt.PyJWTError) as exc:
                cache_age = now - self._last_successful_refresh
                if (
                    not startup
                    and self._keys
                    and cache_age <= self._settings.opr_jwks_max_stale_seconds
                ):
                    # Preserve availability through a bounded issuer outage, but
                    # retry no faster than the configured cooldown. The request
                    # is still cryptographically verified by a previously valid
                    # public key; once max-stale elapses, authorization closes.
                    self._refresh_after = now + self._settings.opr_jwks_refresh_min_interval_seconds
                    logger.warning('LZ JWKS refresh failed; using bounded stale verification keys')
                    return False
                raise _JwksUnavailable('LZ JWKS could not be refreshed') from exc

            self._keys = keys
            self._last_successful_refresh = now
            self._refresh_after = now + self._settings.opr_jwks_cache_ttl_seconds
            return True

    async def _fetch_keys(self) -> dict[str, _VerificationKey]:
        async with self._client.stream('GET', self._settings.opr_jwt_jwks_url) as response:
            if 300 <= response.status_code < 400:
                raise ValueError('JWKS redirects are not allowed')
            response.raise_for_status()

            declared_length = response.headers.get('content-length')
            if declared_length is not None:
                try:
                    declared_bytes = int(declared_length)
                except ValueError as exc:
                    raise ValueError('JWKS Content-Length is invalid') from exc
                if declared_bytes < 0:
                    raise ValueError('JWKS Content-Length is invalid')
                if declared_bytes > _MAX_JWKS_BYTES:
                    raise ValueError('JWKS response is too large')

            body = bytearray()
            async for chunk in response.aiter_bytes(chunk_size=_JWKS_READ_CHUNK_BYTES):
                if len(body) + len(chunk) > _MAX_JWKS_BYTES:
                    raise ValueError('JWKS response is too large')
                body.extend(chunk)

        payload = json.loads(body)
        if not isinstance(payload, dict) or not isinstance(payload.get('keys'), list):
            raise ValueError('JWKS response must contain a keys list')
        raw_keys = payload['keys']
        if not raw_keys or len(raw_keys) > _MAX_JWKS_KEYS:
            raise ValueError('JWKS contains an invalid number of keys')

        keys: dict[str, _VerificationKey] = {}
        for raw_key in raw_keys:
            if not isinstance(raw_key, dict):
                raise ValueError('JWKS keys must be objects')
            kid = raw_key.get('kid')
            if not isinstance(kid, str) or not kid:
                raise ValueError('every JWK must have a non-empty kid')
            if kid in keys:
                raise ValueError('JWKS contains duplicate kid values')
            if raw_key.get('kty') != 'RSA':
                continue
            if raw_key.get('use') not in (None, 'sig'):
                continue
            key_ops = raw_key.get('key_ops')
            if key_ops is not None and (not isinstance(key_ops, list) or 'verify' not in key_ops):
                continue
            algorithm = raw_key.get('alg', self._settings.opr_jwt_algorithm)
            if algorithm != self._settings.opr_jwt_algorithm:
                continue
            jwk = jwt.PyJWK.from_dict(raw_key, algorithm=self._settings.opr_jwt_algorithm)
            keys[kid] = _VerificationKey(key=jwk.key, algorithm=algorithm)

        if not keys:
            raise ValueError('JWKS contains no allowed signing keys')
        return keys


class GraphitiAuthorizer:
    """Dispatch one explicit authentication mode without credential fallback."""

    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._auth_mode = OprAuthMode(getattr(settings, 'opr_auth_mode', OprAuthMode.STATIC))
        self._jwt_verifier = (
            JwtVerifier(settings, http_client=http_client, clock=clock)
            if self._auth_mode is OprAuthMode.LZ_JWT
            else None
        )

    async def start(self) -> None:
        if self._jwt_verifier is not None:
            try:
                await self._jwt_verifier.start()
            except _JwksUnavailable as exc:
                raise RuntimeError('cannot start JWT authorization without LZ JWKS') from exc

    async def close(self) -> None:
        if self._jwt_verifier is not None:
            await self._jwt_verifier.close()

    async def require(
        self,
        permission: Permission,
        authorization: str | None,
        *,
        legacy_token: str | None = None,
    ) -> None:
        if self._auth_mode is OprAuthMode.STATIC:
            expected = getattr(self._settings, _STATIC_SECRET_ATTR[permission]).get_secret_value()
            if permission in {Permission.RECONCILE, Permission.RETIRE}:
                authorized = reconciliation_token_matches(expected, legacy_token)
            else:
                authorized = bearer_token_matches(expected, authorization)
            if not authorized:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=_STATIC_FORBIDDEN_DETAIL[permission],
                )
            return

        if self._jwt_verifier is None:
            raise RuntimeError('JWT auth mode has no verifier')
        try:
            token = _parse_bearer(authorization)
            await self._jwt_verifier.verify(
                token, getattr(self._settings, _JWT_SCOPE_ATTR[permission])
            )
        except _InvalidAccessToken as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='Bearer access token is missing or invalid',
                headers={'WWW-Authenticate': 'Bearer error="invalid_token"'},
            ) from exc
        except _JwksUnavailable as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail='Authorization verification keys are unavailable',
            ) from exc


def build_graphiti_authorizer(settings: Settings) -> GraphitiAuthorizer:
    return GraphitiAuthorizer(settings)


def set_graphiti_authorizer(app: FastAPI, authorizer: GraphitiAuthorizer) -> None:
    setattr(app.state, AUTHORIZER_STATE_ATTR, authorizer)


def graphiti_authorizer_from_app(app: FastAPI) -> GraphitiAuthorizer:
    authorizer = getattr(app.state, AUTHORIZER_STATE_ATTR, None)
    if authorizer is None:
        raise RuntimeError(
            f'no Graphiti authorizer on app.state.{AUTHORIZER_STATE_ATTR}: '
            'application lifespan startup did not run'
        )
    return authorizer


async def get_graphiti_authorizer(http_request: Request) -> GraphitiAuthorizer:
    return graphiti_authorizer_from_app(http_request.app)


GraphitiAuthorizerDep = Annotated[GraphitiAuthorizer, Depends(get_graphiti_authorizer)]
