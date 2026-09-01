import hmac
import re
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from functools import lru_cache
from typing import Annotated, Literal
from urllib.parse import urlsplit

from fastapi import Depends
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore

from graph_service.protocol import is_http_token68

_PRIVILEGED_SECRET_NAMES = (
    'OPR_READ_TOKEN',
    'OPR_WRITE_TOKEN',
    'OPR_RECONCILIATION_TOKEN',
    'OPR_RETIREMENT_TOKEN',
    'OPR_WRITER_FLEET_EPOCH',
    'GRAPHITI_ADMIN_TOKEN',
)
_MIN_PRIVILEGED_SECRET_BYTES = 32
_OAUTH_SCOPE_TOKEN = re.compile(r'[\x21\x23-\x5B\x5D-\x7E]+', re.ASCII)
_CLIENT_ID_TOKEN = re.compile(r'[\x21-\x2B\x2D-\x7E]+', re.ASCII)
MAX_INGEST_DRAIN_TIMEOUT_SECONDS = 15.0
_OPR_DEV_LEGACY_AUTH_COMPATIBILITY_MAX_DAYS = 14


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


class OprAuthMode(str, Enum):
    """Authentication envelope used for the protected OPR graph."""

    STATIC = 'static'
    LZ_JWT = 'lz_jwt'


class Settings(BaseSettings):
    openai_api_key: str
    openai_base_url: str | None = Field(None)
    model_name: str | None = Field(None)
    embedding_model_name: str | None = Field(None)
    neo4j_uri: str | None = Field(None)
    neo4j_user: str | None = Field(None)
    neo4j_password: str | None = Field(None)
    falkordb_host: str | None = Field(None)
    falkordb_port: int | None = Field(None)
    falkordb_database: str | None = Field(None)
    neptune_host: str | None = Field(None)
    neptune_port: int | None = Field(None)
    aoss_host: str | None = Field(None)
    aoss_port: int | None = Field(None)
    db_backend: str = Field('neo4j')
    # Opt-in for downstream deployments that use the protected OPR graph.
    # Keeping the default false preserves the upstream graph service for users
    # that never address the OPR-owned group.
    opr_auth_required: bool = False
    # Temporary incident bridge for the REST service in DEV only. This is a
    # separate, expiring switch so OPR_AUTH_REQUIRED keeps its existing
    # fail-closed runtime meaning.
    graphiti_deployment_environment: str = ''
    opr_dev_legacy_auth_compatibility_enabled: bool = False
    opr_dev_legacy_auth_compatibility_remove_by: date | None = None
    # Static remains the default so upstream and existing deployments preserve
    # their current behavior. A deployment must opt in explicitly before any
    # request is interpreted as an LZ Authorization Service access token.
    opr_auth_mode: OprAuthMode = OprAuthMode.STATIC
    opr_read_token: SecretStr = SecretStr('')
    opr_write_token: SecretStr = SecretStr('')
    opr_reconciliation_token: SecretStr = SecretStr('')
    opr_retirement_token: SecretStr = SecretStr('')
    opr_writer_fleet_epoch: SecretStr = SecretStr('')
    graphiti_admin_token: SecretStr = SecretStr('')
    opr_jwt_issuer: str = ''
    opr_jwt_jwks_url: str = ''
    opr_jwt_audience: str = ''
    opr_jwt_allowed_client_ids: str = ''
    opr_jwt_read_scope: str = ''
    opr_jwt_write_scope: str = ''
    opr_jwt_reconciliation_scope: str = ''
    opr_jwt_retirement_scope: str = ''
    opr_jwt_admin_scope: str = ''
    # LegalZoom Authorization Service currently signs access tokens with
    # RS256. Keeping the accepted algorithm closed at configuration parsing
    # prevents an environment change from enabling symmetric/none algorithms.
    opr_jwt_algorithm: Literal['RS256'] = 'RS256'
    opr_jwt_clock_skew_seconds: float = Field(default=30.0, ge=0, le=120)
    opr_jwks_cache_ttl_seconds: float = Field(default=300.0, gt=0, le=3600)
    opr_jwks_max_stale_seconds: float = Field(default=3600.0, gt=0, le=86400)
    opr_jwks_refresh_min_interval_seconds: float = Field(default=5.0, ge=1, le=300)
    graphiti_admin_clear_enabled: bool = False
    ingest_queue_maxsize: int = Field(gt=0)
    # LegalZoom's 60-second pod grace includes a platform-managed 30-second
    # preStop hook. The image then gives Uvicorn 3 seconds for active requests
    # and bounds graph-client close at 5 seconds. Keep the queue drain at or
    # below 15 seconds so cancellation and scheduler overhead retain margin
    # before Kubernetes sends SIGKILL.
    ingest_drain_timeout_seconds: float = Field(
        default=MAX_INGEST_DRAIN_TIMEOUT_SECONDS,
        gt=0,
        le=MAX_INGEST_DRAIN_TIMEOUT_SECONDS,
    )

    model_config = SettingsConfigDict(
        env_file='.env',
        extra='ignore',
        hide_input_in_errors=True,
    )

    @model_validator(mode='after')
    def validate_dev_legacy_auth_compatibility(self):
        if not self.opr_dev_legacy_auth_compatibility_enabled:
            return self

        if self.graphiti_deployment_environment != 'dev':
            raise ValueError(
                'OPR_DEV_LEGACY_AUTH_COMPATIBILITY_ENABLED=true requires '
                'GRAPHITI_DEPLOYMENT_ENVIRONMENT=dev'
            )
        if self.opr_auth_required:
            raise ValueError(
                'OPR_DEV_LEGACY_AUTH_COMPATIBILITY_ENABLED=true requires OPR_AUTH_REQUIRED=false'
            )
        if self.opr_auth_mode is not OprAuthMode.STATIC:
            raise ValueError(
                'OPR_DEV_LEGACY_AUTH_COMPATIBILITY_ENABLED=true is only valid with '
                'OPR_AUTH_MODE=static'
            )

        remove_by = self.opr_dev_legacy_auth_compatibility_remove_by
        if remove_by is None:
            raise ValueError(
                'OPR_DEV_LEGACY_AUTH_COMPATIBILITY_ENABLED=true requires '
                'OPR_DEV_LEGACY_AUTH_COMPATIBILITY_REMOVE_BY'
            )

        today = _utc_today()
        if remove_by <= today:
            raise ValueError('OPR_DEV_LEGACY_AUTH_COMPATIBILITY_REMOVE_BY must be a future date')
        if remove_by > today + timedelta(days=_OPR_DEV_LEGACY_AUTH_COMPATIBILITY_MAX_DAYS):
            raise ValueError(
                'OPR_DEV_LEGACY_AUTH_COMPATIBILITY_REMOVE_BY must be no more than '
                f'{_OPR_DEV_LEGACY_AUTH_COMPATIBILITY_MAX_DAYS} days from startup'
            )
        return self

    @model_validator(mode='after')
    def require_distinct_privileged_tokens(self):
        secrets = {
            'OPR_READ_TOKEN': self.opr_read_token.get_secret_value(),
            'OPR_WRITE_TOKEN': self.opr_write_token.get_secret_value(),
            'OPR_RECONCILIATION_TOKEN': self.opr_reconciliation_token.get_secret_value(),
            'OPR_RETIREMENT_TOKEN': self.opr_retirement_token.get_secret_value(),
            'OPR_WRITER_FLEET_EPOCH': self.opr_writer_fleet_epoch.get_secret_value(),
            'GRAPHITI_ADMIN_TOKEN': self.graphiti_admin_token.get_secret_value(),
        }

        if self.opr_auth_required and self.opr_auth_mode is OprAuthMode.STATIC:
            missing = [name for name in _PRIVILEGED_SECRET_NAMES if not secrets[name]]
            if missing:
                raise ValueError(
                    'OPR_AUTH_REQUIRED=true requires non-empty privileged values: '
                    + ', '.join(missing)
                )

        invalid_for_http = [
            name
            for name in _PRIVILEGED_SECRET_NAMES
            if secrets[name] and not is_http_token68(secrets[name])
        ]
        if invalid_for_http:
            raise ValueError(
                'privileged credentials require HTTP token68-compatible ASCII values: '
                + ', '.join(invalid_for_http)
            )

        if self.opr_auth_required and self.opr_auth_mode is OprAuthMode.STATIC:
            too_short = [
                name
                for name in _PRIVILEGED_SECRET_NAMES
                if len(secrets[name].encode('utf-8')) < _MIN_PRIVILEGED_SECRET_BYTES
            ]
            if too_short:
                raise ValueError(
                    'OPR_AUTH_REQUIRED=true requires privileged values of at least '
                    f'{_MIN_PRIVILEGED_SECRET_BYTES} bytes: ' + ', '.join(too_short)
                )

        # Preserve the pre-existing safety check for deployments that opt in
        # to the writer epoch without enabling the complete OPR auth profile.
        writer_fleet_epoch = secrets['OPR_WRITER_FLEET_EPOCH']
        if (
            writer_fleet_epoch
            and len(writer_fleet_epoch.encode('utf-8')) < _MIN_PRIVILEGED_SECRET_BYTES
        ):
            raise ValueError(
                f'OPR_WRITER_FLEET_EPOCH must be at least {_MIN_PRIVILEGED_SECRET_BYTES} bytes'
            )

        if self.opr_auth_mode is OprAuthMode.LZ_JWT:
            if not self.opr_auth_required:
                raise ValueError('OPR_AUTH_MODE=lz_jwt requires OPR_AUTH_REQUIRED=true')

            jwt_values = {
                'OPR_JWT_ISSUER': self.opr_jwt_issuer,
                'OPR_JWT_JWKS_URL': self.opr_jwt_jwks_url,
                'OPR_JWT_AUDIENCE': self.opr_jwt_audience,
                'OPR_JWT_ALLOWED_CLIENT_IDS': self.opr_jwt_allowed_client_ids,
                'OPR_JWT_READ_SCOPE': self.opr_jwt_read_scope,
                'OPR_JWT_WRITE_SCOPE': self.opr_jwt_write_scope,
                'OPR_JWT_RECONCILIATION_SCOPE': self.opr_jwt_reconciliation_scope,
                'OPR_JWT_RETIREMENT_SCOPE': self.opr_jwt_retirement_scope,
                'OPR_JWT_ADMIN_SCOPE': self.opr_jwt_admin_scope,
            }
            missing = [name for name, value in jwt_values.items() if not value]
            if not writer_fleet_epoch:
                missing.append('OPR_WRITER_FLEET_EPOCH')
            if missing:
                raise ValueError(
                    'OPR_AUTH_MODE=lz_jwt requires non-empty values: ' + ', '.join(missing)
                )

            for name, value in (
                ('OPR_JWT_ISSUER', self.opr_jwt_issuer),
                ('OPR_JWT_JWKS_URL', self.opr_jwt_jwks_url),
            ):
                parsed = urlsplit(value)
                if (
                    parsed.scheme != 'https'
                    or not parsed.hostname
                    or parsed.username is not None
                    or parsed.password is not None
                    or parsed.fragment
                ):
                    raise ValueError(f'{name} must be an HTTPS URL without credentials or fragment')

            if self.opr_jwks_max_stale_seconds < self.opr_jwks_cache_ttl_seconds:
                raise ValueError(
                    'OPR_JWKS_MAX_STALE_SECONDS must be greater than or equal to '
                    'OPR_JWKS_CACHE_TTL_SECONDS'
                )

            client_ids = self.opr_jwt_allowed_client_ids.split(',')
            if any(
                not client_id
                or not _CLIENT_ID_TOKEN.fullmatch(client_id)
                or client_id != client_id.strip()
                for client_id in client_ids
            ) or len(set(client_ids)) != len(client_ids):
                raise ValueError(
                    'OPR_JWT_ALLOWED_CLIENT_IDS must contain distinct comma-separated '
                    'visible ASCII client IDs without whitespace'
                )

            scope_values = {
                'OPR_JWT_READ_SCOPE': self.opr_jwt_read_scope,
                'OPR_JWT_WRITE_SCOPE': self.opr_jwt_write_scope,
                'OPR_JWT_RECONCILIATION_SCOPE': self.opr_jwt_reconciliation_scope,
                'OPR_JWT_RETIREMENT_SCOPE': self.opr_jwt_retirement_scope,
                'OPR_JWT_ADMIN_SCOPE': self.opr_jwt_admin_scope,
            }
            invalid_scopes = [
                name
                for name, value in scope_values.items()
                if value and not _OAUTH_SCOPE_TOKEN.fullmatch(value)
            ]
            if invalid_scopes:
                raise ValueError(
                    'JWT scope settings must each contain one RFC 6749 scope token: '
                    + ', '.join(invalid_scopes)
                )
            if len(set(scope_values.values())) != len(scope_values):
                raise ValueError('JWT permission scopes must be distinct')

        configured = [(name, value.encode('utf-8')) for name, value in secrets.items() if value]
        for index, (left_name, left_value) in enumerate(configured):
            for right_name, right_value in configured[index + 1 :]:
                if hmac.compare_digest(left_value, right_value):
                    raise ValueError(
                        f'privileged tokens must be distinct: {left_name} and {right_name}'
                    )
        return self


@lru_cache
def get_settings():
    return Settings()  # type: ignore[call-arg]


ZepEnvDep = Annotated[Settings, Depends(get_settings)]
