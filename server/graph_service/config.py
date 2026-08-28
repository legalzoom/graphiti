import hmac
from functools import lru_cache
from typing import Annotated

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
MAX_INGEST_DRAIN_TIMEOUT_SECONDS = 15.0


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
    opr_read_token: SecretStr = SecretStr('')
    opr_write_token: SecretStr = SecretStr('')
    opr_reconciliation_token: SecretStr = SecretStr('')
    opr_retirement_token: SecretStr = SecretStr('')
    opr_writer_fleet_epoch: SecretStr = SecretStr('')
    graphiti_admin_token: SecretStr = SecretStr('')
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
    def require_distinct_privileged_tokens(self):
        secrets = {
            'OPR_READ_TOKEN': self.opr_read_token.get_secret_value(),
            'OPR_WRITE_TOKEN': self.opr_write_token.get_secret_value(),
            'OPR_RECONCILIATION_TOKEN': self.opr_reconciliation_token.get_secret_value(),
            'OPR_RETIREMENT_TOKEN': self.opr_retirement_token.get_secret_value(),
            'OPR_WRITER_FLEET_EPOCH': self.opr_writer_fleet_epoch.get_secret_value(),
            'GRAPHITI_ADMIN_TOKEN': self.graphiti_admin_token.get_secret_value(),
        }

        if self.opr_auth_required:
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

        if self.opr_auth_required:
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
