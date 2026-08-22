import hmac
from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore


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
    opr_read_token: SecretStr = SecretStr('')
    opr_write_token: SecretStr = SecretStr('')
    opr_reconciliation_token: SecretStr = SecretStr('')
    opr_retirement_token: SecretStr = SecretStr('')
    opr_writer_fleet_epoch: SecretStr = SecretStr('')
    graphiti_admin_token: SecretStr = SecretStr('')
    graphiti_admin_clear_enabled: bool = False
    ingest_queue_maxsize: int = Field(gt=0)

    model_config = SettingsConfigDict(
        env_file='.env',
        extra='ignore',
        hide_input_in_errors=True,
    )

    @model_validator(mode='after')
    def require_distinct_privileged_tokens(self):
        writer_fleet_epoch = self.opr_writer_fleet_epoch.get_secret_value()
        if writer_fleet_epoch and len(writer_fleet_epoch.encode('utf-8')) < 32:
            raise ValueError('OPR_WRITER_FLEET_EPOCH must be at least 32 bytes')
        tokens = {
            'OPR_READ_TOKEN': self.opr_read_token.get_secret_value(),
            'OPR_WRITE_TOKEN': self.opr_write_token.get_secret_value(),
            'OPR_RECONCILIATION_TOKEN': self.opr_reconciliation_token.get_secret_value(),
            'OPR_RETIREMENT_TOKEN': self.opr_retirement_token.get_secret_value(),
            'OPR_WRITER_FLEET_EPOCH': writer_fleet_epoch,
            'GRAPHITI_ADMIN_TOKEN': self.graphiti_admin_token.get_secret_value(),
        }
        configured = [(name, value) for name, value in tokens.items() if value]
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
