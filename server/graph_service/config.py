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
    kuzu_db: str | None = Field(None)
    kuzu_max_concurrent_queries: int | None = Field(None)
    db_backend: str = Field('neo4j')
    opr_reconciliation_token: SecretStr = SecretStr('')
    opr_retirement_token: SecretStr = SecretStr('')

    model_config = SettingsConfigDict(
        env_file='.env',
        extra='ignore',
        hide_input_in_errors=True,
    )

    @model_validator(mode='after')
    def require_distinct_opr_privileged_tokens(self):
        listing_token = self.opr_reconciliation_token.get_secret_value()
        retirement_token = self.opr_retirement_token.get_secret_value()
        if (
            listing_token
            and retirement_token
            and hmac.compare_digest(listing_token, retirement_token)
        ):
            raise ValueError('OPR reconciliation and retirement tokens must be distinct')
        return self


@lru_cache
def get_settings():
    return Settings()  # type: ignore[call-arg]


ZepEnvDep = Annotated[Settings, Depends(get_settings)]
