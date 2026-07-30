from datetime import datetime, timezone

from pydantic import BaseModel, Field

from graph_service.dto.common import Message


class SearchQuery(BaseModel):
    group_ids: list[str] | None = Field(
        None, description='The group ids for the memories to search'
    )
    query: str
    max_facts: int = Field(default=10, description='The maximum number of facts to retrieve')


class FactResult(BaseModel):
    uuid: str
    name: str
    fact: str
    valid_at: datetime | None
    invalid_at: datetime | None
    created_at: datetime
    expired_at: datetime | None
    # Provenance: the episodes this fact was extracted from. episodes is
    # the raw uuid list from the edge; episode_names maps each uuid that
    # still resolves to an Episodic node to that node's name (e.g.
    # "curated:{doc_path}"). A uuid absent from episode_names means the
    # episode node no longer exists in the graph.
    episodes: list[str]
    episode_names: dict[str, str]

    class Config:
        json_encoders = {datetime: lambda v: v.astimezone(timezone.utc).isoformat()}


class SearchResults(BaseModel):
    facts: list[FactResult]


class EpisodeStatus(BaseModel):
    """Answer to "has episode <uuid> been processed into the graph?".

    Lets fire-and-forget writers (queued /messages ingestion) verify
    durability instead of trusting a 202.
    """

    uuid: str
    exists: bool
    name: str | None


class GetMemoryRequest(BaseModel):
    group_id: str = Field(..., description='The group id of the memory to get')
    max_facts: int = Field(default=10, description='The maximum number of facts to retrieve')
    center_node_uuid: str | None = Field(
        ..., description='The uuid of the node to center the retrieval on'
    )
    messages: list[Message] = Field(
        ..., description='The messages to build the retrieval query from '
    )


class GetMemoryResponse(BaseModel):
    facts: list[FactResult] = Field(..., description='The facts that were retrieved from the graph')
