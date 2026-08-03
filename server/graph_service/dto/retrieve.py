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
    score: float = Field(
        default=0.0,
        description=(
            'Reranker score from the search that produced this fact. Higher is more '
            'relevant. Scale depends on the configured reranker: /search and /get-memory '
            'use RRF with rank_const=1, so a fact ranked first in one candidate list '
            'scores 1.0, second 0.5, third 0.333, and appearing in both the BM25 and '
            'cosine lists sums the two contributions (max 2.0). NOT a cosine similarity '
            'and not comparable to one. Defaults to 0.0 on paths that do not rank, such '
            'as fetching a single edge by uuid. That default is only unambiguous while '
            'these routes use RRF, whose contributions are strictly positive: a '
            'cross-encoder or MMR reranker can emit 0.0 or negative scores legitimately, '
            'so switching reranker means revisiting this sentinel.'
        ),
    )
    source_node_uuid: str = Field(description='uuid of the entity node this fact originates from')
    target_node_uuid: str = Field(description='uuid of the entity node this fact points to')
    source_node: str | None = Field(
        default=None,
        description=(
            'Name of the source entity node, resolved from source_node_uuid. None when '
            'the node could not be looked up. Names are what callers need to join facts '
            'against their own entity records; uuids alone are not portable.'
        ),
    )
    target_node: str | None = Field(
        default=None,
        description='Name of the target entity node, resolved from target_node_uuid.',
    )

    class Config:
        json_encoders = {datetime: lambda v: v.astimezone(timezone.utc).isoformat()}


class SearchResults(BaseModel):
    facts: list[FactResult]


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
