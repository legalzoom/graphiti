from uuid import UUID

from graphiti_core.nodes import EpisodeType
from pydantic import BaseModel, Field

from graph_service.dto.common import Message


class AddMessagesRequest(BaseModel):
    group_id: str = Field(..., description='The group id of the messages to add')
    messages: list[Message] = Field(..., description='The messages to add')


class AddEntityNodeRequest(BaseModel):
    uuid: str = Field(..., description='The uuid of the node to add')
    group_id: str = Field(..., description='The group id of the node to add')
    name: str = Field(..., description='The name of the node to add')
    summary: str = Field(default='', description='The summary of the node to add')


class DeleteEpisodeIfMatchRequest(BaseModel):
    """Complete episode identity required for an atomic conditional delete."""

    group_id: str = Field(..., min_length=1, description='The owning graph group')
    name: str = Field(..., description='The exact stored episode name')
    content: str = Field(..., description='The exact stored episode content')
    source: EpisodeType = Field(..., description='The exact stored episode source type')
    source_description: str = Field(
        ...,
        description='The exact stored producer provenance',
    )
    retirement_request_id: UUID = Field(
        ...,
        description='The durable OPR idempotency key for this retirement',
    )
