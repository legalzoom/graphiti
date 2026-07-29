from datetime import datetime, timezone

from fastapi import APIRouter, status
from graphiti_core.errors import NodeNotFoundError  # type: ignore
from graphiti_core.nodes import EpisodicNode  # type: ignore

from graph_service.dto import (
    EpisodeStatus,
    GetMemoryRequest,
    GetMemoryResponse,
    Message,
    SearchQuery,
    SearchResults,
)
from graph_service.zep_graphiti import (
    ZepGraphitiDep,
    get_fact_result_from_edge,
    resolve_episode_names,
)

router = APIRouter()


@router.post('/search', status_code=status.HTTP_200_OK)
async def search(query: SearchQuery, graphiti: ZepGraphitiDep):
    relevant_edges = await graphiti.search(
        group_ids=query.group_ids,
        query=query.query,
        num_results=query.max_facts,
    )
    episode_names = await resolve_episode_names(graphiti.driver, relevant_edges)
    facts = [get_fact_result_from_edge(edge, episode_names) for edge in relevant_edges]
    return SearchResults(
        facts=facts,
    )


@router.get('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def get_entity_edge(uuid: str, graphiti: ZepGraphitiDep):
    entity_edge = await graphiti.get_entity_edge(uuid)
    episode_names = await resolve_episode_names(graphiti.driver, [entity_edge])
    return get_fact_result_from_edge(entity_edge, episode_names)


@router.get('/episodes/status/{uuid}', status_code=status.HTTP_200_OK)
async def episode_status(uuid: str, graphiti: ZepGraphitiDep):
    """Has this episode been processed into the graph?

    The durability check for queued /messages ingestion: writers that
    received a 202 can confirm the episode actually exists instead of
    trusting the in-memory queue. Only a missing node maps to
    exists=false; any other failure propagates as an error.
    """
    try:
        episode = await EpisodicNode.get_by_uuid(graphiti.driver, uuid)
    except NodeNotFoundError:
        return EpisodeStatus(uuid=uuid, exists=False, name=None)
    return EpisodeStatus(uuid=uuid, exists=True, name=episode.name)


@router.get('/episodes/{group_id}', status_code=status.HTTP_200_OK)
async def get_episodes(group_id: str, last_n: int, graphiti: ZepGraphitiDep):
    episodes = await graphiti.retrieve_episodes(
        group_ids=[group_id], last_n=last_n, reference_time=datetime.now(timezone.utc)
    )
    return episodes


@router.post('/get-memory', status_code=status.HTTP_200_OK)
async def get_memory(
    request: GetMemoryRequest,
    graphiti: ZepGraphitiDep,
):
    combined_query = compose_query_from_messages(request.messages)
    result = await graphiti.search(
        group_ids=[request.group_id],
        query=combined_query,
        num_results=request.max_facts,
    )
    episode_names = await resolve_episode_names(graphiti.driver, result)
    facts = [get_fact_result_from_edge(edge, episode_names) for edge in result]
    return GetMemoryResponse(facts=facts)


def compose_query_from_messages(messages: list[Message]):
    combined_query = ''
    for message in messages:
        combined_query += f'{message.role_type or ""}({message.role or ""}): {message.content}\n'
    return combined_query
