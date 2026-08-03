from datetime import datetime, timezone

from fastapi import APIRouter, status
from graphiti_core.search.search_config import SearchConfig
from graphiti_core.search.search_config import SearchResults as CoreSearchResults
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

from graph_service.dto import (
    FactResult,
    GetMemoryRequest,
    GetMemoryResponse,
    Message,
    SearchQuery,
    SearchResults,
)
from graph_service.zep_graphiti import ZepGraphitiDep, get_fact_result_from_edge

router = APIRouter()


def _rrf_search_config(max_facts: int) -> SearchConfig:
    """Build the edge search config these routes use, with limit applied.

    Deep-copied because EDGE_HYBRID_SEARCH_RRF is a module-level singleton and
    assigning limit on it would mutate shared state for every other caller in
    the process. Graphiti.search() has this bug; these routes do not inherit it.

    RRF over BM25 + cosine candidate lists. This is the same config
    Graphiti.search() selects when center_node_uuid is None, so ranking here is
    unchanged from before scores were plumbed through.
    """
    config = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    config.limit = max_facts
    return config


def _facts_with_scores(results: CoreSearchResults) -> list[FactResult]:
    """Zip edges with their reranker scores.

    edge_reranker_scores is a parallel list to edges. Indexed defensively
    rather than with zip() so that a short or empty score list degrades to
    score 0.0 instead of silently dropping facts from the response.
    """
    scores = results.edge_reranker_scores
    return [
        get_fact_result_from_edge(edge, scores[i] if i < len(scores) else 0.0)
        for i, edge in enumerate(results.edges)
    ]


@router.post('/search', status_code=status.HTTP_200_OK)
async def search(query: SearchQuery, graphiti: ZepGraphitiDep):
    # search_() rather than search(): the latter returns list[EntityEdge] and
    # discards SearchResults.edge_reranker_scores, leaving consumers no way to
    # rank or threshold results.
    results = await graphiti.search_(
        query=query.query,
        config=_rrf_search_config(query.max_facts),
        group_ids=query.group_ids,
    )
    return SearchResults(
        facts=_facts_with_scores(results),
    )


@router.get('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def get_entity_edge(uuid: str, graphiti: ZepGraphitiDep):
    entity_edge = await graphiti.get_entity_edge(uuid)
    return get_fact_result_from_edge(entity_edge)


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
    # request.center_node_uuid is deliberately not forwarded: the previous
    # implementation did not forward it either, and passing it would switch the
    # reranker to node-distance and change this endpoint's ranking. Scores are
    # the only intended change here.
    results = await graphiti.search_(
        query=combined_query,
        config=_rrf_search_config(request.max_facts),
        group_ids=[request.group_id],
    )
    return GetMemoryResponse(facts=_facts_with_scores(results))


def compose_query_from_messages(messages: list[Message]):
    combined_query = ''
    for message in messages:
        combined_query += f'{message.role_type or ""}({message.role or ""}): {message.content}\n'
    return combined_query
