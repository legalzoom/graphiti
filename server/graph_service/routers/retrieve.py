from datetime import datetime, timezone
from importlib.metadata import version
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, status

from graph_service.config import ZepEnvDep
from graph_service.dto import (
    GetMemoryRequest,
    GetMemoryResponse,
    Message,
    SearchQuery,
    SearchResults,
)
from graph_service.protocol import (
    GRAPHITI_RECONCILIATION_GROUP_ID,
    GRAPHITI_RECONCILIATION_PROTOCOL,
    reconciliation_token_matches,
)
from graph_service.zep_graphiti import ZepGraphitiDep, get_fact_result_from_edge

router = APIRouter()


def _authorize_reconciliation_listing(
    settings: ZepEnvDep,
    supplied_token: str | None,
    group_id: str,
) -> None:
    expected_token = settings.opr_reconciliation_token.get_secret_value()
    if group_id != GRAPHITI_RECONCILIATION_GROUP_ID or not reconciliation_token_matches(
        expected_token, supplied_token
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='retired episode reconciliation is not authorized',
        )


@router.post('/search', status_code=status.HTTP_200_OK)
async def search(query: SearchQuery, graphiti: ZepGraphitiDep):
    relevant_edges = await graphiti.search(
        group_ids=query.group_ids,
        query=query.query,
        num_results=query.max_facts,
    )
    facts = [get_fact_result_from_edge(edge) for edge in relevant_edges]
    return SearchResults(
        facts=facts,
    )


@router.get('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def get_entity_edge(uuid: str, graphiti: ZepGraphitiDep):
    entity_edge = await graphiti.get_entity_edge(uuid)
    return get_fact_result_from_edge(entity_edge)


@router.get('/episodes/{group_id}', status_code=status.HTTP_200_OK)
async def get_episodes(
    group_id: str,
    last_n: int,
    graphiti: ZepGraphitiDep,
    settings: ZepEnvDep,
    include_retired_for_reconciliation: bool = False,
    x_opr_reconciliation_token: Annotated[str | None, Header()] = None,
):
    if include_retired_for_reconciliation:
        _authorize_reconciliation_listing(settings, x_opr_reconciliation_token, group_id)
        return await graphiti.retrieve_episodes_for_reconciliation(group_id, last_n)
    episodes = await graphiti.retrieve_episodes(
        group_ids=[group_id], last_n=last_n, reference_time=datetime.now(timezone.utc)
    )
    return episodes


@router.get('/episodes/{group_id}/reconciliation/v3', status_code=status.HTTP_200_OK)
async def get_episodes_for_reconciliation(
    group_id: str,
    last_n: int,
    graphiti: ZepGraphitiDep,
    settings: ZepEnvDep,
    x_opr_reconciliation_token: Annotated[str | None, Header()] = None,
):
    """Return the privileged ledger with an in-band capability attestation.

    This distinct route makes older servers fail with 404 rather than silently
    ignoring a query flag. The protocol and core version travel with the
    exact listing response, so a load balancer cannot split capability proof
    and destructive audit across different pods.
    """
    _authorize_reconciliation_listing(settings, x_opr_reconciliation_token, group_id)
    episodes = await graphiti.retrieve_episodes_for_reconciliation(group_id, last_n)
    return {
        'reconciliation_protocol': GRAPHITI_RECONCILIATION_PROTOCOL,
        'graphiti_core_version': version('graphiti-core'),
        'episodes': episodes,
    }


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
    facts = [get_fact_result_from_edge(edge) for edge in result]
    return GetMemoryResponse(facts=facts)


def compose_query_from_messages(messages: list[Message]):
    combined_query = ''
    for message in messages:
        combined_query += f'{message.role_type or ""}({message.role or ""}): {message.content}\n'
    return combined_query
