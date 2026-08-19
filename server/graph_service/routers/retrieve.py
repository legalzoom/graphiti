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
    bearer_token_matches,
    reconciliation_token_matches,
    writer_fleet_epoch_sha256,
)
from graph_service.zep_graphiti import ZepGraphitiDep, get_fact_result_from_edge

router = APIRouter()


def _authorize_opr_read(
    settings: ZepEnvDep,
    authorization: str | None,
    group_id: str,
) -> None:
    if group_id == GRAPHITI_RECONCILIATION_GROUP_ID and not bearer_token_matches(
        settings.opr_read_token.get_secret_value(), authorization
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='OPR graph read is not authorized',
        )


def _authorize_reconciliation_listing(
    settings: ZepEnvDep,
    supplied_token: str | None,
    supplied_writer_fleet_epoch: str | None,
    group_id: str,
) -> None:
    expected_token = settings.opr_reconciliation_token.get_secret_value()
    expected_writer_fleet_epoch = settings.opr_writer_fleet_epoch.get_secret_value()
    if (
        group_id != GRAPHITI_RECONCILIATION_GROUP_ID
        or not reconciliation_token_matches(expected_token, supplied_token)
        or not reconciliation_token_matches(
            expected_writer_fleet_epoch, supplied_writer_fleet_epoch
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='retired episode reconciliation is not authorized',
        )


@router.post('/search', status_code=status.HTTP_200_OK)
async def search(
    query: SearchQuery,
    graphiti: ZepGraphitiDep,
    settings: ZepEnvDep,
    authorization: Annotated[str | None, Header()] = None,
):
    # An omitted group list means an unrestricted search, which can include
    # OPR-owned data and therefore requires the OPR credential too.
    if (
        not query.group_ids
        or query.group_ids == ['']
        or GRAPHITI_RECONCILIATION_GROUP_ID in query.group_ids
    ):
        _authorize_opr_read(settings, authorization, GRAPHITI_RECONCILIATION_GROUP_ID)
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
async def get_entity_edge(
    uuid: str,
    graphiti: ZepGraphitiDep,
    settings: ZepEnvDep,
    authorization: Annotated[str | None, Header()] = None,
):
    entity_edge = await graphiti.get_entity_edge(uuid)
    _authorize_opr_read(settings, authorization, entity_edge.group_id)
    return get_fact_result_from_edge(entity_edge)


@router.get('/episodes/{group_id}', status_code=status.HTTP_200_OK)
async def get_episodes(
    group_id: str,
    last_n: int,
    graphiti: ZepGraphitiDep,
    settings: ZepEnvDep,
    authorization: Annotated[str | None, Header()] = None,
):
    _authorize_opr_read(settings, authorization, group_id)
    episodes = await graphiti.retrieve_episodes(
        group_ids=[group_id], last_n=last_n, reference_time=datetime.now(timezone.utc)
    )
    return episodes


@router.get('/episodes/{group_id}/reconciliation/v5', status_code=status.HTTP_200_OK)
async def get_episodes_for_reconciliation(
    group_id: str,
    last_n: int,
    graphiti: ZepGraphitiDep,
    settings: ZepEnvDep,
    x_opr_reconciliation_token: Annotated[str | None, Header()] = None,
    x_opr_writer_fleet_epoch: Annotated[str | None, Header()] = None,
):
    """Return the privileged ledger with an in-band capability attestation.

    This distinct route makes older servers fail with 404 rather than silently
    ignoring a query flag. The protocol and core version travel with the
    exact listing response, so a load balancer cannot split capability proof
    and destructive audit across different pods.
    """
    _authorize_reconciliation_listing(
        settings,
        x_opr_reconciliation_token,
        x_opr_writer_fleet_epoch,
        group_id,
    )
    episodes = await graphiti.retrieve_episodes_for_reconciliation(group_id, last_n)
    return {
        'reconciliation_protocol': GRAPHITI_RECONCILIATION_PROTOCOL,
        'graphiti_core_version': version('graphiti-core'),
        'writer_fleet_epoch_sha256': writer_fleet_epoch_sha256(
            settings.opr_writer_fleet_epoch.get_secret_value()
        ),
        'episodes': episodes,
    }


@router.post('/get-memory', status_code=status.HTTP_200_OK)
async def get_memory(
    request: GetMemoryRequest,
    graphiti: ZepGraphitiDep,
    settings: ZepEnvDep,
    authorization: Annotated[str | None, Header()] = None,
):
    _authorize_opr_read(settings, authorization, request.group_id)
    combined_query = compose_query_from_messages(request.messages)
    result = await graphiti.search(
        group_ids=[request.group_id],
        query=combined_query,
        center_node_uuid=request.center_node_uuid,
        num_results=request.max_facts,
    )
    facts = [get_fact_result_from_edge(edge) for edge in result]
    return GetMemoryResponse(facts=facts)


def compose_query_from_messages(messages: list[Message]):
    combined_query = ''
    for message in messages:
        combined_query += f'{message.role_type or ""}({message.role or ""}): {message.content}\n'
    return combined_query
