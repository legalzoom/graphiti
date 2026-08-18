import asyncio
import logging
from contextlib import asynccontextmanager
from functools import partial
from importlib.metadata import version
from typing import Annotated

from fastapi import APIRouter, FastAPI, Header, HTTPException, status
from graphiti_core.nodes import EpisodeType  # type: ignore
from graphiti_core.utils.maintenance.graph_data_operations import clear_data  # type: ignore
from starlette.responses import JSONResponse

from graph_service.config import ZepEnvDep
from graph_service.dto import (
    AddEntityNodeRequest,
    AddMessagesRequest,
    DeleteEpisodeIfMatchRequest,
    Message,
    Result,
)
from graph_service.protocol import (
    GRAPHITI_RECONCILIATION_GROUP_ID,
    GRAPHITI_RECONCILIATION_OPERATION_RETIRE_EPISODE,
    GRAPHITI_RECONCILIATION_PROTOCOL,
    reconciliation_token_matches,
)
from graph_service.zep_graphiti import ZepGraphitiDep

logger = logging.getLogger('uvicorn.error')


def _authorize_episode_retirement(
    settings: ZepEnvDep,
    supplied_token: str | None,
    operation: str | None,
    group_id: str,
) -> None:
    expected_token = settings.opr_retirement_token.get_secret_value()
    if not reconciliation_token_matches(expected_token, supplied_token) or (
        operation != GRAPHITI_RECONCILIATION_OPERATION_RETIRE_EPISODE
        or group_id != GRAPHITI_RECONCILIATION_GROUP_ID
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='conditional episode retirement is not authorized',
        )


class AsyncWorker:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.task = None

    async def worker(self):
        while True:
            try:
                job = await self.queue.get()
                logger.info('Processing job (remaining in queue: %d)', self.queue.qsize())
                await job()
                logger.info('Job completed successfully')
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception('Job failed')

    async def start(self):
        self.task = asyncio.create_task(self.worker())

    async def stop(self):
        if self.task:
            self.task.cancel()
            await self.task
        while not self.queue.empty():
            self.queue.get_nowait()


async_worker = AsyncWorker()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await async_worker.start()
    yield
    await async_worker.stop()


router = APIRouter(lifespan=lifespan)


@router.post('/messages', status_code=status.HTTP_202_ACCEPTED)
async def add_messages(
    request: AddMessagesRequest,
    graphiti: ZepGraphitiDep,
):
    async def add_messages_task(m: Message):
        await graphiti.add_episode(
            uuid=m.uuid,
            group_id=request.group_id,
            name=m.name,
            episode_body=f'{m.role or ""}({m.role_type}): {m.content}',
            reference_time=m.timestamp,
            source=EpisodeType.message,
            source_description=m.source_description,
        )

    for m in request.messages:
        await async_worker.queue.put(partial(add_messages_task, m))

    return Result(message='Messages added to processing queue', success=True)


@router.post('/entity-node', status_code=status.HTTP_201_CREATED)
async def add_entity_node(
    request: AddEntityNodeRequest,
    graphiti: ZepGraphitiDep,
):
    node = await graphiti.save_entity_node(
        uuid=request.uuid,
        group_id=request.group_id,
        name=request.name,
        summary=request.summary,
    )
    return node


@router.delete('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def delete_entity_edge(uuid: str, graphiti: ZepGraphitiDep):
    await graphiti.delete_entity_edge(uuid)
    return Result(message='Entity Edge deleted', success=True)


@router.delete('/group/{group_id}', status_code=status.HTTP_200_OK)
async def delete_group(group_id: str, graphiti: ZepGraphitiDep):
    await graphiti.delete_group(group_id)
    return Result(message='Group deleted', success=True)


@router.delete('/episode/{uuid}', status_code=status.HTTP_200_OK)
async def delete_episode(uuid: str, graphiti: ZepGraphitiDep):
    await graphiti.delete_episodic_node(uuid)
    return Result(message='Episode deleted', success=True)


@router.delete('/episode/{uuid}/retire/v3', status_code=status.HTTP_200_OK)
async def delete_episode_if_matches(
    uuid: str,
    request: DeleteEpisodeIfMatchRequest,
    graphiti: ZepGraphitiDep,
    settings: ZepEnvDep,
    x_opr_retirement_token: Annotated[str | None, Header()] = None,
    x_opr_reconciliation_operation: Annotated[str | None, Header()] = None,
):
    _authorize_episode_retirement(
        settings,
        x_opr_retirement_token,
        x_opr_reconciliation_operation,
        request.group_id,
    )
    deleted = await graphiti.delete_episodic_node_if_matches(
        uuid,
        group_id=request.group_id,
        name=request.name,
        content=request.content,
        source_description=request.source_description,
        retirement_request_id=str(request.retirement_request_id),
    )
    if deleted is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail='Episode retirement request receipt conflicts with durable state',
        )
    if deleted is False:
        return JSONResponse(
            status_code=status.HTTP_412_PRECONDITION_FAILED,
            content={
                'message': 'Episode identity precondition failed',
                'success': False,
                'outcome': 'not_applied',
                'reconciliation_protocol': GRAPHITI_RECONCILIATION_PROTOCOL,
                'graphiti_core_version': version('graphiti-core'),
                'retirement_request_id': str(request.retirement_request_id),
            },
        )
    return {
        'message': 'Episode conditionally deleted',
        'success': True,
        'outcome': 'retired',
        'reconciliation_protocol': GRAPHITI_RECONCILIATION_PROTOCOL,
        'graphiti_core_version': version('graphiti-core'),
        'retirement_request_id': str(request.retirement_request_id),
    }


@router.post('/episode/{uuid}/retirement/v3', status_code=status.HTTP_200_OK)
async def get_episode_retirement_status(
    uuid: str,
    retirement_request_id: str,
    group_id: str,
    graphiti: ZepGraphitiDep,
    settings: ZepEnvDep,
    x_opr_retirement_token: Annotated[str | None, Header()] = None,
    x_opr_reconciliation_operation: Annotated[str | None, Header()] = None,
):
    _authorize_episode_retirement(
        settings,
        x_opr_retirement_token,
        x_opr_reconciliation_operation,
        group_id,
    )
    outcome = await graphiti.episode_retirement_outcome(
        uuid,
        group_id=group_id,
        retirement_request_id=retirement_request_id,
    )
    if outcome is None:
        raise HTTPException(
            status_code=status.HTTP_412_PRECONDITION_FAILED,
            detail='Episode retirement receipt is not durable',
        )
    return {
        'message': 'Episode retirement outcome is durable',
        'success': outcome == 'retired',
        'outcome': outcome,
        'reconciliation_protocol': GRAPHITI_RECONCILIATION_PROTOCOL,
        'graphiti_core_version': version('graphiti-core'),
        'retirement_request_id': retirement_request_id,
    }


@router.post('/clear', status_code=status.HTTP_200_OK)
async def clear(
    graphiti: ZepGraphitiDep,
):
    await clear_data(graphiti.driver)
    await graphiti.build_indices_and_constraints()
    return Result(message='Graph cleared', success=True)
