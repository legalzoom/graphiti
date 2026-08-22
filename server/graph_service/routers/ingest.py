import asyncio
import logging
import os
from contextlib import asynccontextmanager
from functools import partial
from importlib.metadata import version
from typing import Annotated

from fastapi import APIRouter, FastAPI, Header, HTTPException, status
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.errors import NodeGroupMismatchError
from graphiti_core.helpers import query_result_record_count
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
    bearer_token_matches,
    reconciliation_token_matches,
    writer_fleet_epoch_sha256,
)
from graph_service.zep_graphiti import ZepGraphitiDep

logger = logging.getLogger(__name__)

# The worker drains one job at a time (see AsyncWorker.worker below), so a job
# that is mid-flight when a caller is rejected may still take several seconds
# to clear (an episode add is an LLM round trip, not a cheap local write).
# This is a hint, not a promise of drain time -- callers should back off with
# jitter rather than hammer on exactly this cadence.
INGEST_QUEUE_RETRY_AFTER_SECONDS = 5


def _required_positive_int_env(name: str) -> int:
    """Read a required positive-integer environment variable.

    No default is applied. An unset or malformed value is a deploy-time
    misconfiguration -- it must fail the process at startup, not silently pick
    a number nobody chose.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == '':
        raise RuntimeError(f'{name} must be set to a positive integer')
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f'{name} must be a positive integer, got {raw!r}') from exc
    if value <= 0:
        raise RuntimeError(f'{name} must be a positive integer, got {raw!r}')
    return value


INGEST_QUEUE_MAXSIZE = _required_positive_int_env('INGEST_QUEUE_MAXSIZE')


def _authorize_opr_write(
    settings: ZepEnvDep,
    authorization: str | None,
    group_id: str,
) -> None:
    if group_id == GRAPHITI_RECONCILIATION_GROUP_ID and not bearer_token_matches(
        settings.opr_write_token.get_secret_value(), authorization
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='OPR graph write is not authorized',
        )


def _authorize_graphiti_admin(settings: ZepEnvDep, authorization: str | None) -> None:
    if not bearer_token_matches(settings.graphiti_admin_token.get_secret_value(), authorization):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Graphiti administrative access is not authorized',
        )


def _authorize_episode_retirement(
    settings: ZepEnvDep,
    supplied_token: str | None,
    supplied_writer_fleet_epoch: str | None,
    operation: str | None,
    group_id: str,
) -> None:
    expected_token = settings.opr_retirement_token.get_secret_value()
    expected_writer_fleet_epoch = settings.opr_writer_fleet_epoch.get_secret_value()
    if (
        not reconciliation_token_matches(expected_token, supplied_token)
        or not reconciliation_token_matches(
            expected_writer_fleet_epoch, supplied_writer_fleet_epoch
        )
        or (
            operation != GRAPHITI_RECONCILIATION_OPERATION_RETIRE_EPISODE
            or group_id != GRAPHITI_RECONCILIATION_GROUP_ID
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='conditional episode retirement is not authorized',
        )


class AsyncWorker:
    def __init__(self, maxsize: int | None = None):
        self.queue: asyncio.Queue = asyncio.Queue(
            maxsize=maxsize if maxsize is not None else INGEST_QUEUE_MAXSIZE
        )
        self.task: asyncio.Task | None = None

    async def worker(self):
        while True:
            try:
                job = await self.queue.get()
                await job()
            except asyncio.CancelledError:
                break

    async def start(self):
        self.task = asyncio.create_task(self.worker())

    async def stop(self):
        if self.task:
            self.task.cancel()
            await self.task
        dropped = self.queue.qsize()
        if dropped:
            # The queue is in-memory only: this is the known, accepted
            # restart-loss behavior (durable retry lives in the OPR outbox
            # producer, not here). Not silent -- logged so an operator sees
            # exactly how many jobs a restart discarded.
            logger.warning(
                f'Dropping {dropped} unprocessed job(s) from the in-memory ingest '
                'queue on shutdown; this queue is not durable across restarts.'
            )
        while not self.queue.empty():
            self.queue.get_nowait()


async_worker = AsyncWorker()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await async_worker.start()
    yield
    await async_worker.stop()


router = APIRouter(lifespan=lifespan)


@router.post(
    '/messages',
    status_code=status.HTTP_202_ACCEPTED,
    # The success and 503-rejection payloads are different shapes (Result vs.
    # a plain error dict), so there is no single Pydantic response_model to
    # infer from the `Result | JSONResponse` return annotation. Declare it
    # explicitly rather than let FastAPI fail at startup trying to build one.
    response_model=None,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            'description': 'Ingestion queue is at capacity; retry after the given delay.',
        },
    },
)
async def add_messages(
    request: AddMessagesRequest,
    graphiti: ZepGraphitiDep,
    settings: ZepEnvDep,
    authorization: Annotated[str | None, Header()] = None,
) -> Result | JSONResponse:
    _authorize_opr_write(settings, authorization, request.group_id)

    try:
        for message in request.messages:
            if message.uuid:
                await graphiti.assert_episode_uuid_group(
                    message.uuid,
                    request.group_id,
                )
    except NodeGroupMismatchError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail='episode UUID is already owned by another graph group',
        ) from exc

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

    def _queue_full_response() -> JSONResponse:
        depth = async_worker.queue.qsize()
        maxsize = async_worker.queue.maxsize
        logger.warning(
            f'Rejecting message batch for group_id={request.group_id!r}: '
            f'ingest queue is full (depth={depth}, maxsize={maxsize}). '
            'Producer is expected to retry.'
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={'Retry-After': str(INGEST_QUEUE_RETRY_AFTER_SECONDS)},
            content={
                'success': False,
                'error': 'ingest_queue_full',
                'message': 'Ingestion queue is at capacity; retry later.',
                'queue_depth': depth,
                'queue_maxsize': maxsize,
            },
        )

    # Reject the whole batch, not just the messages past the limit. Enqueueing
    # some of a batch and then telling the caller to retry the whole batch
    # would requeue those already-accepted messages a second time on retry;
    # messages without a caller-supplied uuid have no dedup key, so a partial
    # enqueue here would become a duplicate episode there. No `await` runs
    # between this check and the final `put_nowait` below, so nothing else on
    # this event loop can change queue occupancy in between: the check and the
    # enqueue are effectively one atomic step.
    if async_worker.queue.qsize() + len(request.messages) > async_worker.queue.maxsize:
        return _queue_full_response()

    for m in request.messages:
        try:
            async_worker.queue.put_nowait(partial(add_messages_task, m))
        except asyncio.QueueFull:
            # Should be unreachable given the capacity check above (this
            # endpoint is the queue's only producer). Fail loudly rather than
            # assume: something violated that single-producer assumption.
            raise RuntimeError(
                'ingest queue rejected a put_nowait despite the capacity check '
                'that should have prevented it; the single-producer invariant '
                'for this queue has been violated'
            ) from None

    return Result(message='Messages added to processing queue', success=True)


@router.post('/entity-node', status_code=status.HTTP_201_CREATED)
async def add_entity_node(
    request: AddEntityNodeRequest,
    graphiti: ZepGraphitiDep,
    settings: ZepEnvDep,
    authorization: Annotated[str | None, Header()] = None,
):
    _authorize_opr_write(settings, authorization, request.group_id)
    try:
        node = await graphiti.save_entity_node(
            uuid=request.uuid,
            group_id=request.group_id,
            name=request.name,
            summary=request.summary,
        )
    except NodeGroupMismatchError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail='entity UUID is already owned by another graph group',
        ) from exc
    return node


@router.delete('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def delete_entity_edge(
    uuid: str,
    graphiti: ZepGraphitiDep,
    settings: ZepEnvDep,
    authorization: Annotated[str | None, Header()] = None,
):
    _authorize_graphiti_admin(settings, authorization)
    await graphiti.delete_entity_edge(uuid)
    return Result(message='Entity Edge deleted', success=True)


@router.delete('/group/{group_id}', status_code=status.HTTP_200_OK)
async def delete_group(
    group_id: str,
    graphiti: ZepGraphitiDep,
    settings: ZepEnvDep,
    authorization: Annotated[str | None, Header()] = None,
):
    _authorize_graphiti_admin(settings, authorization)
    if group_id == GRAPHITI_RECONCILIATION_GROUP_ID:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='the OPR group requires request-bound episode retirement',
        )
    await graphiti.delete_group(group_id)
    return Result(message='Group deleted', success=True)


@router.delete('/episode/{uuid}', status_code=status.HTTP_200_OK)
async def delete_episode(
    uuid: str,
    graphiti: ZepGraphitiDep,
    settings: ZepEnvDep,
    authorization: Annotated[str | None, Header()] = None,
):
    _authorize_graphiti_admin(settings, authorization)
    if graphiti.driver.provider in {GraphProvider.KUZU, GraphProvider.FALKORDB}:
        # Kuzu cannot use the transient-property lock. FalkorDB stores groups
        # in separate graph databases, but this UUID-only route has no trusted
        # group with which to select one. Both must fail closed rather than run
        # a group guard in the wrong database or reintroduce a read/delete race.
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail='safe legacy episode deletion is unavailable for this backend',
        )
    # Keep the group check and deletion in one mutation query. A separate
    # lookup followed by the legacy UUID-only delete would reintroduce the
    # exact TOCTOU window that the v5 OPR retirement protocol closes.
    lock_clause = """
        SET episode._opr_conditional_delete_lock = true
        REMOVE episode._opr_conditional_delete_lock
        WITH episode
        """
    result = await graphiti.driver.execute_query(
        """
        MATCH (episode:Episodic {uuid: $uuid})
        """
        + lock_clause
        + """
        WHERE episode.group_id <> $opr_group_id
          AND coalesce(episode.opr_deleted, false) = false
        DETACH DELETE episode
        RETURN $uuid AS uuid
        """,
        uuid=uuid,
        opr_group_id=GRAPHITI_RECONCILIATION_GROUP_ID,
    )
    if await query_result_record_count(result) != 1:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='legacy episode deletion is not authorized for this episode',
        )
    return Result(message='Episode deleted', success=True)


@router.delete('/episode/{uuid}/retire/v5', status_code=status.HTTP_200_OK)
async def delete_episode_if_matches(
    uuid: str,
    request: DeleteEpisodeIfMatchRequest,
    graphiti: ZepGraphitiDep,
    settings: ZepEnvDep,
    x_opr_retirement_token: Annotated[str | None, Header()] = None,
    x_opr_writer_fleet_epoch: Annotated[str | None, Header()] = None,
    x_opr_reconciliation_operation: Annotated[str | None, Header()] = None,
):
    _authorize_episode_retirement(
        settings,
        x_opr_retirement_token,
        x_opr_writer_fleet_epoch,
        x_opr_reconciliation_operation,
        request.group_id,
    )
    deleted = await graphiti.delete_episodic_node_if_matches(
        uuid,
        group_id=request.group_id,
        name=request.name,
        content=request.content,
        source=request.source.value,
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
                'writer_fleet_epoch_sha256': writer_fleet_epoch_sha256(
                    settings.opr_writer_fleet_epoch.get_secret_value()
                ),
                'retirement_request_id': str(request.retirement_request_id),
            },
        )
    return {
        'message': 'Episode conditionally deleted',
        'success': True,
        'outcome': 'retired',
        'reconciliation_protocol': GRAPHITI_RECONCILIATION_PROTOCOL,
        'graphiti_core_version': version('graphiti-core'),
        'writer_fleet_epoch_sha256': writer_fleet_epoch_sha256(
            settings.opr_writer_fleet_epoch.get_secret_value()
        ),
        'retirement_request_id': str(request.retirement_request_id),
    }


@router.post('/episode/{uuid}/retirement/v5', status_code=status.HTTP_200_OK)
async def get_episode_retirement_status(
    uuid: str,
    retirement_request_id: str,
    group_id: str,
    graphiti: ZepGraphitiDep,
    settings: ZepEnvDep,
    x_opr_retirement_token: Annotated[str | None, Header()] = None,
    x_opr_writer_fleet_epoch: Annotated[str | None, Header()] = None,
    x_opr_reconciliation_operation: Annotated[str | None, Header()] = None,
):
    _authorize_episode_retirement(
        settings,
        x_opr_retirement_token,
        x_opr_writer_fleet_epoch,
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
        'writer_fleet_epoch_sha256': writer_fleet_epoch_sha256(
            settings.opr_writer_fleet_epoch.get_secret_value()
        ),
        'retirement_request_id': retirement_request_id,
    }


@router.post('/clear', status_code=status.HTTP_200_OK)
async def clear(
    graphiti: ZepGraphitiDep,
    settings: ZepEnvDep,
    authorization: Annotated[str | None, Header()] = None,
):
    _authorize_graphiti_admin(settings, authorization)
    if not settings.graphiti_admin_clear_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Graphiti administrative clear is disabled',
        )
    await clear_data(graphiti.driver)
    await graphiti.build_indices_and_constraints()
    return Result(message='Graph cleared', success=True)
