import asyncio
import logging
import os
from contextlib import asynccontextmanager
from functools import partial

from fastapi import APIRouter, FastAPI, HTTPException, status
from graphiti_core.errors import NodeNotFoundError  # type: ignore
from graphiti_core.nodes import EpisodeType, EpisodicNode  # type: ignore
from graphiti_core.utils.maintenance.graph_data_operations import clear_data  # type: ignore

from graph_service.dto import AddEntityNodeRequest, AddMessagesRequest, Message, Result
from graph_service.zep_graphiti import ZepGraphitiDep

logger = logging.getLogger('uvicorn.error')

DEFAULT_INGEST_QUEUE_MAX = 200


def _ingest_queue_maxsize() -> int:
    """Read the bounded queue capacity from the environment.

    Read directly rather than through graph_service.config.Settings: Settings
    requires database and LLM credentials that are irrelevant to sizing this
    queue, and this value is needed at import time to construct the
    module-level worker, before any request-scoped Settings dependency runs.
    """
    raw = os.environ.get('GRAPHITI_INGEST_QUEUE_MAX')
    if not raw:
        return DEFAULT_INGEST_QUEUE_MAX
    try:
        maxsize = int(raw)
    except ValueError as exc:
        raise RuntimeError(f'GRAPHITI_INGEST_QUEUE_MAX must be an integer, got {raw!r}') from exc
    if maxsize < 1:
        # asyncio.Queue treats maxsize <= 0 as unbounded, which would
        # silently disable backpressure; fail loudly instead.
        raise RuntimeError(f'GRAPHITI_INGEST_QUEUE_MAX must be >= 1, got {raw!r}')
    return maxsize


class AsyncWorker:
    def __init__(self, maxsize: int = DEFAULT_INGEST_QUEUE_MAX):
        self.queue = asyncio.Queue(maxsize=maxsize)
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
        # Queued jobs are discarded on shutdown; clients hold a durable retry
        # queue on their side and are expected to resend anything unconfirmed.
        discarded = self.queue.qsize()
        if discarded:
            logger.warning('discarding %d queued job(s) on shutdown', discarded)
        while not self.queue.empty():
            self.queue.get_nowait()


async_worker = AsyncWorker(maxsize=_ingest_queue_maxsize())


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
        # Idempotency guard: an episodic node is persisted only after the
        # full extraction pipeline succeeds, so uuid-exists means fully
        # processed and the LLM re-extraction can be skipped. Scope limits:
        # messages without a uuid are not covered (callers wanting dedup
        # must send a stable uuid), and a retry racing a not-yet-persisted
        # first attempt on another replica can still double-extract; that
        # window is the extraction latency (seconds), while client retries
        # arrive minutes later from their durable queue.
        if m.uuid:
            try:
                await EpisodicNode.get_by_uuid(graphiti.driver, m.uuid)
            except NodeNotFoundError:
                pass  # not yet ingested, proceed with extraction
            except Exception:
                # Ingest availability wins over the idempotency guard: if the
                # existence check itself fails, extract rather than drop the episode.
                logger.warning(
                    'episode existence check failed, proceeding with add_episode: uuid=%s',
                    m.uuid,
                    exc_info=True,
                )
            else:
                logger.info('episode already ingested, skipping re-extraction: uuid=%s', m.uuid)
                return

        await graphiti.add_episode(
            uuid=m.uuid,
            group_id=request.group_id,
            name=m.name,
            episode_body=f'{m.role or ""}({m.role_type}): {m.content}',
            reference_time=m.timestamp,
            source=EpisodeType.message,
            source_description=m.source_description,
        )

    queued = 0
    for m in request.messages:
        try:
            async_worker.queue.put_nowait(partial(add_messages_task, m))
            queued += 1
        except asyncio.QueueFull:
            rejected = len(request.messages) - queued
            logger.warning(
                'ingest queue full, rejecting message(s): queued=%d rejected=%d',
                queued,
                rejected,
            )
            # The detail reports counts only, not per-message uuids: clients
            # retry the whole batch on any non-2xx, and the idempotency
            # guard absorbs the already-queued portion on the retry.
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f'ingest queue full: queued {queued} of {len(request.messages)} messages, '
                    f'rejected {rejected}; retry the rejected messages later'
                ),
            ) from None

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


@router.post('/clear', status_code=status.HTTP_200_OK)
async def clear(
    graphiti: ZepGraphitiDep,
):
    await clear_data(graphiti.driver)
    await graphiti.build_indices_and_constraints()
    return Result(message='Graph cleared', success=True)
