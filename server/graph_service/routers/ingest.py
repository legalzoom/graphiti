import asyncio
import logging
from collections.abc import Callable, Coroutine
from contextlib import asynccontextmanager, suppress
from functools import partial
from importlib.metadata import version
from typing import Annotated, Any

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request, status
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.errors import NodeGroupMismatchError
from graphiti_core.helpers import query_result_record_count
from graphiti_core.nodes import EpisodeType  # type: ignore
from graphiti_core.utils.maintenance.graph_data_operations import clear_data  # type: ignore
from starlette.responses import JSONResponse

from graph_service.config import ZepEnvDep, get_settings
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
from graph_service.zep_graphiti import ZepGraphitiDep, graphiti_client_from_app

logger = logging.getLogger(__name__)

# A queued job: a zero-argument callable returning a coroutine, produced by
# `partial(add_messages_task, m)` below.
Job = Callable[[], Coroutine[Any, Any, None]]
_MAX_WORKER_CANCEL_WAIT_SECONDS = 1.0


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
    def __init__(self, maxsize: int | None = None, drain_timeout_seconds: float = 25.0):
        self._queue: asyncio.Queue[Job] | None = None
        self.task: asyncio.Task[None] | None = None
        # True exactly while `worker()` is inside `await job()`. A job in this
        # state has already left the queue (so `queue.qsize()` cannot count
        # it) but has not finished, so it still needs to be counted as
        # dropped if shutdown interrupts it.
        self._job_in_flight = False
        self._accepting = False
        self._draining = False
        self._drain_timeout_seconds = drain_timeout_seconds
        self._last_shutdown_dropped_jobs = 0
        if maxsize is not None:
            self.configure(maxsize, drain_timeout_seconds)

    def configure(self, maxsize: int, drain_timeout_seconds: float = 25.0) -> None:
        """Bind this worker to a bounded queue of the given size.

        Called once, at lifespan startup, with the size resolved from
        `Settings.ingest_queue_maxsize` and its shutdown budget. Not called at
        import time: the module-level `async_worker` singleton below is
        otherwise unconfigured, and using it before `configure()` raises
        rather than silently assuming an unbounded queue.
        """
        if self.task is not None and not self.task.done():
            raise RuntimeError('cannot reconfigure a running AsyncWorker')
        if maxsize <= 0:
            raise ValueError('AsyncWorker maxsize must be positive')
        if not 0 < drain_timeout_seconds <= 50:
            raise ValueError('AsyncWorker drain timeout must be greater than 0 and at most 50')
        self._queue = asyncio.Queue(maxsize=maxsize)
        self.task = None
        self._job_in_flight = False
        self._accepting = False
        self._draining = False
        self._drain_timeout_seconds = drain_timeout_seconds
        self._last_shutdown_dropped_jobs = 0

    @property
    def queue(self) -> asyncio.Queue[Job]:
        if self._queue is None:
            raise RuntimeError('AsyncWorker.configure() must be called before use')
        return self._queue

    @property
    def depth(self) -> int:
        return self.queue.qsize()

    @property
    def capacity(self) -> int:
        return self.queue.maxsize

    @property
    def accepting(self) -> bool:
        """Whether the producer may atomically add another batch."""
        worker_alive = self.task is not None and not self.task.done()
        return self._accepting and not self._draining and worker_alive

    @property
    def draining(self) -> bool:
        return self._draining

    @property
    def ready(self) -> bool:
        """Whether the worker can accept and consume application traffic."""
        return self.accepting and self.task is not None and not self.task.done()

    @property
    def last_shutdown_dropped_jobs(self) -> int:
        return self._last_shutdown_dropped_jobs

    async def worker(self):
        while True:
            try:
                job = await self.queue.get()
                self._job_in_flight = True
                try:
                    await job()
                except Exception:
                    # A failing job must not take the consumer down with it.
                    # Before this, any non-CancelledError exception (an
                    # embedder timeout, a graph driver error) propagated out of
                    # `while True` and ended the worker task for good. Nothing
                    # logged it, because the task is only ever awaited in
                    # `stop()`, so the exception sat unretrieved and the only
                    # symptom was ingest going quiet while `/messages` kept
                    # accepting work -- a dead consumer against a live producer
                    # is exactly how this queue grew until the pod was
                    # OOMKilled. Log the whole traceback and keep draining.
                    #
                    # `asyncio.CancelledError` derives from BaseException, so
                    # it is not caught here: it unwinds through the `finally`
                    # below to the loop's own handler, which breaks.
                    logger.exception(
                        f'Ingest worker job raised; dropping that job and continuing to '
                        f'drain the queue (depth={self.queue.qsize()}, '
                        f'maxsize={self.queue.maxsize}).'
                    )
                finally:
                    self._job_in_flight = False
                    # Every successful `get()` must be paired with exactly one
                    # `task_done()`, even when the job fails or cancellation
                    # interrupts it. Shutdown's `queue.join()` relies on this
                    # accounting to know that every accepted job has finished.
                    self.queue.task_done()
            except asyncio.CancelledError:
                break

    def _log_worker_exit(self, task: asyncio.Task[None]) -> None:
        """Surface a worker task that died instead of letting it go unnoticed.

        `worker()` above no longer exits on a job failure, but if it ever does
        exit unexpectedly the queue has no consumer at all and every later
        `/messages` call fills it until it rejects with 503. `stop()` is the
        only other place the task is awaited, and on a crashed worker that
        await may never happen, so retrieve and log the exception here.
        """
        self._accepting = False
        if task.cancelled() or self._draining:
            return
        exc = task.exception()
        if exc is not None:
            logger.critical(
                'Ingest worker task exited unexpectedly; the ingest queue now has no '
                'consumer and will fill until /messages rejects every batch.',
                exc_info=exc,
            )
        else:
            logger.critical(
                'Ingest worker task exited unexpectedly without an exception; '
                'the ingest queue will reject new work.'
            )

    async def start(self):
        if self.task is not None and not self.task.done():
            raise RuntimeError('AsyncWorker is already running')
        # Accessing the property fails fast if startup forgot to configure the
        # bounded queue before opening admission.
        _ = self.queue
        self._draining = False
        self._accepting = False
        self._last_shutdown_dropped_jobs = 0
        self.task = asyncio.create_task(self.worker())
        self.task.add_done_callback(self._log_worker_exit)
        self._accepting = True

    def begin_drain(self) -> None:
        """Close admission synchronously before shutdown first yields."""
        self._accepting = False
        self._draining = True

    def _outstanding_jobs(self) -> int:
        return self.queue.qsize() + (1 if self._job_in_flight else 0)

    def _discard_queued_jobs(self) -> int:
        discarded = 0
        while True:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                return discarded
            else:
                discarded += 1
                self.queue.task_done()

    async def _cancel_worker(self) -> bool:
        """Request cancellation without letting a hostile job block shutdown.

        Python tasks may suppress ``CancelledError``. Waiting on such a task
        without a second bound would turn the configured drain timeout into an
        unbounded shutdown. Return false after a short grace and let process
        termination provide the final isolation boundary.
        """
        if self.task is None or self.task.done():
            return True
        self.task.cancel()
        cancel_wait = min(self._drain_timeout_seconds, _MAX_WORKER_CANCEL_WAIT_SECONDS)
        done, _ = await asyncio.wait({self.task}, timeout=cancel_wait)
        if self.task in done:
            # A task cancelled before its coroutine takes its first turn does
            # not reach `worker()`'s own CancelledError handler.
            with suppress(asyncio.CancelledError):
                self.task.result()
            return True

        logger.critical(
            f'Ingest worker did not stop within {cancel_wait:g}s after cancellation; '
            'returning from shutdown so the process termination deadline remains bounded.'
        )
        # Make one final non-blocking request. Never await it here: code inside
        # a job can suppress repeated cancellations too.
        self.task.cancel()
        return False

    async def stop(self) -> int:
        """Stop admission, drain accepted work, then stop the consumer.

        Returns the number of accepted jobs interrupted or discarded by this
        shutdown. Ordinary job failures are logged when they happen and remain
        the durable producer's reconciliation responsibility. A nonzero
        shutdown result is logged at CRITICAL because this queue is in-memory.
        """
        self.begin_drain()
        self._last_shutdown_dropped_jobs = 0

        worker_running = self.task is not None and not self.task.done()
        if worker_running:
            try:
                await asyncio.wait_for(
                    self.queue.join(),
                    timeout=self._drain_timeout_seconds,
                )
            except asyncio.TimeoutError:
                dropped = self._outstanding_jobs()
                self._last_shutdown_dropped_jobs = dropped
                logger.critical(
                    f'Ingest shutdown drain timed out after '
                    f'{self._drain_timeout_seconds:g}s with {dropped} '
                    'unprocessed job(s); cancelling the active job and dropping '
                    'queued jobs. Producer reconciliation is required.'
                )
                await self._cancel_worker()
                self._discard_queued_jobs()
                return dropped

            await self._cancel_worker()
            return 0

        dropped = self._outstanding_jobs()
        if dropped:
            self._last_shutdown_dropped_jobs = dropped
            logger.critical(
                f'Ingest shutdown cannot drain because its worker is not running; '
                f'dropping {dropped} unprocessed job(s). Producer reconciliation '
                'is required.'
            )
            self._discard_queued_jobs()
        return dropped


async_worker = AsyncWorker()


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    async_worker.configure(
        settings.ingest_queue_maxsize,
        settings.ingest_drain_timeout_seconds,
    )
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
            'description': (
                'The worker is unavailable, ingestion is draining, or the queue is at capacity; '
                'retry later.'
            ),
        },
    },
)
async def add_messages(
    request: AddMessagesRequest,
    http_request: Request,
    graphiti: ZepGraphitiDep,
    settings: ZepEnvDep,
    authorization: Annotated[str | None, Header()] = None,
) -> Result | JSONResponse:
    _authorize_opr_write(settings, authorization, request.group_id)

    def _admission_closed_response() -> JSONResponse:
        depth = async_worker.depth
        maxsize = async_worker.capacity
        draining = async_worker.draining
        error = 'ingest_draining' if draining else 'ingest_worker_unavailable'
        message = (
            'Ingestion is draining for shutdown; retry another instance.'
            if draining
            else 'The ingestion worker is unavailable; retry another instance.'
        )
        logger.warning(
            f'Rejecting message batch for group_id={request.group_id!r}: '
            f'ingest admission is closed ({error}; depth={depth}, maxsize={maxsize}). '
            'Producer is expected to retry another instance.'
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                'success': False,
                'error': error,
                'message': message,
                'queue_depth': depth,
                'queue_maxsize': maxsize,
            },
        )

    # Check before UUID ownership I/O so a request that starts after shutdown
    # does not spend part of the finite termination grace doing work that
    # cannot be admitted.
    if not async_worker.accepting:
        return _admission_closed_response()

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

    # The ownership checks above await graph I/O. Shutdown may have closed
    # admission while this request was suspended, so check again here. There
    # is no await between this gate and the batch's `put_nowait` calls, making
    # the admission decision and enqueue atomic on this event loop.
    if not async_worker.accepting:
        return _admission_closed_response()

    # Capture the application, not the request-scoped client, and not the
    # request body: a queued job must pin as little as possible, because it can
    # sit in the queue for as long as the queue is deep.
    app = http_request.app
    group_id = request.group_id

    async def add_messages_task(m: Message):
        # Resolve the shared client when the job runs, not when it is queued.
        # Closing over the injected `graphiti` value was two bugs at once: it
        # pinned a whole per-request client stack (three AsyncOpenAI clients
        # with their own connection pools, plus a graph driver) inside the
        # queue, and `/messages` returns 202 before the job runs, so the old
        # per-request dependency had already closed that driver by the time the
        # job called into it.
        client = graphiti_client_from_app(app)
        await client.add_episode(
            uuid=m.uuid,
            group_id=group_id,
            name=m.name,
            episode_body=f'{m.role or ""}({m.role_type}): {m.content}',
            reference_time=m.timestamp,
            source=EpisodeType.message,
            source_description=m.source_description,
        )

    def _queue_full_response() -> JSONResponse:
        depth = async_worker.depth
        maxsize = async_worker.capacity
        logger.warning(
            f'Rejecting message batch for group_id={request.group_id!r}: '
            f'ingest queue is full (depth={depth}, maxsize={maxsize}). '
            'Producer is expected to retry.'
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
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
    # would requeue those already-accepted messages a second time on retry.
    # Messages without a caller-supplied uuid have no dedup key, so a partial
    # enqueue here would become a duplicate episode there. No `await` runs
    # between this check and the final `put_nowait` below, so nothing else on
    # this event loop can change queue occupancy in between: the check and the
    # enqueue are effectively one atomic step.
    if async_worker.depth + len(request.messages) > async_worker.capacity:
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
