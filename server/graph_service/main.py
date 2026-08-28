import asyncio
import logging
from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from graph_service.config import get_settings
from graph_service.dto import ReadinessResponse
from graph_service.routers import ingest, retrieve
from graph_service.zep_graphiti import (
    GRAPHITI_CLIENT_STATE_ATTR,
    ZepGraphiti,
    build_graphiti_client,
    set_graphiti_client,
)

logger = logging.getLogger(__name__)
_GRAPHITI_CLIENT_CLOSE_TIMEOUT_SECONDS = 5.0


def _log_late_client_close(task: asyncio.Task[None]) -> None:
    """Retrieve a late close result after bounded application shutdown returns."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.critical(
            'Graphiti client close failed after the application shutdown deadline.',
            exc_info=exc,
        )


async def _close_graphiti_client(client: ZepGraphiti) -> None:
    """Bound client cleanup so Kubernetes retains time to terminate cleanly."""
    close_task = asyncio.create_task(client.close())
    done, _ = await asyncio.wait(
        {close_task},
        timeout=_GRAPHITI_CLIENT_CLOSE_TIMEOUT_SECONDS,
    )
    if close_task in done:
        close_task.result()
        return

    logger.critical(
        'Graphiti client did not close within %.1fs; cancelling cleanup so the pod '
        'termination deadline remains bounded.',
        _GRAPHITI_CLIENT_CLOSE_TIMEOUT_SECONDS,
    )
    close_task.add_done_callback(_log_late_client_close)
    close_task.cancel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # One Graphiti client for the whole process, owned here and installed on
    # app state. Request dependencies and queued ingest jobs both resolve this
    # same instance; nothing constructs a client per request.
    client = build_graphiti_client(settings)
    set_graphiti_client(app, client)
    try:
        await client.build_indices_and_constraints()
        yield
    finally:
        # The only place this client is closed. The ingest router's lifespan is
        # merged inside this one by `include_router`, so it has already closed
        # admission and completed its bounded drain/cancellation attempt. A job
        # that deliberately suppresses cancellation remains in doubt until the
        # process exits; bound client cleanup too so it cannot consume the rest
        # of the pod's termination deadline.
        await _close_graphiti_client(client)


app = FastAPI(lifespan=lifespan)


app.include_router(retrieve.router)
app.include_router(ingest.router)


@app.get('/healthcheck')
async def healthcheck():
    return JSONResponse(
        content={
            'status': 'healthy',
            'graphiti_core_version': version('graphiti-core'),
            'ingest_queue_depth': ingest.async_worker.depth,
            'ingest_queue_maxsize': ingest.async_worker.capacity,
        },
        status_code=200,
    )


@app.get(
    '/readyz',
    response_model=ReadinessResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {'model': ReadinessResponse}},
)
async def readiness(request: Request):
    """Report whether lifecycle prerequisites for application traffic are ready.

    Settings are validated before the application lifespan starts, so a
    deployment with required-but-missing OPR credentials never reaches this
    endpoint. Once started, fail readiness if either the shared graph client
    was not installed or the ingest consumer has exited. Queue capacity is a
    transient backpressure signal returned by `/messages`, not pod readiness.
    """
    settings = get_settings()
    worker_task = ingest.async_worker.task
    graphiti_ready = hasattr(request.app.state, GRAPHITI_CLIENT_STATE_ATTR)
    worker_running = worker_task is not None and not worker_task.done()
    ingest_ready = ingest.async_worker.ready
    ready = graphiti_ready and ingest_ready
    return JSONResponse(
        content={
            'status': 'ready' if ready else 'not_ready',
            'graphiti_core_version': version('graphiti-core'),
            'opr_auth_required': settings.opr_auth_required,
            'ingest_worker_running': worker_running,
            'ingest_accepting': ingest.async_worker.accepting,
            'ingest_draining': ingest.async_worker.draining,
        },
        status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
    )
