import asyncio
import logging
from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from graphiti_core.async_limiter import AsyncCapacityOverloadedError

from graph_service.auth import (
    AUTHORIZER_STATE_ATTR,
    build_graphiti_authorizer,
    set_graphiti_authorizer,
)
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
    authorizer = build_graphiti_authorizer(settings)
    client: ZepGraphiti | None = None
    try:
        # JWT mode eagerly resolves the configured LZ JWKS. A pod never starts
        # accepting traffic with an empty verification-key cache.
        await authorizer.start()
        set_graphiti_authorizer(app, authorizer)

        # One Graphiti client for the whole process, owned here and installed on
        # app state. Request dependencies and queued ingest jobs both resolve this
        # same instance; nothing constructs a client per request.
        client = build_graphiti_client(settings)
        set_graphiti_client(app, client)
        await client.build_indices_and_constraints()
        yield
    finally:
        try:
            if client is not None:
                # The only place this client is closed. The ingest router's lifespan is
                # merged inside this one by `include_router`, so it has already closed
                # admission and completed its bounded drain/cancellation attempt. A job
                # that deliberately suppresses cancellation remains in doubt until the
                # process exits; bound client cleanup too so it cannot consume the rest
                # of the pod's termination deadline.
                await _close_graphiti_client(client)
        finally:
            await authorizer.close()


app = FastAPI(lifespan=lifespan)


@app.exception_handler(AsyncCapacityOverloadedError)
async def async_capacity_overloaded(
    _request: Request, error: AsyncCapacityOverloadedError
) -> JSONResponse:
    """Expose bounded search admission as transient backpressure to REST callers."""
    return JSONResponse(
        content={'detail': str(error)},
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        headers={'Retry-After': str(error.retry_after_seconds)},
    )


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
    endpoint. Once started, fail readiness if the shared graph client was not
    installed, the authorization keys are no longer usable, or the ingest
    consumer has exited. Queue capacity is a transient backpressure signal
    returned by `/messages`, not pod readiness.
    """
    settings = get_settings()
    worker_task = ingest.async_worker.task
    graphiti_ready = hasattr(request.app.state, GRAPHITI_CLIENT_STATE_ATTR)
    authorizer = getattr(request.app.state, AUTHORIZER_STATE_ATTR, None)
    authorization_ready = authorizer is not None and await authorizer.is_ready()
    worker_running = worker_task is not None and not worker_task.done()
    ingest_ready = ingest.async_worker.ready
    ready = graphiti_ready and authorization_ready and ingest_ready
    return JSONResponse(
        content={
            'status': 'ready' if ready else 'not_ready',
            'graphiti_core_version': version('graphiti-core'),
            'opr_auth_required': settings.opr_auth_required,
            'opr_auth_mode': settings.opr_auth_mode.value,
            'authorization_ready': authorization_ready,
            'ingest_worker_running': worker_running,
            'ingest_accepting': ingest.async_worker.accepting,
            'ingest_draining': ingest.async_worker.draining,
        },
        status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
    )
