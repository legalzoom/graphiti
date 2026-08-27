from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from graph_service.config import get_settings
from graph_service.routers import ingest, retrieve
from graph_service.zep_graphiti import (
    GRAPHITI_CLIENT_STATE_ATTR,
    build_graphiti_client,
    set_graphiti_client,
)


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
        # merged inside this one by `include_router`, so the ingest worker has
        # already been stopped by the time we get here and no queued job can
        # still be holding the client.
        await client.close()


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


@app.get('/readyz')
async def readiness(request: Request):
    """Report whether this process can safely receive application traffic.

    Settings are validated before the application lifespan starts, so a
    deployment with required-but-missing OPR credentials never reaches this
    endpoint. Once started, fail readiness if either the shared graph client
    was not installed or the ingest consumer has exited.
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
