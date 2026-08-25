from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from graph_service.config import get_settings
from graph_service.routers import ingest, retrieve
from graph_service.zep_graphiti import build_graphiti_client, set_graphiti_client


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
