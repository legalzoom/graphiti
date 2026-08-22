from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from graph_service.config import get_settings
from graph_service.routers import ingest, retrieve
from graph_service.zep_graphiti import initialize_graphiti


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    await initialize_graphiti(settings)
    yield
    # Shutdown
    # No need to close Graphiti here, as it's handled per-request


app = FastAPI(lifespan=lifespan)


app.include_router(retrieve.router)
app.include_router(ingest.router)


@app.get('/healthcheck')
async def healthcheck():
    return JSONResponse(
        content={
            'status': 'healthy',
            'graphiti_core_version': version('graphiti-core'),
            'ingest_queue_depth': ingest.async_worker.queue.qsize(),
            'ingest_queue_maxsize': ingest.async_worker.queue.maxsize,
        },
        status_code=200,
    )
