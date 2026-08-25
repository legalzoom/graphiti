"""Shared test-collection setup for the graph-service test suite.

`graph_service.routers.ingest.async_worker` is a module-level singleton that
starts unconfigured: it only gets a bound queue size from
`Settings.ingest_queue_maxsize` when the app's lifespan runs (see
`AsyncWorker.configure` in that module). A handful of tests call the real
`add_messages` handler directly, without spinning up the app or monkeypatching
`ingest.async_worker`, so they need the shared singleton configured before
they run. This fixture does that the same way production does: by calling
`configure()`, not by exporting an environment variable for import-time code
to read.
"""

import pytest

from graph_service.routers import ingest


@pytest.fixture(autouse=True)
def _configure_shared_ingest_queue():
    ingest.async_worker.configure(1000)
