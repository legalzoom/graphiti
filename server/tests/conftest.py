"""Shared test-collection setup for the graph-service test suite.

`graph_service.routers.ingest` resolves `INGEST_QUEUE_MAXSIZE` at import time:
the module-level `async_worker` singleton needs a bound queue size to
construct itself, and the application code applies no default for it. An
unset value is a deploy-time misconfiguration, not something to paper over
(see `_required_positive_int_env` in that module).

Several test files import handler functions straight from
`graph_service.routers.ingest`, which triggers that import-time resolution
during collection. Give the test environment an explicit value here, the same
way CI sets `SEMAPHORE_LIMIT` explicitly for the live server suite, without
touching every test file individually.
"""

import os

os.environ.setdefault('INGEST_QUEUE_MAXSIZE', '1000')
