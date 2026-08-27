import asyncio
import json
import time
from contextlib import suppress
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.responses import JSONResponse

import graph_service.main as graph_service_main
from graph_service.config import Settings, get_settings
from graph_service.dto import AddMessagesRequest, Message, Result
from graph_service.routers import ingest
from graph_service.routers.ingest import AsyncWorker, add_messages
from graph_service.zep_graphiti import (
    GRAPHITI_CLIENT_STATE_ATTR,
    ZepGraphiti,
    get_graphiti,
    graphiti_client_from_app,
)


@pytest_asyncio.fixture
async def non_consuming_worker(monkeypatch):
    """Start workers with live admission while leaving queued jobs untouched."""
    tasks: list[asyncio.Task[None]] = []

    async def start(worker: AsyncWorker) -> AsyncWorker:
        wait_forever = asyncio.Event()

        async def idle_worker() -> None:
            await wait_forever.wait()

        monkeypatch.setattr(worker, 'worker', idle_worker)
        await worker.start()
        assert worker.task is not None
        tasks.append(worker.task)
        return worker

    yield start

    for task in tasks:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


def _settings_values(**overrides) -> dict:
    values = {'openai_api_key': 'test-key', 'ingest_queue_maxsize': 1000}
    values.update(overrides)
    return values


def _required_opr_settings_values(**overrides) -> dict:
    values = _settings_values(
        opr_auth_required=True,
        opr_read_token='read-' + ('a' * 32),
        opr_write_token='write-' + ('b' * 32),
        opr_reconciliation_token='reconcile-' + ('c' * 32),
        opr_retirement_token='retire-' + ('d' * 32),
        opr_writer_fleet_epoch='epoch-' + ('e' * 32),
        graphiti_admin_token='admin-' + ('f' * 32),
    )
    values.update(overrides)
    return values


def _settings() -> Settings:
    # group_id below is never the OPR reconciliation group, so
    # `_authorize_opr_write` never touches these attributes.
    return cast(Settings, SimpleNamespace())


async def _noop_job() -> None:
    return None


def _json_body(response: JSONResponse) -> dict:
    return json.loads(bytes(response.body))


def _graphiti() -> ZepGraphiti:
    return cast(
        ZepGraphiti,
        SimpleNamespace(
            assert_episode_uuid_group=AsyncMock(),
            add_episode=AsyncMock(),
        ),
    )


def _client_args() -> tuple[Request, ZepGraphiti]:
    """The (http_request, graphiti) pair `add_messages` takes, sharing one client."""
    graphiti = _graphiti()
    return _http_request(graphiti), graphiti


def _http_request(graphiti: ZepGraphiti | None = None) -> Request:
    """A stand-in for the Starlette request `add_messages` takes.

    Only `.app.state` matters: that is where the one shared Graphiti client
    lives and where a queued job resolves it from at execution time.
    """
    state = SimpleNamespace()
    if graphiti is not None:
        setattr(state, GRAPHITI_CLIENT_STATE_ATTR, graphiti)
    return cast(Request, SimpleNamespace(app=SimpleNamespace(state=state)))


def _request(count: int, group_id: str = 'not-opr') -> AddMessagesRequest:
    return AddMessagesRequest(
        group_id=group_id,
        messages=[
            Message(content=f'message-{i}', role_type='user', role=None) for i in range(count)
        ],
    )


@pytest.mark.asyncio
async def test_enqueue_up_to_maxsize_succeeds(monkeypatch, non_consuming_worker):
    worker = AsyncWorker(maxsize=2)
    monkeypatch.setattr(ingest, 'async_worker', worker)
    await non_consuming_worker(worker)

    result = await add_messages(_request(2), *_client_args(), _settings())

    assert isinstance(result, Result)
    assert result.success is True
    assert worker.queue.qsize() == 2


@pytest.mark.asyncio
async def test_configured_worker_does_not_accept_before_consumer_starts(monkeypatch):
    worker = AsyncWorker(maxsize=2)
    monkeypatch.setattr(ingest, 'async_worker', worker)

    response = await add_messages(_request(1), *_client_args(), _settings())

    assert isinstance(response, JSONResponse)
    assert response.status_code == 503
    assert _json_body(response) == {
        'success': False,
        'error': 'ingest_worker_unavailable',
        'message': 'The ingestion worker is unavailable; retry another instance.',
        'queue_depth': 0,
        'queue_maxsize': 2,
    }
    assert worker.queue.qsize() == 0


@pytest.mark.asyncio
async def test_request_past_maxsize_gets_503_and_does_not_grow_queue(
    monkeypatch, non_consuming_worker
):
    worker = AsyncWorker(maxsize=2)
    monkeypatch.setattr(ingest, 'async_worker', worker)
    await non_consuming_worker(worker)
    await add_messages(_request(2), *_client_args(), _settings())
    assert worker.queue.qsize() == 2

    response = await add_messages(_request(1), *_client_args(), _settings())

    assert isinstance(response, JSONResponse)
    assert response.status_code == 503
    assert 'Retry-After' not in response.headers
    body = _json_body(response)
    assert body == {
        'success': False,
        'error': 'ingest_queue_full',
        'message': 'Ingestion queue is at capacity; retry later.',
        'queue_depth': 2,
        'queue_maxsize': 2,
    }
    # The queue never grew past its bound: the rejected message was not enqueued.
    assert worker.queue.qsize() == 2


@pytest.mark.asyncio
async def test_oversized_batch_is_rejected_atomically_with_no_partial_enqueue(
    monkeypatch, non_consuming_worker
):
    """A batch bigger than the remaining capacity enqueues none of it.

    Partially enqueueing a batch and then telling the caller to retry the
    whole batch would requeue the already-accepted messages a second time;
    for messages without a caller-supplied uuid that becomes a duplicate
    episode. The capacity check must be all-or-nothing.
    """
    worker = AsyncWorker(maxsize=3)
    monkeypatch.setattr(ingest, 'async_worker', worker)
    await non_consuming_worker(worker)

    response = await add_messages(_request(4), *_client_args(), _settings())

    assert isinstance(response, JSONResponse)
    assert response.status_code == 503
    assert worker.queue.qsize() == 0


@pytest.mark.asyncio
async def test_worker_draining_frees_capacity_for_the_next_request(
    monkeypatch, non_consuming_worker
):
    worker = AsyncWorker(maxsize=1)
    monkeypatch.setattr(ingest, 'async_worker', worker)
    await non_consuming_worker(worker)
    await add_messages(_request(1), *_client_args(), _settings())
    assert worker.queue.qsize() == 1

    rejected = await add_messages(_request(1), *_client_args(), _settings())
    assert isinstance(rejected, JSONResponse)
    assert rejected.status_code == 503

    # Simulate the worker draining one job, exactly as AsyncWorker.worker() does.
    job = worker.queue.get_nowait()
    await job()
    worker.queue.task_done()
    assert worker.queue.qsize() == 0

    accepted = await add_messages(_request(1), *_client_args(), _settings())
    assert isinstance(accepted, Result)
    assert accepted.success is True
    assert worker.queue.qsize() == 1


@pytest.mark.asyncio
async def test_stop_without_a_live_worker_counts_and_critically_logs_dropped_jobs(
    monkeypatch, caplog
):
    worker = AsyncWorker(maxsize=5)
    monkeypatch.setattr(ingest, 'async_worker', worker)
    for _ in range(3):
        worker.queue.put_nowait(_noop_job)
    assert worker.queue.qsize() == 3

    with caplog.at_level('CRITICAL'):
        dropped = await worker.stop()

    assert dropped == 3
    assert worker.last_shutdown_dropped_jobs == 3
    assert worker.queue.qsize() == 0
    await asyncio.wait_for(worker.queue.join(), timeout=1)
    assert any(
        'worker is not running' in record.message and 'dropping 3 unprocessed job' in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_stop_timeout_counts_inflight_and_queued_jobs_and_logs_critical(monkeypatch, caplog):
    started_processing = asyncio.Event()
    finish_processing = asyncio.Event()

    async def slow_job():
        started_processing.set()
        await finish_processing.wait()

    worker = AsyncWorker(maxsize=5, drain_timeout_seconds=0.01)
    monkeypatch.setattr(ingest, 'async_worker', worker)
    worker.queue.put_nowait(slow_job)
    worker.queue.put_nowait(slow_job)
    worker.queue.put_nowait(slow_job)
    await worker.start()
    await started_processing.wait()
    assert worker.queue.qsize() == 2  # one job is already in flight

    with caplog.at_level('CRITICAL'):
        dropped = await worker.stop()

    assert dropped == 3
    assert worker.last_shutdown_dropped_jobs == 3
    assert worker.queue.qsize() == 0
    await asyncio.wait_for(worker.queue.join(), timeout=1)
    assert any(
        'drain timed out' in record.message and '3 unprocessed job' in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_stop_remains_bounded_when_a_job_suppresses_repeated_cancellation(
    monkeypatch, caplog
):
    started_processing = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release_job = asyncio.Event()

    async def cancellation_suppressing_job():
        started_processing.set()
        while not release_job.is_set():
            try:
                await release_job.wait()
            except asyncio.CancelledError:
                cancellation_seen.set()

    worker = AsyncWorker(maxsize=1, drain_timeout_seconds=0.01)
    monkeypatch.setattr(ingest, 'async_worker', worker)
    worker.queue.put_nowait(cancellation_suppressing_job)
    await worker.start()
    await started_processing.wait()

    with caplog.at_level('CRITICAL'):
        dropped = await asyncio.wait_for(worker.stop(), timeout=0.25)

    assert dropped == 1
    assert cancellation_seen.is_set()
    assert worker.last_shutdown_dropped_jobs == 1
    assert any(
        'termination deadline remains bounded' in record.message for record in caplog.records
    )

    # Release the deliberately non-cooperative test job, then cancel the
    # consumer once it returns to its normal queue wait so no task leaks out of
    # the test event loop.
    release_job.set()
    await asyncio.sleep(0)
    assert worker.task is not None
    worker.task.cancel()
    with suppress(asyncio.CancelledError):
        await worker.task


@pytest.mark.asyncio
async def test_stop_closes_admission_and_successfully_drains_every_accepted_job(monkeypatch):
    started_processing = asyncio.Event()
    finish_processing = asyncio.Event()
    completed: list[int] = []

    async def first_job():
        started_processing.set()
        await finish_processing.wait()
        completed.append(1)

    async def second_job():
        completed.append(2)

    worker = AsyncWorker(maxsize=5, drain_timeout_seconds=1)
    monkeypatch.setattr(ingest, 'async_worker', worker)
    worker.queue.put_nowait(first_job)
    worker.queue.put_nowait(second_job)
    await worker.start()
    await started_processing.wait()

    stop_task = asyncio.create_task(worker.stop())
    await asyncio.sleep(0)
    assert worker.accepting is False
    assert worker.draining is True
    assert stop_task.done() is False

    finish_processing.set()
    assert await stop_task == 0
    assert completed == [1, 2]
    assert worker.last_shutdown_dropped_jobs == 0
    assert worker.task is not None and worker.task.done()
    await asyncio.wait_for(worker.queue.join(), timeout=1)


@pytest.mark.asyncio
async def test_healthcheck_exposes_ingest_queue_depth_and_maxsize(
    monkeypatch, non_consuming_worker
):
    worker = AsyncWorker(maxsize=5)
    monkeypatch.setattr(ingest, 'async_worker', worker)
    await non_consuming_worker(worker)
    await add_messages(_request(2), *_client_args(), _settings())

    response = await graph_service_main.healthcheck()

    body = _json_body(response)
    assert body['ingest_queue_depth'] == 2
    assert body['ingest_queue_maxsize'] == 5


@pytest.mark.asyncio
async def test_readiness_requires_shared_client_and_live_ingest_worker(monkeypatch):
    settings = Settings.model_validate(_required_opr_settings_values())
    monkeypatch.setattr(graph_service_main, 'get_settings', lambda: settings)
    worker = AsyncWorker(maxsize=5)
    monkeypatch.setattr(ingest, 'async_worker', worker)

    unavailable = await graph_service_main.readiness(_http_request())
    assert unavailable.status_code == 503
    assert _json_body(unavailable) == {
        'status': 'not_ready',
        'graphiti_core_version': _json_body(unavailable)['graphiti_core_version'],
        'opr_auth_required': True,
        'ingest_worker_running': False,
        'ingest_accepting': False,
        'ingest_draining': False,
    }

    await worker.start()
    await asyncio.sleep(0)
    try:
        ready = await graph_service_main.readiness(_http_request(_graphiti()))
        assert ready.status_code == 200
        assert _json_body(ready)['status'] == 'ready'
        assert _json_body(ready)['opr_auth_required'] is True
        assert _json_body(ready)['ingest_worker_running'] is True
        assert _json_body(ready)['ingest_accepting'] is True
        assert _json_body(ready)['ingest_draining'] is False
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_drain_makes_readiness_false_and_rejects_new_ingest(monkeypatch):
    settings = Settings.model_validate(_settings_values())
    monkeypatch.setattr(graph_service_main, 'get_settings', lambda: settings)
    worker = AsyncWorker(maxsize=5)
    monkeypatch.setattr(ingest, 'async_worker', worker)
    await worker.start()
    worker.begin_drain()

    try:
        readiness = await graph_service_main.readiness(_http_request(_graphiti()))
        assert readiness.status_code == 503
        assert _json_body(readiness)['status'] == 'not_ready'
        assert _json_body(readiness)['ingest_worker_running'] is True
        assert _json_body(readiness)['ingest_accepting'] is False
        assert _json_body(readiness)['ingest_draining'] is True

        response = await add_messages(_request(1), *_client_args(), _settings())
        assert isinstance(response, JSONResponse)
        assert response.status_code == 503
        assert _json_body(response) == {
            'success': False,
            'error': 'ingest_draining',
            'message': 'Ingestion is draining for shutdown; retry another instance.',
            'queue_depth': 0,
            'queue_maxsize': 5,
        }
        assert worker.queue.qsize() == 0
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_request_suspended_in_uuid_preflight_cannot_enqueue_after_drain_begins(
    monkeypatch, non_consuming_worker
):
    preflight_started = asyncio.Event()
    finish_preflight = asyncio.Event()

    async def assert_episode_uuid_group(_uuid: str, _group_id: str):
        preflight_started.set()
        await finish_preflight.wait()

    graphiti = cast(
        ZepGraphiti,
        SimpleNamespace(
            assert_episode_uuid_group=AsyncMock(side_effect=assert_episode_uuid_group),
            add_episode=AsyncMock(),
        ),
    )
    worker = AsyncWorker(maxsize=5)
    monkeypatch.setattr(ingest, 'async_worker', worker)
    await non_consuming_worker(worker)
    request = AddMessagesRequest(
        group_id='not-opr',
        messages=[
            Message(uuid='episode-id', content='message', role_type='user', role=None),
        ],
    )

    request_task = asyncio.create_task(
        add_messages(request, _http_request(graphiti), graphiti, _settings())
    )
    await preflight_started.wait()
    worker.begin_drain()
    finish_preflight.set()

    response = await request_task
    assert isinstance(response, JSONResponse)
    assert response.status_code == 503
    assert _json_body(response)['error'] == 'ingest_draining'
    assert worker.queue.qsize() == 0


def test_settings_requires_ingest_queue_maxsize():
    values = _settings_values()
    del values['ingest_queue_maxsize']
    with pytest.raises(ValidationError, match='ingest_queue_maxsize'):
        Settings.model_validate(values)


def test_settings_rejects_non_integer_ingest_queue_maxsize():
    with pytest.raises(ValidationError, match='ingest_queue_maxsize'):
        Settings.model_validate(_settings_values(ingest_queue_maxsize='not-an-int'))


@pytest.mark.parametrize('value', [0, -1])
def test_settings_rejects_non_positive_ingest_queue_maxsize(value):
    with pytest.raises(ValidationError, match='ingest_queue_maxsize'):
        Settings.model_validate(_settings_values(ingest_queue_maxsize=value))


@pytest.mark.parametrize('value', [0, -1, 51])
def test_settings_rejects_drain_timeout_outside_pod_shutdown_budget(value):
    with pytest.raises(ValidationError, match='ingest_drain_timeout_seconds'):
        Settings.model_validate(_settings_values(ingest_drain_timeout_seconds=value))


def test_settings_defaults_drain_timeout_to_25_seconds():
    settings = Settings.model_validate(_settings_values())

    assert settings.ingest_drain_timeout_seconds == 25


def test_async_worker_unconfigured_use_raises_instead_of_falling_back():
    """No configure() call means no assumed queue size, not an unbounded one."""
    worker = AsyncWorker()

    with pytest.raises(RuntimeError, match='configure'):
        _ = worker.depth


@pytest.mark.asyncio
async def test_lifespan_configures_async_worker_from_settings(monkeypatch):
    """The worker's bound size comes from Settings at startup, not import time."""
    settings = Settings.model_validate(_settings_values(ingest_queue_maxsize=7))
    monkeypatch.setattr(ingest, 'get_settings', lambda: settings)
    worker = AsyncWorker()
    monkeypatch.setattr(ingest, 'async_worker', worker)

    async with ingest.lifespan(cast(FastAPI, SimpleNamespace())):
        assert worker.capacity == 7
        assert worker.task is not None
        # Let the worker task actually take its first turn before the context
        # exits and cancels it; cancelling a task before it has ever run once
        # raises CancelledError straight through `await self.task` instead of
        # letting `worker()`'s own try/except handle it.
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_worker_logs_a_failing_job_and_keeps_draining_the_queue(monkeypatch, caplog):
    """A job that raises must not kill the consumer.

    This is the failure that turned an embedder timeout into an OOM loop:
    `worker()` used to catch only `asyncio.CancelledError`, so any other
    exception propagated out of `while True`, ended the worker task, and was
    never logged (the task is only awaited in `stop()`). `/messages` kept
    returning 202 into a queue with no consumer until the pod was killed.
    """
    worker = AsyncWorker(maxsize=5)
    monkeypatch.setattr(ingest, 'async_worker', worker)
    second_job_ran = asyncio.Event()

    async def failing_job():
        raise RuntimeError('embedder request timed out')

    async def following_job():
        second_job_ran.set()

    worker.queue.put_nowait(failing_job)
    worker.queue.put_nowait(following_job)

    with caplog.at_level('ERROR'):
        await worker.start()
        # The worker survived the failure only if it reaches the next job.
        await asyncio.wait_for(second_job_ran.wait(), timeout=5)
        assert worker.task is not None
        assert not worker.task.done()
        await worker.stop()

    failures = [record for record in caplog.records if record.levelname == 'ERROR']
    assert failures, 'a failing ingest job must be logged, not silently swallowed'
    assert 'Ingest worker job raised' in failures[0].message
    # logger.exception, so the traceback (and the original error) is attached.
    assert failures[0].exc_info is not None
    assert 'embedder request timed out' in caplog.text


@pytest.mark.asyncio
async def test_worker_task_death_is_logged_and_its_exception_retrieved(monkeypatch, caplog):
    """If the consumer ever does exit, it must not exit unobserved."""
    worker = AsyncWorker(maxsize=1)

    async def exploding_worker():
        raise RuntimeError('worker loop died')

    monkeypatch.setattr(worker, 'worker', exploding_worker)

    with caplog.at_level('CRITICAL'):
        await worker.start()
        assert worker.task is not None
        with pytest.raises(RuntimeError, match='worker loop died'):
            await worker.task
        # Let the done callback registered by start() run.
        await asyncio.sleep(0)

    critical = [record for record in caplog.records if record.levelname == 'CRITICAL']
    assert critical, 'a dead ingest worker must be logged'
    assert 'no consumer' in critical[0].message
    assert 'worker loop died' in caplog.text


@pytest.mark.asyncio
async def test_queued_job_resolves_the_shared_client_at_execution_time(
    monkeypatch, non_consuming_worker
):
    """The job must read app state when it runs, not capture a client when queued.

    Closing over the injected client pinned an entire per-request client stack
    inside the queue (three AsyncOpenAI clients and a graph driver per item),
    and because `/messages` returns 202 before the job runs, that client's
    driver had already been closed by the time the job used it.
    """
    worker = AsyncWorker(maxsize=2)
    monkeypatch.setattr(ingest, 'async_worker', worker)
    await non_consuming_worker(worker)
    request_time_client = _graphiti()
    http_request = _http_request(request_time_client)

    await add_messages(_request(1), http_request, request_time_client, _settings())

    # Swap the client on app state after the response was produced. A job that
    # captured the request-time client would still call that one.
    execution_time_client = _graphiti()
    setattr(http_request.app.state, GRAPHITI_CLIENT_STATE_ATTR, execution_time_client)

    job = worker.queue.get_nowait()
    await job()
    worker.queue.task_done()

    cast(AsyncMock, execution_time_client.add_episode).assert_awaited_once()
    cast(AsyncMock, request_time_client.add_episode).assert_not_awaited()


@pytest.mark.asyncio
async def test_get_graphiti_returns_the_shared_client_and_never_closes_it():
    """The dependency hands back the lifespan-owned client, per request, unclosed."""
    close = AsyncMock()
    client = cast(ZepGraphiti, SimpleNamespace(close=close))
    http_request = _http_request(client)

    assert await get_graphiti(http_request) is client
    assert await get_graphiti(http_request) is client

    close.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_graphiti_raises_when_lifespan_never_installed_a_client():
    """No shared client means fail loudly, not build a private one on demand."""
    with pytest.raises(RuntimeError, match='lifespan'):
        await get_graphiti(_http_request())


@pytest.mark.asyncio
async def test_app_lifespan_builds_exactly_one_graphiti_client_and_closes_it(monkeypatch):
    """One client per process, not one per HTTP request."""
    settings = Settings.model_validate(_settings_values())
    monkeypatch.setattr(graph_service_main, 'get_settings', lambda: settings)

    built: list[ZepGraphiti] = []

    def fake_build(_settings) -> ZepGraphiti:
        client = cast(
            ZepGraphiti,
            SimpleNamespace(
                build_indices_and_constraints=AsyncMock(),
                close=AsyncMock(),
            ),
        )
        built.append(client)
        return client

    monkeypatch.setattr(graph_service_main, 'build_graphiti_client', fake_build)

    app = FastAPI()
    async with graph_service_main.lifespan(app):
        assert len(built) == 1
        # Every request and every queued job resolves this same instance.
        assert graphiti_client_from_app(app) is built[0]
        assert await get_graphiti(cast(Request, SimpleNamespace(app=app))) is built[0]
        cast(AsyncMock, built[0].build_indices_and_constraints).assert_awaited_once()
        cast(AsyncMock, built[0].close).assert_not_awaited()

    assert len(built) == 1
    cast(AsyncMock, built[0].close).assert_awaited_once()


def test_app_builds_one_client_for_many_requests_and_drains_jobs_against_it(monkeypatch):
    """End-to-end through the real ASGI stack, not by calling handlers directly.

    Proves the whole point of the shared-client change: N `/messages` requests
    construct exactly one Graphiti client, the queued jobs resolve that same
    client after their 202 responses were already sent, and the client is
    closed once at shutdown rather than once per request.
    """
    settings = Settings.model_validate(_settings_values(ingest_queue_maxsize=8))
    monkeypatch.setattr(graph_service_main, 'get_settings', lambda: settings)
    monkeypatch.setattr(ingest, 'get_settings', lambda: settings)

    built: list[ZepGraphiti] = []

    def fake_build(_settings) -> ZepGraphiti:
        client = cast(
            ZepGraphiti,
            SimpleNamespace(
                build_indices_and_constraints=AsyncMock(),
                close=AsyncMock(),
                assert_episode_uuid_group=AsyncMock(),
                add_episode=AsyncMock(),
            ),
        )
        built.append(client)
        return client

    monkeypatch.setattr(graph_service_main, 'build_graphiti_client', fake_build)
    # The route's own `ZepEnvDep` resolves `get_settings` through FastAPI, not
    # through this module's globals, so it needs a dependency override too.
    # setitem, so the override is removed again when this test finishes.
    monkeypatch.setitem(graph_service_main.app.dependency_overrides, get_settings, lambda: settings)

    request_count = 3
    with TestClient(graph_service_main.app) as client:
        for index in range(request_count):
            response = client.post(
                '/messages',
                json={
                    'group_id': 'not-opr',
                    'messages': [
                        {'content': f'message-{index}', 'role_type': 'user', 'role': None}
                    ],
                },
            )
            assert response.status_code == 202

        # Let the worker drain; bounded so a regression fails rather than hangs.
        for _ in range(200):
            if client.get('/healthcheck').json()['ingest_queue_depth'] == 0:
                break
            time.sleep(0.01)

        assert client.get('/healthcheck').json()['ingest_queue_depth'] == 0
        readiness = client.get('/readyz')
        assert readiness.status_code == 200
        assert readiness.json()['status'] == 'ready'
        assert len(built) == 1, 'a client per HTTP request is the leak this change removes'
        assert cast(AsyncMock, built[0].add_episode).await_count == request_count
        cast(AsyncMock, built[0].close).assert_not_awaited()

    assert len(built) == 1
    cast(AsyncMock, built[0].close).assert_awaited_once()
