import asyncio
import json
import time
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
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


def _settings_values(**overrides) -> dict:
    values = {'openai_api_key': 'test-key', 'ingest_queue_maxsize': 1000}
    values.update(overrides)
    return values


def _settings() -> Settings:
    # group_id below is never the OPR reconciliation group, so
    # `_authorize_opr_write` never touches these attributes.
    return cast(Settings, SimpleNamespace())


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
async def test_enqueue_up_to_maxsize_succeeds(monkeypatch):
    worker = AsyncWorker(maxsize=2)
    monkeypatch.setattr(ingest, 'async_worker', worker)

    result = await add_messages(_request(2), *_client_args(), _settings())

    assert isinstance(result, Result)
    assert result.success is True
    assert worker.queue.qsize() == 2


@pytest.mark.asyncio
async def test_request_past_maxsize_gets_503_and_does_not_grow_queue(monkeypatch):
    worker = AsyncWorker(maxsize=2)
    monkeypatch.setattr(ingest, 'async_worker', worker)
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
async def test_oversized_batch_is_rejected_atomically_with_no_partial_enqueue(monkeypatch):
    """A batch bigger than the remaining capacity enqueues none of it.

    Partially enqueueing a batch and then telling the caller to retry the
    whole batch would requeue the already-accepted messages a second time;
    for messages without a caller-supplied uuid that becomes a duplicate
    episode. The capacity check must be all-or-nothing.
    """
    worker = AsyncWorker(maxsize=3)
    monkeypatch.setattr(ingest, 'async_worker', worker)

    response = await add_messages(_request(4), *_client_args(), _settings())

    assert isinstance(response, JSONResponse)
    assert response.status_code == 503
    assert worker.queue.qsize() == 0


@pytest.mark.asyncio
async def test_worker_draining_frees_capacity_for_the_next_request(monkeypatch):
    worker = AsyncWorker(maxsize=1)
    monkeypatch.setattr(ingest, 'async_worker', worker)
    await add_messages(_request(1), *_client_args(), _settings())
    assert worker.queue.qsize() == 1

    rejected = await add_messages(_request(1), *_client_args(), _settings())
    assert isinstance(rejected, JSONResponse)
    assert rejected.status_code == 503

    # Simulate the worker draining one job, exactly as AsyncWorker.worker() does.
    job = worker.queue.get_nowait()
    await job()
    assert worker.queue.qsize() == 0

    accepted = await add_messages(_request(1), *_client_args(), _settings())
    assert isinstance(accepted, Result)
    assert accepted.success is True
    assert worker.queue.qsize() == 1


@pytest.mark.asyncio
async def test_stop_drops_unprocessed_queue_items_documented_restart_loss(monkeypatch, caplog):
    """Pins the existing restart-loss behavior; this PR does not fix it.

    An in-memory queue cannot survive a pod restart. The OPR outbox producer
    is durable and retries, so this loss is recovered by the caller. The drop
    itself must still be logged, not silent.
    """
    worker = AsyncWorker(maxsize=5)
    monkeypatch.setattr(ingest, 'async_worker', worker)
    await add_messages(_request(3), *_client_args(), _settings())
    assert worker.queue.qsize() == 3

    with caplog.at_level('WARNING'):
        await worker.stop()

    assert worker.queue.qsize() == 0
    assert any('Dropping 3 unprocessed job' in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_stop_counts_a_job_interrupted_mid_flight_as_dropped(monkeypatch, caplog):
    """A job the worker had already dequeued is invisible to `queue.qsize()`.

    If `stop()` only counted what is still sitting in the queue, cancelling
    the worker while it is inside `await job()` would undercount the drop by
    exactly the in-flight job, understating what a restart actually loses.
    """
    started_processing = asyncio.Event()
    finish_processing = asyncio.Event()

    async def slow_job():
        started_processing.set()
        await finish_processing.wait()

    worker = AsyncWorker(maxsize=5)
    monkeypatch.setattr(ingest, 'async_worker', worker)
    worker.queue.put_nowait(slow_job)
    await worker.start()
    await started_processing.wait()
    assert worker.queue.qsize() == 0  # dequeued into the worker, not sitting in the queue

    with caplog.at_level('WARNING'):
        await worker.stop()

    assert any('Dropping 1 unprocessed job' in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_healthcheck_exposes_ingest_queue_depth_and_maxsize(monkeypatch):
    worker = AsyncWorker(maxsize=5)
    monkeypatch.setattr(ingest, 'async_worker', worker)
    await add_messages(_request(2), *_client_args(), _settings())

    response = await graph_service_main.healthcheck()

    body = _json_body(response)
    assert body['ingest_queue_depth'] == 2
    assert body['ingest_queue_maxsize'] == 5


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
async def test_queued_job_resolves_the_shared_client_at_execution_time(monkeypatch):
    """The job must read app state when it runs, not capture a client when queued.

    Closing over the injected client pinned an entire per-request client stack
    inside the queue (three AsyncOpenAI clients and a graph driver per item),
    and because `/messages` returns 202 before the job runs, that client's
    driver had already been closed by the time the job used it.
    """
    worker = AsyncWorker(maxsize=2)
    monkeypatch.setattr(ingest, 'async_worker', worker)
    request_time_client = _graphiti()
    http_request = _http_request(request_time_client)

    await add_messages(_request(1), http_request, request_time_client, _settings())

    # Swap the client on app state after the response was produced. A job that
    # captured the request-time client would still call that one.
    execution_time_client = _graphiti()
    setattr(http_request.app.state, GRAPHITI_CLIENT_STATE_ATTR, execution_time_client)

    job = worker.queue.get_nowait()
    await job()

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
        assert len(built) == 1, 'a client per HTTP request is the leak this change removes'
        assert cast(AsyncMock, built[0].add_episode).await_count == request_count
        cast(AsyncMock, built[0].close).assert_not_awaited()

    assert len(built) == 1
    cast(AsyncMock, built[0].close).assert_awaited_once()
