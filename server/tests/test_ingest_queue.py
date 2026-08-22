import asyncio
import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from starlette.responses import JSONResponse

import graph_service.main as graph_service_main
from graph_service.config import Settings
from graph_service.dto import AddMessagesRequest, Message, Result
from graph_service.routers import ingest
from graph_service.routers.ingest import AsyncWorker, add_messages
from graph_service.zep_graphiti import ZepGraphiti


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

    result = await add_messages(_request(2), _graphiti(), _settings())

    assert isinstance(result, Result)
    assert result.success is True
    assert worker.queue.qsize() == 2


@pytest.mark.asyncio
async def test_request_past_maxsize_gets_503_with_retry_after_and_does_not_grow_queue(
    monkeypatch,
):
    worker = AsyncWorker(maxsize=2)
    monkeypatch.setattr(ingest, 'async_worker', worker)
    await add_messages(_request(2), _graphiti(), _settings())
    assert worker.queue.qsize() == 2

    response = await add_messages(_request(1), _graphiti(), _settings())

    assert isinstance(response, JSONResponse)
    assert response.status_code == 503
    assert response.headers['Retry-After'] == str(ingest.INGEST_QUEUE_RETRY_AFTER_SECONDS)
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

    response = await add_messages(_request(4), _graphiti(), _settings())

    assert isinstance(response, JSONResponse)
    assert response.status_code == 503
    assert worker.queue.qsize() == 0


@pytest.mark.asyncio
async def test_worker_draining_frees_capacity_for_the_next_request(monkeypatch):
    worker = AsyncWorker(maxsize=1)
    monkeypatch.setattr(ingest, 'async_worker', worker)
    await add_messages(_request(1), _graphiti(), _settings())
    assert worker.queue.qsize() == 1

    rejected = await add_messages(_request(1), _graphiti(), _settings())
    assert isinstance(rejected, JSONResponse)
    assert rejected.status_code == 503

    # Simulate the worker draining one job, exactly as AsyncWorker.worker() does.
    job = worker.queue.get_nowait()
    await job()
    assert worker.queue.qsize() == 0

    accepted = await add_messages(_request(1), _graphiti(), _settings())
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
    await add_messages(_request(3), _graphiti(), _settings())
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
    await add_messages(_request(2), _graphiti(), _settings())

    response = await graph_service_main.healthcheck()

    body = _json_body(response)
    assert body['ingest_queue_depth'] == 2
    assert body['ingest_queue_maxsize'] == 5


def test_required_positive_int_env_rejects_missing_value(monkeypatch):
    monkeypatch.delenv('INGEST_QUEUE_MAXSIZE', raising=False)
    with pytest.raises(RuntimeError, match='INGEST_QUEUE_MAXSIZE'):
        ingest._required_positive_int_env('INGEST_QUEUE_MAXSIZE')


def test_required_positive_int_env_rejects_non_integer_value(monkeypatch):
    monkeypatch.setenv('INGEST_QUEUE_MAXSIZE', 'not-an-int')
    with pytest.raises(RuntimeError, match='INGEST_QUEUE_MAXSIZE'):
        ingest._required_positive_int_env('INGEST_QUEUE_MAXSIZE')


@pytest.mark.parametrize('value', ['0', '-1'])
def test_required_positive_int_env_rejects_zero_or_negative_value(monkeypatch, value):
    monkeypatch.setenv('INGEST_QUEUE_MAXSIZE', value)
    with pytest.raises(RuntimeError, match='INGEST_QUEUE_MAXSIZE'):
        ingest._required_positive_int_env('INGEST_QUEUE_MAXSIZE')
