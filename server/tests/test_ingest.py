"""Unit tests for the /messages ingest route: idempotency guard and backpressure.

These call the route function and queued job directly (no HTTP layer, no
background worker task) so behavior is exercised deterministically and in the
foreground, matching graph_service's existing convention of stubbing the
graphiti dependency rather than hitting a live database.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from graphiti_core.errors import NodeNotFoundError

from graph_service.dto import AddMessagesRequest, Message
from graph_service.routers import ingest as ingest_router


def _message(uuid: str | None = None, content: str = 'hello') -> Message:
    return Message(content=content, uuid=uuid, role_type='user', role='user')


class _GraphitiStub:
    def __init__(self):
        self.driver = object()
        self.add_episode_calls: list[str | None] = []

    async def add_episode(self, **kwargs):
        self.add_episode_calls.append(kwargs.get('uuid'))


@pytest.fixture
def graphiti_stub():
    return _GraphitiStub()


@pytest.fixture(autouse=True)
def bounded_queue(monkeypatch):
    """Give each test its own bounded queue instead of the shared module singleton."""
    queue = asyncio.Queue(maxsize=5)
    monkeypatch.setattr(ingest_router.async_worker, 'queue', queue)
    return queue


@pytest.mark.asyncio
async def test_fresh_uuid_calls_add_episode(graphiti_stub, monkeypatch):
    monkeypatch.setattr(
        ingest_router.EpisodicNode,
        'get_by_uuid',
        AsyncMock(side_effect=NodeNotFoundError('new-uuid')),
    )
    request = AddMessagesRequest(group_id='g1', messages=[_message(uuid='new-uuid')])

    await ingest_router.add_messages(request, graphiti_stub)
    job = ingest_router.async_worker.queue.get_nowait()
    await job()

    assert graphiti_stub.add_episode_calls == ['new-uuid']


@pytest.mark.asyncio
async def test_duplicate_uuid_skips_add_episode(graphiti_stub, monkeypatch):
    monkeypatch.setattr(
        ingest_router.EpisodicNode,
        'get_by_uuid',
        AsyncMock(return_value=object()),
    )
    request = AddMessagesRequest(group_id='g1', messages=[_message(uuid='dup-uuid')])

    await ingest_router.add_messages(request, graphiti_stub)
    job = ingest_router.async_worker.queue.get_nowait()
    await job()

    assert graphiti_stub.add_episode_calls == []


@pytest.mark.asyncio
async def test_lookup_error_proceeds_with_add_episode(graphiti_stub, monkeypatch):
    """Availability wins over the guard: an unexpected existence-check
    failure must extract, not drop the episode."""
    monkeypatch.setattr(
        ingest_router.EpisodicNode,
        'get_by_uuid',
        AsyncMock(side_effect=RuntimeError('driver timeout')),
    )
    request = AddMessagesRequest(group_id='g1', messages=[_message(uuid='check-err-uuid')])

    await ingest_router.add_messages(request, graphiti_stub)
    job = ingest_router.async_worker.queue.get_nowait()
    await job()

    assert graphiti_stub.add_episode_calls == ['check-err-uuid']


@pytest.mark.asyncio
async def test_stop_logs_and_drains_discarded_jobs(caplog):
    # The worker task is deliberately not started: a running worker would
    # race the test by consuming the queued jobs before stop() counts them.
    worker = ingest_router.AsyncWorker(maxsize=5)
    worker.queue.put_nowait(lambda: None)
    worker.queue.put_nowait(lambda: None)

    with caplog.at_level('WARNING', logger='uvicorn.error'):
        await worker.stop()

    assert worker.queue.empty()
    assert any('discarding 2 queued job(s)' in r.message for r in caplog.records)


def test_queue_maxsize_rejects_non_integer(monkeypatch):
    monkeypatch.setenv('GRAPHITI_INGEST_QUEUE_MAX', 'not-a-number')
    with pytest.raises(RuntimeError, match='must be an integer'):
        ingest_router._ingest_queue_maxsize()


def test_queue_maxsize_rejects_non_positive(monkeypatch):
    """asyncio.Queue treats maxsize <= 0 as unbounded, which would silently
    disable backpressure; the config must refuse it."""
    monkeypatch.setenv('GRAPHITI_INGEST_QUEUE_MAX', '0')
    with pytest.raises(RuntimeError, match='must be >= 1'):
        ingest_router._ingest_queue_maxsize()


def test_queue_maxsize_defaults_when_unset(monkeypatch):
    monkeypatch.delenv('GRAPHITI_INGEST_QUEUE_MAX', raising=False)
    assert ingest_router._ingest_queue_maxsize() == ingest_router.DEFAULT_INGEST_QUEUE_MAX


@pytest.mark.asyncio
async def test_queue_full_returns_503_and_reports_accepted_count(graphiti_stub, bounded_queue):
    # Pre-fill the queue, leaving room for exactly one more message.
    for _ in range(bounded_queue.maxsize - 1):
        bounded_queue.put_nowait(lambda: None)

    request = AddMessagesRequest(
        group_id='g1',
        messages=[_message(content='a'), _message(content='b'), _message(content='c')],
    )

    with pytest.raises(HTTPException) as exc_info:
        await ingest_router.add_messages(request, graphiti_stub)

    assert exc_info.value.status_code == 503
    assert 'queued 1' in exc_info.value.detail
    assert 'rejected 2' in exc_info.value.detail
    # The one message that fit stays queued; it is not lost.
    assert bounded_queue.qsize() == bounded_queue.maxsize
