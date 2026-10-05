"""Synthetic transport cases for the recorded graph-clear/keyword-orphan failure.

Internal incident records are deliberately not published in this public repository.
Only external I/O is mocked; maintenance, session, driver and SDK bulk code are real.
"""

import asyncio
import json
import threading
from typing import cast
from unittest.mock import MagicMock

import pytest
from opensearchpy import JSONSerializer

from graphiti_core.driver.neptune.keyword_cleanup import KeywordIndexCleanupError
from graphiti_core.driver.neptune.operations.entity_node_ops import NeptuneEntityNodeOperations
from graphiti_core.driver.neptune.operations.graph_ops import NeptuneGraphMaintenanceOperations
from graphiti_core.driver.neptune_driver import NeptuneDriver, text_aoss_indices
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

STALE_UUID = '00000000-0000-4000-8000-000000000001'
LIVE_UUID = '00000000-0000-4000-8000-000000000002'
FOREIGN_UUID = '00000000-0000-4000-8000-000000000003'


def hit(uuid=STALE_UUID, group='team-a', *, sequence=7):
    return {
        '_index': 'edge_name_and_fact',
        '_id': uuid,
        '_source': {'uuid': uuid, 'group_id': group},
        '_seq_no': sequence,
        '_primary_term': 1,
        'sort': [uuid],
    }


def response(hits):
    return {'timed_out': False, '_shards': {'failed': 0}, 'hits': {'hits': hits}}


def driver_for(hits, *, live=()):
    driver = object.__new__(NeptuneDriver)
    driver.client = MagicMock()
    driver.client.query.side_effect = lambda query, params: (
        [{'uuid': uuid} for uuid in params['uuids'] if uuid in live] if 'uuids' in params else []
    )
    driver.aoss_client = MagicMock()
    cast(MagicMock, driver.aoss_client).transport.serializer = JSONSerializer()
    cast(MagicMock, driver.aoss_client).search.side_effect = lambda *, index, body: response(
        hits if index == 'edge_name_and_fact' and 'search_after' not in body else []
    )
    cast(MagicMock, driver.aoss_client).bulk.return_value = {
        'errors': False,
        'items': [{'delete': {'status': 200, '_id': STALE_UUID}}],
    }
    driver._entity_node_ops = NeptuneEntityNodeOperations(driver)
    driver._graph_ops = NeptuneGraphMaintenanceOperations(driver)
    return driver


@pytest.mark.asyncio
async def test_group_clear_removes_keyword_documents_even_when_graph_is_already_empty():
    driver = driver_for([hit()])
    await clear_data(driver, ['team-a'])
    cast(MagicMock, driver.aoss_client).bulk.assert_called_once()
    body = cast(MagicMock, driver.aoss_client).bulk.call_args.kwargs['body']
    assert STALE_UUID in body
    assert 'if_seq_no' in body
    assert 'if_primary_term' in body
    assert json.loads(body)['delete'] == {
        '_index': 'edge_name_and_fact',
        '_id': STALE_UUID,
        'if_seq_no': 7,
        'if_primary_term': 1,
    }


@pytest.mark.asyncio
async def test_repair_preserves_live_records_and_exactly_excludes_similar_group_names():
    driver = driver_for(
        [hit(), hit(LIVE_UUID), hit(FOREIGN_UUID, 'team-a-other')], live=[LIVE_UUID]
    )
    result = await driver.graph_ops.prune_search_index(driver, 'team-a', dry_run=False)
    assert sum(item.removed for item in result) == 1
    assert sum(item.retained for item in result) == 1
    body = cast(MagicMock, driver.aoss_client).bulk.call_args.kwargs['body']
    assert STALE_UUID in body
    assert LIVE_UUID not in body
    assert FOREIGN_UUID not in body
    assert all(
        'DELETE' not in call.args[0] for call in cast(MagicMock, driver.client).query.call_args_list
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('index', text_aoss_indices, ids=lambda index: index['index_name'])
async def test_each_keyword_index_checks_its_graph_population(index):
    driver = driver_for([])
    cast(MagicMock, driver.aoss_client).search.side_effect = lambda *, index, body: response(
        [{**hit(), '_index': target}] if index == target and 'search_after' not in body else []
    )
    target = index['index_name']
    result = await driver.graph_ops.prune_search_index(driver, 'team-a', dry_run=False)
    assert sum(item.removed for item in result) == 1
    graph_read = cast(MagicMock, driver.client).query.call_args
    assert index['graph_match'] in graph_read.args[0]
    assert graph_read.kwargs['params'] == {'uuids': [STALE_UUID], 'group_id': 'team-a'}
    assert (
        json.loads(cast(MagicMock, driver.aoss_client).bulk.call_args.kwargs['body'])['delete'][
            '_index'
        ]
        == target
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('group_ids', [None, ['']])
async def test_clear_supports_existing_empty_default_group(group_ids):
    driver = driver_for([hit(group='')])
    await clear_data(driver, group_ids)
    cast(MagicMock, driver.aoss_client).bulk.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'bad_response',
    [
        None,
        {},
        {'timed_out': True},
        {'timed_out': False, '_shards': {'failed': 1}},
        {'timed_out': False, '_shards': {'failed': 0}, 'hits': None},
        {**response([hit()]), 'terminated_early': True},
    ],
)
async def test_incomplete_search_refuses_mutation(bad_response):
    driver = driver_for([])
    cast(MagicMock, driver.aoss_client).search.side_effect = None
    cast(MagicMock, driver.aoss_client).search.return_value = bad_response
    with pytest.raises(KeywordIndexCleanupError):
        await driver.graph_ops.prune_search_index(driver, 'team-a', dry_run=False)
    cast(MagicMock, driver.aoss_client).bulk.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('_source', {}),
        ('_index', 'foreign_index'),
        ('_id', 'mismatch'),
        ('_seq_no', None),
        ('_seq_no', True),
        ('_seq_no', -1),
        ('_primary_term', 0),
        ('sort', [LIVE_UUID]),
    ],
)
async def test_incomplete_identity_or_preconditions_refuse_mutation(field, value):
    malformed = hit()
    malformed[field] = value
    driver = driver_for([malformed])
    with pytest.raises(KeywordIndexCleanupError):
        await driver.graph_ops.prune_search_index(driver, 'team-a', dry_run=False)
    cast(MagicMock, driver.aoss_client).bulk.assert_not_called()


@pytest.mark.asyncio
async def test_graph_read_failure_prevents_bulk_write():
    driver = driver_for([hit()])
    cast(MagicMock, driver.client).query.side_effect = RuntimeError('private-provider-detail')
    with pytest.raises(KeywordIndexCleanupError, match='error_type=RuntimeError') as raised:
        await driver.graph_ops.prune_search_index(driver, 'team-a', dry_run=False)
    assert 'private-provider-detail' not in str(raised.value)
    cast(MagicMock, driver.aoss_client).bulk.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [403, 409, 429, 500])
async def test_failed_conditional_delete_has_no_unconditional_fallback(status):
    driver = driver_for([hit()])
    cast(MagicMock, driver.aoss_client).bulk.return_value = {
        'errors': True,
        'items': [{'delete': {'status': status, '_id': STALE_UUID}}],
    }
    with pytest.raises(KeywordIndexCleanupError, match='Conditional keyword cleanup failed'):
        await driver.graph_ops.prune_search_index(driver, 'team-a', dry_run=False)
    cast(MagicMock, driver.aoss_client).bulk.assert_called_once()


@pytest.mark.asyncio
async def test_bulk_does_not_claim_success_for_unaccounted_documents():
    driver = driver_for([hit(), hit(LIVE_UUID)])
    with pytest.raises(KeywordIndexCleanupError, match='account for every'):
        await driver.graph_ops.prune_search_index(driver, 'team-a', dry_run=False)


@pytest.mark.asyncio
async def test_bounded_pages_continue_past_foreign_documents(monkeypatch):
    monkeypatch.setattr('graphiti_core.driver.neptune_driver.MAX_AOSS_QUERY_SIZE', 1)
    driver = driver_for([])
    pages = {None: [hit(group='team-a-other')], STALE_UUID: [hit(LIVE_UUID)], LIVE_UUID: []}
    cast(MagicMock, driver.aoss_client).search.side_effect = lambda *, index, body: response(
        pages[body.get('search_after', [None])[0]] if index == 'edge_name_and_fact' else []
    )
    result = await driver.graph_ops.prune_search_index(driver, 'team-a', dry_run=False)
    assert sum(item.examined for item in result) == 1
    assert sum(item.removed for item in result) == 1
    body = json.loads(cast(MagicMock, driver.aoss_client).bulk.call_args.kwargs['body'])
    assert body['delete']['_id'] == LIVE_UUID


@pytest.mark.asyncio
async def test_repeated_cursor_refuses_a_second_delete(monkeypatch):
    monkeypatch.setattr('graphiti_core.driver.neptune_driver.MAX_AOSS_QUERY_SIZE', 1)
    driver = driver_for([])
    cast(MagicMock, driver.aoss_client).search.side_effect = lambda *, index, body: response(
        [hit()] if index == 'edge_name_and_fact' else []
    )
    with pytest.raises(KeywordIndexCleanupError, match='cursor did not advance'):
        await driver.graph_ops.prune_search_index(driver, 'team-a', dry_run=False)
    cast(MagicMock, driver.aoss_client).bulk.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['clear', 'repair'])
async def test_cancellation_joins_inflight_index_mutation(operation):
    driver = driver_for([hit()])
    started, release = threading.Event(), threading.Event()
    result = cast(MagicMock, driver.aoss_client).bulk.return_value

    def blocked_bulk(**kwargs):
        started.set()
        assert release.wait(5)
        return result

    cast(MagicMock, driver.aoss_client).bulk.side_effect = blocked_bulk
    task = asyncio.create_task(
        clear_data(driver, ['team-a'])
        if operation == 'clear'
        else driver.graph_ops.prune_search_index(driver, 'team-a', dry_run=False)
    )
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
    cast(MagicMock, driver.aoss_client).bulk.assert_called_once()


@pytest.mark.asyncio
async def test_dry_run_does_not_write_graph_or_search_index():
    driver = driver_for([hit()])
    result = await driver.graph_ops.prune_search_index(driver, 'team-a', dry_run=True)
    assert sum(item.orphaned for item in result) == 1
    assert sum(item.removed for item in result) == 0
    cast(MagicMock, driver.aoss_client).bulk.assert_not_called()
    assert all(
        'DELETE' not in call.args[0] for call in cast(MagicMock, driver.client).query.call_args_list
    )


@pytest.mark.asyncio
async def test_episode_tombstone_is_not_retained_as_a_search_candidate():
    driver = driver_for([])
    cast(MagicMock, driver.aoss_client).search.side_effect = lambda *, index, body: response(
        [{**hit(), '_index': 'episode_content'}] if index == 'episode_content' else []
    )
    cast(MagicMock, driver.client).query.side_effect = lambda query, params: (
        [] if 'coalesce(record.opr_deleted, false) = false' in query else [{'uuid': STALE_UUID}]
    )
    result = await driver.graph_ops.prune_search_index(driver, 'team-a', dry_run=False)
    assert sum(item.removed for item in result) == 1
    assert sum(item.retained for item in result) == 0
    assert all(
        'DELETE' not in call.args[0] for call in cast(MagicMock, driver.client).query.call_args_list
    )


@pytest.mark.asyncio
async def test_analyzer_empty_group_keeps_exact_scope_without_false_empty_result():
    driver = driver_for([])

    def search(*, index, body):
        clause = body['query']['match_phrase']['group_id']
        assert clause == {'query': '---', 'zero_terms_query': 'all'}
        return response(
            [hit(group='---'), hit(LIVE_UUID, 'other')] if index == 'edge_name_and_fact' else []
        )

    cast(MagicMock, driver.aoss_client).search.side_effect = search
    result = await driver.graph_ops.prune_search_index(driver, '---', dry_run=False)
    assert sum(item.removed for item in result) == 1
    assert LIVE_UUID not in cast(MagicMock, driver.aoss_client).bulk.call_args.kwargs['body']


@pytest.mark.asyncio
async def test_already_absent_document_is_not_counted_as_a_new_removal():
    driver = driver_for([hit()])
    cast(MagicMock, driver.aoss_client).bulk.return_value = {
        'errors': True,
        'items': [{'delete': {'status': 404, '_id': STALE_UUID, 'result': 'not_found'}}],
    }
    result = await driver.graph_ops.prune_search_index(driver, 'team-a', dry_run=False)
    assert sum(item.removed for item in result) == 0
    assert sum(item.orphaned for item in result) == 1
