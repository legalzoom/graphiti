"""Tests for Neptune driver async behavior."""

from __future__ import annotations

import asyncio
import gc
import threading
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.async_limiter import AsyncCapacityLimiter, AsyncCapacityOverloadedError
from graphiti_core.driver import neptune_driver as neptune_driver_module
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.neptune.operations import search_ops as neptune_search_ops
from graphiti_core.driver.neptune.operations.search_ops import NeptuneSearchOperations
from graphiti_core.driver.neptune_driver import (
    AOSS_QUERY_CONCURRENCY,
    MAX_AOSS_QUERY_SIZE,
    NEPTUNE_BOTO_CONFIG,
    AossProjectionError,
    NeptuneAnalyticsClient,
    NeptuneDatabaseClient,
    NeptuneDriver,
    VectorIndexConfigurationError,
    VectorIndexUnsupportedError,
    cosine_similarity_from_knn_score,
)
from graphiti_core.edges import EntityEdge
from graphiti_core.errors import NodeGroupMismatchError
from graphiti_core.graphiti import Graphiti
from graphiti_core.helpers import EPISODE_AOSS_WRITE_VERSION
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.search import search_utils
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    run_neptune_similarity_pipeline,
    run_neptune_similarity_scorer,
    score_neptune_edge_match_records,
    score_neptune_similarity_records,
)
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk_tx


def _episode() -> EpisodicNode:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return EpisodicNode(
        name='episode',
        group_id='group',
        source=EpisodeType.text,
        source_description='test',
        content='content',
        valid_at=now,
        created_at=now,
    )


class TestNeptuneDriverAsyncBoundary:
    @pytest.mark.asyncio
    async def test_execute_query_does_not_block_event_loop(self):
        driver = object.__new__(NeptuneDriver)

        def slow_run_query(query, params):
            time.sleep(0.05)
            return [{'ok': True}], None, None

        driver._run_query = slow_run_query

        tick = asyncio.Event()

        async def mark_event_loop_progress():
            await asyncio.sleep(0.01)
            tick.set()

        query_task = asyncio.create_task(driver.execute_query('RETURN 1'))
        marker_task = asyncio.create_task(mark_event_loop_progress())

        await asyncio.wait_for(tick.wait(), timeout=0.03)
        assert not query_task.done()

        result, _, _ = await query_task
        await marker_task
        assert result == [{'ok': True}]

    @pytest.mark.asyncio
    async def test_execute_query_list_preserves_last_result(self):
        driver = object.__new__(NeptuneDriver)
        calls = []

        def run_query(query, params):
            calls.append((query, params))
            return [{'query': query}], None, None

        driver._run_query = run_query

        result, _, _ = await driver.execute_query(
            [
                ('RETURN 1', {'first': True}),
                ('RETURN 2', {'second': True}),
            ]
        )

        assert calls == [
            ('RETURN 1', {'first': True}),
            ('RETURN 2', {'second': True}),
        ]
        assert result == [{'query': 'RETURN 2'}]

    @pytest.mark.asyncio
    async def test_aoss_search_does_not_block_event_loop_and_uses_requested_limit(self):
        driver = object.__new__(NeptuneDriver)
        driver.aoss_client = MagicMock()
        captured_request = {}

        def slow_search(*, body, index):
            captured_request.update(body=body, index=index)
            time.sleep(0.05)
            return {'hits': {'total': {'value': 0}, 'hits': []}}

        driver.aoss_client.search.side_effect = slow_search
        index_template = {
            'index_name': 'node_name_and_summary',
            'query': {
                'query': {'multi_match': {'query': '', 'fields': ['name', 'summary']}},
                'size': 10,
            },
        }

        tick = asyncio.Event()

        async def mark_event_loop_progress():
            await asyncio.sleep(0.01)
            tick.set()

        with patch('graphiti_core.driver.neptune_driver.aoss_indices', [index_template]):
            search_task = asyncio.create_task(
                driver.run_aoss_query('node_name_and_summary', 'needle', limit=37)
            )
            marker_task = asyncio.create_task(mark_event_loop_progress())

            await asyncio.wait_for(tick.wait(), timeout=0.03)
            assert not search_task.done()
            result = await search_task
            await marker_task

        assert result == {'hits': {'total': {'value': 0}, 'hits': []}}
        assert captured_request == {
            'body': {
                'query': {
                    'multi_match': {
                        'query': 'needle',
                        'fields': ['name', 'summary'],
                    }
                },
                'size': 37,
            },
            'index': 'node_name_and_summary',
        }
        assert index_template['query']['query']['multi_match']['query'] == ''

    @pytest.mark.asyncio
    async def test_concurrent_aoss_searches_use_isolated_query_bodies(self):
        driver = object.__new__(NeptuneDriver)
        driver.aoss_client = MagicMock()
        barrier = threading.Barrier(2)
        captured_bodies = []

        def search(*, body, index):
            barrier.wait(timeout=1)
            captured_bodies.append((body, index))
            return {'query': body['query']['multi_match']['query']}

        driver.aoss_client.search.side_effect = search
        index_template = {
            'index_name': 'node_name_and_summary',
            'query': {
                'query': {'multi_match': {'query': '', 'fields': ['name']}},
                'size': 10,
            },
        }

        with patch('graphiti_core.driver.neptune_driver.aoss_indices', [index_template]):
            results = await asyncio.gather(
                driver.run_aoss_query('node_name_and_summary', 'first', limit=3),
                driver.run_aoss_query('node_name_and_summary', 'second', limit=7),
            )

        assert results == [{'query': 'first'}, {'query': 'second'}]
        requests_by_text = {
            body['query']['multi_match']['query']: (body, index) for body, index in captured_bodies
        }
        assert requests_by_text['first'][0]['size'] == 3
        assert requests_by_text['second'][0]['size'] == 7
        assert all(index == 'node_name_and_summary' for _, index in captured_bodies)
        assert index_template['query']['query']['multi_match']['query'] == ''

    @pytest.mark.asyncio
    async def test_cancelled_aoss_waiter_leaves_in_flight_search_isolated(self):
        driver = object.__new__(NeptuneDriver)
        driver.aoss_client = MagicMock()
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def search(*, body, index):
            started.set()
            release.wait(timeout=1)
            finished.set()
            return {'body': body, 'index': index}

        driver.aoss_client.search.side_effect = search
        index_template = {
            'index_name': 'node_name_and_summary',
            'query': {
                'query': {'multi_match': {'query': '', 'fields': ['name']}},
                'size': 10,
            },
        }

        with patch('graphiti_core.driver.neptune_driver.aoss_indices', [index_template]):
            search_task = asyncio.create_task(
                driver.run_aoss_query('node_name_and_summary', 'cancelled')
            )
            assert await asyncio.to_thread(started.wait, 1)

            search_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await search_task

            assert not finished.is_set()
            assert index_template['query']['query']['multi_match']['query'] == ''
            release.set()
            assert await asyncio.to_thread(finished.wait, 1)

    @pytest.mark.asyncio
    async def test_aoss_search_caps_requested_result_size(self):
        driver = object.__new__(NeptuneDriver)
        driver.aoss_client = MagicMock()
        driver.aoss_client.search.return_value = {'hits': {'total': {'value': 0}, 'hits': []}}
        index_template = {
            'index_name': 'node_name_and_summary',
            'query': {
                'query': {'multi_match': {'query': '', 'fields': ['name']}},
                'size': 10,
            },
        }

        with patch('graphiti_core.driver.neptune_driver.aoss_indices', [index_template]):
            await driver.run_aoss_query(
                'node_name_and_summary', 'needle', limit=MAX_AOSS_QUERY_SIZE + 1
            )

        assert driver.aoss_client.search.call_args.kwargs['body']['size'] == MAX_AOSS_QUERY_SIZE

    @pytest.mark.asyncio
    async def test_awaited_aoss_failure_releases_capacity_for_replacement(self):
        driver = object.__new__(NeptuneDriver)
        driver.aoss_client = MagicMock()
        capacity = AsyncCapacityLimiter(1, max_waiters=1)

        def search(*, body, index):
            query_text = body['query']['multi_match']['query']
            if query_text == 'failed':
                raise RuntimeError('expected search failure')
            return {'query': query_text, 'index': index}

        driver.aoss_client.search.side_effect = search
        index_template = {
            'index_name': 'node_name_and_summary',
            'query': {
                'query': {'multi_match': {'query': '', 'fields': ['name']}},
                'size': 10,
            },
        }

        with (
            patch.object(neptune_driver_module, '_AOSS_QUERY_CAPACITY', capacity),
            patch('graphiti_core.driver.neptune_driver.aoss_indices', [index_template]),
        ):
            with pytest.raises(RuntimeError, match='expected search failure'):
                await driver.run_aoss_query('node_name_and_summary', 'failed')

            result = await asyncio.wait_for(
                driver.run_aoss_query('node_name_and_summary', 'replacement'), timeout=1
            )

        assert result == {'query': 'replacement', 'index': 'node_name_and_summary'}

    @pytest.mark.asyncio
    async def test_aoss_submit_failure_releases_capacity_for_replacement(self):
        driver = object.__new__(NeptuneDriver)
        driver.aoss_client = MagicMock()
        driver.aoss_client.search.return_value = {'ok': True}
        capacity = AsyncCapacityLimiter(1, max_waiters=1)
        index_template = {
            'index_name': 'node_name_and_summary',
            'query': {
                'query': {'multi_match': {'query': '', 'fields': ['name']}},
                'size': 10,
            },
        }

        with (
            patch.object(neptune_driver_module, '_AOSS_QUERY_CAPACITY', capacity),
            patch('graphiti_core.driver.neptune_driver.aoss_indices', [index_template]),
        ):
            with (
                patch.object(
                    neptune_driver_module._AOSS_QUERY_EXECUTOR,
                    'submit',
                    side_effect=RuntimeError('expected submit failure'),
                ),
                pytest.raises(RuntimeError, match='expected submit failure'),
            ):
                await driver.run_aoss_query('node_name_and_summary', 'failed')

            result = await asyncio.wait_for(
                driver.run_aoss_query('node_name_and_summary', 'replacement'), timeout=1
            )

        assert result == {'ok': True}

    @pytest.mark.asyncio
    async def test_aoss_cancellation_holds_capacity_until_worker_finishes(self):
        driver = object.__new__(NeptuneDriver)
        driver.aoss_client = MagicMock()
        started = [threading.Event() for _ in range(AOSS_QUERY_CONCURRENCY + 1)]
        release = [threading.Event() for _ in range(AOSS_QUERY_CONCURRENCY + 1)]
        finished = [threading.Event() for _ in range(AOSS_QUERY_CONCURRENCY + 1)]

        def search(*, body, index):
            call_index = int(body['query']['multi_match']['query'])
            started[call_index].set()
            release[call_index].wait(timeout=2)
            finished[call_index].set()
            return {'body': body, 'index': index}

        driver.aoss_client.search.side_effect = search
        index_template = {
            'index_name': 'node_name_and_summary',
            'query': {
                'query': {'multi_match': {'query': '', 'fields': ['name']}},
                'size': 10,
            },
        }
        tasks = []
        third = None
        try:
            with patch('graphiti_core.driver.neptune_driver.aoss_indices', [index_template]):
                tasks = [
                    asyncio.create_task(driver.run_aoss_query('node_name_and_summary', str(i)))
                    for i in range(AOSS_QUERY_CONCURRENCY)
                ]
                for event in started[:AOSS_QUERY_CONCURRENCY]:
                    assert await asyncio.to_thread(event.wait, 1)

                third = asyncio.create_task(
                    driver.run_aoss_query('node_name_and_summary', str(AOSS_QUERY_CONCURRENCY))
                )
                await asyncio.sleep(0)
                assert not started[AOSS_QUERY_CONCURRENCY].is_set()

                tasks[0].cancel()
                with pytest.raises(asyncio.CancelledError):
                    await tasks[0]
                assert not started[AOSS_QUERY_CONCURRENCY].is_set()

                release[0].set()
                assert await asyncio.to_thread(finished[0].wait, 1)
                assert await asyncio.to_thread(started[AOSS_QUERY_CONCURRENCY].wait, 1)
        finally:
            for event in release:
                event.set()
            remaining = [task for task in tasks if not task.done()]
            if third is not None:
                remaining.append(third)
            await asyncio.gather(*remaining, return_exceptions=True)
            for index, event in enumerate(started):
                if event.is_set():
                    assert await asyncio.to_thread(finished[index].wait, 1)

    def test_aoss_cancellation_releases_capacity_after_origin_loop_closes(self):
        driver = object.__new__(NeptuneDriver)
        driver.aoss_client = MagicMock()
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def search(*, body, index):
            query_text = body['query']['multi_match']['query']
            if query_text == 'cancelled':
                started.set()
                release.wait(timeout=2)
                finished.set()
            return {'query': query_text, 'index': index}

        driver.aoss_client.search.side_effect = search
        index_template = {
            'index_name': 'node_name_and_summary',
            'query': {
                'query': {'multi_match': {'query': '', 'fields': ['name']}},
                'size': 10,
            },
        }
        capacity = AsyncCapacityLimiter(1)
        origin_loop = asyncio.new_event_loop()
        origin_task = None

        async def wait_until_started():
            deadline = asyncio.get_running_loop().time() + 1
            while not started.is_set():
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError('AOSS worker did not start')
                await asyncio.sleep(0.001)

        async def run_replacement():
            return await asyncio.wait_for(
                driver.run_aoss_query('node_name_and_summary', 'replacement'), timeout=1
            )

        try:
            with (
                patch.object(neptune_driver_module, '_AOSS_QUERY_CAPACITY', capacity),
                patch('graphiti_core.driver.neptune_driver.aoss_indices', [index_template]),
            ):
                origin_task = origin_loop.create_task(
                    driver.run_aoss_query('node_name_and_summary', 'cancelled')
                )
                origin_loop.run_until_complete(wait_until_started())

                origin_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    origin_loop.run_until_complete(origin_task)
                assert not finished.is_set()

                origin_loop.close()
                release.set()
                assert finished.wait(timeout=1)

                result = asyncio.run(run_replacement())

            assert result == {'query': 'replacement', 'index': 'node_name_and_summary'}
        finally:
            release.set()
            if not origin_loop.is_closed():
                if origin_task is not None and not origin_task.done():
                    origin_task.cancel()
                    origin_loop.run_until_complete(
                        asyncio.gather(origin_task, return_exceptions=True)
                    )
                origin_loop.close()

    def test_cancelled_aoss_failure_is_logged_without_unretrieved_wrapper(self, caplog):
        driver = object.__new__(NeptuneDriver)
        driver.aoss_client = MagicMock()
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        exception_contexts = []

        def search(*, body, index):
            started.set()
            release.wait(timeout=2)
            finished.set()
            raise RuntimeError('detached AOSS failure')

        driver.aoss_client.search.side_effect = search
        index_template = {
            'index_name': 'node_name_and_summary',
            'query': {
                'query': {'multi_match': {'query': '', 'fields': ['name']}},
                'size': 10,
            },
        }
        capacity = AsyncCapacityLimiter(1)
        origin_loop = asyncio.new_event_loop()
        origin_loop.set_exception_handler(lambda _loop, context: exception_contexts.append(context))

        async def cancel_and_finish():
            task = asyncio.create_task(driver.run_aoss_query('node_name_and_summary', 'cancelled'))
            deadline = asyncio.get_running_loop().time() + 1
            while not started.is_set():
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError('AOSS worker did not start')
                await asyncio.sleep(0.001)

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            release.set()
            deadline = asyncio.get_running_loop().time() + 1
            while not finished.is_set():
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError('AOSS worker did not finish')
                await asyncio.sleep(0.001)

            while 'Detached AOSS query failed after caller cancellation' not in caplog.messages:
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError('detached AOSS failure was not logged')
                await asyncio.sleep(0.001)

            for _ in range(3):
                await asyncio.sleep(0)

        try:
            caplog.set_level('ERROR', logger=neptune_driver_module.__name__)
            with (
                patch.object(neptune_driver_module, '_AOSS_QUERY_CAPACITY', capacity),
                patch('graphiti_core.driver.neptune_driver.aoss_indices', [index_template]),
            ):
                origin_loop.run_until_complete(cancel_and_finish())
                gc.collect()
                origin_loop.run_until_complete(asyncio.sleep(0))

            assert 'Detached AOSS query failed after caller cancellation' in caplog.messages
            assert not exception_contexts
        finally:
            release.set()
            origin_loop.close()

    @pytest.mark.asyncio
    async def test_cancelled_aoss_waiter_storm_is_removed_immediately(self):
        leases = [
            await neptune_driver_module._AOSS_QUERY_CAPACITY.acquire()
            for _ in range(AOSS_QUERY_CONCURRENCY)
        ]
        waiters = [
            asyncio.create_task(neptune_driver_module._AOSS_QUERY_CAPACITY.acquire())
            for _ in range(200)
        ]
        try:
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            for waiter in waiters:
                waiter.cancel()
            await asyncio.gather(*waiters, return_exceptions=True)

            assert not neptune_driver_module._AOSS_QUERY_CAPACITY._waiters
        finally:
            for lease in leases:
                lease.release()

    @pytest.mark.asyncio
    async def test_aoss_vector_mutation_does_not_block_event_loop(self):
        driver = object.__new__(NeptuneDriver)

        def slow_save(_name, data):
            time.sleep(0.05)
            return len(data)

        driver.save_vector_to_aoss = slow_save  # type: ignore[method-assign]
        tick = asyncio.Event()

        async def mark_event_loop_progress():
            await asyncio.sleep(0.01)
            tick.set()

        mutation = asyncio.create_task(
            driver.save_vector_to_aoss_async('node_name_embedding', [{'uuid': 'node-1'}])
        )
        marker = asyncio.create_task(mark_event_loop_progress())

        await asyncio.wait_for(tick.wait(), timeout=0.03)
        assert not mutation.done()
        assert await mutation == 1
        await marker

    @pytest.mark.asyncio
    async def test_cancelled_aoss_mutation_holds_capacity_and_bounds_waiters(self):
        driver = object.__new__(NeptuneDriver)
        capacity = AsyncCapacityLimiter(1, max_waiters=1)
        started = [threading.Event(), threading.Event()]
        release = [threading.Event(), threading.Event()]
        finished = [threading.Event(), threading.Event()]

        def save(_name, data):
            index = int(data[0]['uuid'])
            started[index].set()
            release[index].wait(timeout=2)
            finished[index].set()
            return len(data)

        driver.save_vector_to_aoss = save  # type: ignore[method-assign]
        first = None
        second = None
        try:
            with patch.object(neptune_driver_module, '_AOSS_MUTATION_CAPACITY', capacity):
                first = asyncio.create_task(
                    driver.save_vector_to_aoss_async('node_name_embedding', [{'uuid': '0'}])
                )
                assert await asyncio.to_thread(started[0].wait, 1)

                second = asyncio.create_task(
                    driver.save_vector_to_aoss_async('node_name_embedding', [{'uuid': '1'}])
                )
                await asyncio.sleep(0)
                with pytest.raises(AsyncCapacityOverloadedError):
                    await driver.save_vector_to_aoss_async('node_name_embedding', [{'uuid': '2'}])

                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                assert not started[1].is_set()

                release[0].set()
                assert await asyncio.to_thread(finished[0].wait, 1)
                assert await asyncio.to_thread(started[1].wait, 1)
        finally:
            for event in release:
                event.set()
            remaining = [task for task in (first, second) if task is not None and not task.done()]
            await asyncio.gather(*remaining, return_exceptions=True)
            for index, event in enumerate(started):
                if event.is_set():
                    assert await asyncio.to_thread(finished[index].wait, 1)


class TestNeptuneSimilarityCapacity:
    @pytest.mark.asyncio
    async def test_third_pipeline_waits_for_one_of_two_active_pipelines(self):
        started = [asyncio.Event() for _ in range(3)]
        release = [asyncio.Event() for _ in range(3)]

        async def operation(index):
            started[index].set()
            await release[index].wait()
            return index

        tasks = [
            asyncio.create_task(run_neptune_similarity_pipeline(lambda i=i: operation(i)))
            for i in range(2)
        ]
        third_task = None
        try:
            await asyncio.wait_for(asyncio.gather(started[0].wait(), started[1].wait()), 1)
            third_task = asyncio.create_task(run_neptune_similarity_pipeline(lambda: operation(2)))
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not started[2].is_set()

            release[0].set()
            await asyncio.wait_for(started[2].wait(), 1)
        finally:
            for event in release:
                event.set()
            if third_task is not None:
                tasks.append(third_task)
            await asyncio.gather(*tasks, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_cancel_during_fetch_holds_capacity_and_skips_later_phases(self, monkeypatch):
        capacity = AsyncCapacityLimiter(1, max_waiters=1)
        fetch_started = threading.Event()
        release_fetch = threading.Event()
        fetch_finished = threading.Event()
        scorer_called = threading.Event()
        hydration_called = threading.Event()
        replacement_started = asyncio.Event()
        release_replacement = asyncio.Event()
        query_count = 0

        def run_query(query, params):
            nonlocal query_count
            query_count += 1
            if query_count == 1:
                fetch_started.set()
                release_fetch.wait(timeout=2)
                fetch_finished.set()
                return ([{'id': 7, 'embedding': '1.0,0.0'}], None, None)
            hydration_called.set()
            return ([{'uuid': 'hydrated'}], None, None)

        def scorer(*args):
            scorer_called.set()
            return [{'id': 7, 'score': 1.0}]

        async def replacement_operation():
            replacement_started.set()
            await release_replacement.wait()

        driver = object.__new__(NeptuneDriver)
        driver._run_query = run_query
        monkeypatch.setattr(search_utils, '_NEPTUNE_SIMILARITY_CAPACITY', capacity)
        monkeypatch.setattr(neptune_search_ops, 'score_neptune_similarity_records', scorer)

        search_task = asyncio.create_task(
            NeptuneSearchOperations().node_similarity_search(
                driver,
                [1.0, 0.0],
                SearchFilters(),
            )
        )
        replacement = None
        try:
            assert await asyncio.to_thread(fetch_started.wait, 1)
            search_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await search_task

            replacement = asyncio.create_task(
                run_neptune_similarity_pipeline(replacement_operation)
            )
            await asyncio.sleep(0)
            assert not replacement_started.is_set()

            release_fetch.set()
            assert await asyncio.to_thread(fetch_finished.wait, 1)
            await asyncio.wait_for(replacement_started.wait(), 1)
            assert not scorer_called.is_set()
            assert not hydration_called.is_set()
        finally:
            release_fetch.set()
            release_replacement.set()
            remaining = []
            if not search_task.done():
                remaining.append(search_task)
            if replacement is not None:
                remaining.append(replacement)
            await asyncio.gather(*remaining, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_cancel_during_scorer_holds_capacity_until_worker_finishes(self):
        scorer_started = threading.Event()
        release_scorer = threading.Event()
        scorer_finished = threading.Event()
        second_started = asyncio.Event()
        release_second = asyncio.Event()
        third_started = asyncio.Event()
        release_third = asyncio.Event()

        def scorer():
            scorer_started.set()
            release_scorer.wait(timeout=1)
            scorer_finished.set()
            return []

        async def score_operation():
            return await run_neptune_similarity_scorer(scorer)

        async def held_operation(started, release):
            started.set()
            await release.wait()

        first = asyncio.create_task(run_neptune_similarity_pipeline(score_operation))
        second = asyncio.create_task(
            run_neptune_similarity_pipeline(lambda: held_operation(second_started, release_second))
        )
        third = None
        try:
            assert await asyncio.to_thread(scorer_started.wait, 1)
            await asyncio.wait_for(second_started.wait(), 1)
            third = asyncio.create_task(
                run_neptune_similarity_pipeline(
                    lambda: held_operation(third_started, release_third)
                )
            )
            await asyncio.sleep(0)

            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert not third_started.is_set()

            release_scorer.set()
            assert await asyncio.to_thread(scorer_finished.wait, 1)
            await asyncio.wait_for(third_started.wait(), 1)
        finally:
            release_scorer.set()
            release_second.set()
            release_third.set()
            remaining = [second]
            if third is not None:
                remaining.append(third)
            await asyncio.gather(*remaining, return_exceptions=True)

    def test_cancelled_scorer_releases_capacity_after_origin_loop_closes(self):
        capacity = AsyncCapacityLimiter(1, max_waiters=1)
        scorer_started = threading.Event()
        release_scorer = threading.Event()
        scorer_finished = threading.Event()
        replacement_started = threading.Event()
        replacement_results: list[str] = []
        replacement_errors: list[BaseException] = []
        origin_loop = asyncio.new_event_loop()

        def scorer():
            scorer_started.set()
            release_scorer.wait(timeout=2)
            scorer_finished.set()
            return []

        async def score_operation():
            return await run_neptune_similarity_scorer(scorer)

        async def cancel_after_start():
            task = asyncio.create_task(run_neptune_similarity_pipeline(score_operation))
            deadline = asyncio.get_running_loop().time() + 1
            while not scorer_started.is_set():
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError('similarity scorer did not start')
                await asyncio.sleep(0.001)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        async def replacement_operation():
            replacement_started.set()
            return 'replacement'

        def run_replacement():
            try:
                replacement_results.append(
                    asyncio.run(run_neptune_similarity_pipeline(replacement_operation))
                )
            except BaseException as error:
                replacement_errors.append(error)

        replacement_thread = threading.Thread(target=run_replacement)
        try:
            with patch.object(search_utils, '_NEPTUNE_SIMILARITY_CAPACITY', capacity):
                origin_loop.run_until_complete(cancel_after_start())
                assert not scorer_finished.is_set()
                origin_loop.close()

                replacement_thread.start()
                assert not replacement_started.wait(timeout=0.05)

                release_scorer.set()
                assert scorer_finished.wait(timeout=1)
                assert replacement_started.wait(timeout=1)
                replacement_thread.join(timeout=1)

            assert not replacement_thread.is_alive()
            assert replacement_results == ['replacement']
            assert not replacement_errors
        finally:
            release_scorer.set()
            if not origin_loop.is_closed():
                origin_loop.close()
            if replacement_thread.ident is not None:
                replacement_thread.join(timeout=1)

    def test_cancelled_query_releases_capacity_after_origin_loop_closes(self):
        capacity = AsyncCapacityLimiter(1, max_waiters=1)
        query_started = threading.Event()
        release_query = threading.Event()
        query_finished = threading.Event()
        replacement_started = threading.Event()
        replacement_results: list[str] = []
        replacement_errors: list[BaseException] = []
        origin_loop = asyncio.new_event_loop()

        def run_query(query, params):
            query_started.set()
            release_query.wait(timeout=2)
            query_finished.set()
            return ([{'ok': True}], None, None)

        driver = object.__new__(NeptuneDriver)
        driver._run_query = run_query

        async def query_operation():
            return await driver.execute_query('RETURN 1')

        async def cancel_after_start():
            task = asyncio.create_task(run_neptune_similarity_pipeline(query_operation))
            deadline = asyncio.get_running_loop().time() + 1
            while not query_started.is_set():
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError('Neptune query did not start')
                await asyncio.sleep(0.001)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        async def replacement_operation():
            replacement_started.set()
            return 'replacement'

        def run_replacement():
            try:
                replacement_results.append(
                    asyncio.run(run_neptune_similarity_pipeline(replacement_operation))
                )
            except BaseException as error:
                replacement_errors.append(error)

        replacement_thread = threading.Thread(target=run_replacement)
        try:
            with patch.object(search_utils, '_NEPTUNE_SIMILARITY_CAPACITY', capacity):
                origin_loop.run_until_complete(cancel_after_start())
                assert not query_finished.is_set()
                origin_loop.close()

                replacement_thread.start()
                assert not replacement_started.wait(timeout=0.05)

                release_query.set()
                assert query_finished.wait(timeout=1)
                assert replacement_started.wait(timeout=1)
                replacement_thread.join(timeout=1)

            assert not replacement_thread.is_alive()
            assert replacement_results == ['replacement']
            assert not replacement_errors
        finally:
            release_query.set()
            if not origin_loop.is_closed():
                origin_loop.close()
            if replacement_thread.ident is not None:
                replacement_thread.join(timeout=1)

    @pytest.mark.asyncio
    async def test_cancelled_queued_acquisition_does_not_leak_capacity(self):
        started = [asyncio.Event() for _ in range(4)]
        release = [asyncio.Event() for _ in range(4)]

        async def operation(index):
            started[index].set()
            await release[index].wait()
            return index

        first = asyncio.create_task(run_neptune_similarity_pipeline(lambda: operation(0)))
        second = asyncio.create_task(run_neptune_similarity_pipeline(lambda: operation(1)))
        replacement = None
        try:
            await asyncio.wait_for(asyncio.gather(started[0].wait(), started[1].wait()), 1)
            queued = asyncio.create_task(run_neptune_similarity_pipeline(lambda: operation(2)))
            await asyncio.sleep(0)
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            assert not started[2].is_set()

            release[0].set()
            replacement = asyncio.create_task(run_neptune_similarity_pipeline(lambda: operation(3)))
            await asyncio.wait_for(started[3].wait(), 1)
            assert not started[2].is_set()
        finally:
            for event in release:
                event.set()
            remaining = [first, second]
            if replacement is not None:
                remaining.append(replacement)
            await asyncio.gather(*remaining, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_pipeline_exception_releases_capacity(self):
        async def fail():
            raise RuntimeError('expected failure')

        with pytest.raises(RuntimeError, match='expected failure'):
            await run_neptune_similarity_pipeline(fail)

        started = [asyncio.Event(), asyncio.Event()]
        release = [asyncio.Event(), asyncio.Event()]

        async def operation(index):
            started[index].set()
            await release[index].wait()

        tasks = [
            asyncio.create_task(run_neptune_similarity_pipeline(lambda i=i: operation(i)))
            for i in range(2)
        ]
        try:
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), 1)
        finally:
            for event in release:
                event.set()
            await asyncio.gather(*tasks, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_direct_neptune_search_holds_capacity_through_hydration(self, monkeypatch):
        capacity = AsyncCapacityLimiter(2, max_waiters=2)
        hydration_started = threading.Event()
        release_hydration = threading.Event()
        holder_started = asyncio.Event()
        release_holder = asyncio.Event()
        third_started = asyncio.Event()
        release_third = asyncio.Event()
        parser_called = threading.Event()
        query_count = 0

        def run_query(query, params):
            nonlocal query_count
            query_count += 1
            if query_count == 1:
                return ([{'id': 7, 'embedding': '1.0,0.0'}], None, None)
            hydration_started.set()
            release_hydration.wait(timeout=2)
            return ([{'uuid': 'hydrated'}], None, None)

        async def held_operation(started, release):
            started.set()
            await release.wait()

        monkeypatch.setattr(
            neptune_search_ops,
            'score_neptune_similarity_records',
            lambda search_vector, records, min_score: [{'id': 7, 'score': 1.0}],
        )
        monkeypatch.setattr(search_utils, '_NEPTUNE_SIMILARITY_CAPACITY', capacity)
        monkeypatch.setattr(
            neptune_search_ops,
            'entity_node_from_record',
            lambda record: parser_called.set() or record,
        )
        driver = object.__new__(NeptuneDriver)
        driver._run_query = run_query

        search_task = asyncio.create_task(
            NeptuneSearchOperations().node_similarity_search(
                driver,
                [1.0, 0.0],
                SearchFilters(),
            )
        )
        holder_task = None
        third_task = None
        try:
            assert await asyncio.to_thread(hydration_started.wait, 1)
            holder_task = asyncio.create_task(
                run_neptune_similarity_pipeline(
                    lambda: held_operation(holder_started, release_holder)
                )
            )
            await asyncio.wait_for(holder_started.wait(), 1)
            third_task = asyncio.create_task(
                run_neptune_similarity_pipeline(
                    lambda: held_operation(third_started, release_third)
                )
            )
            await asyncio.sleep(0)

            search_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await search_task
            assert not third_started.is_set()

            release_hydration.set()
            await asyncio.wait_for(third_started.wait(), 1)
            assert not parser_called.is_set()
        finally:
            release_hydration.set()
            release_holder.set()
            release_third.set()
            remaining = []
            if not search_task.done():
                remaining.append(search_task)
            if holder_task is not None:
                remaining.append(holder_task)
            if third_task is not None:
                remaining.append(third_task)
            await asyncio.gather(*remaining, return_exceptions=True)


def test_score_neptune_similarity_records_parses_and_filters_embeddings():
    results = score_neptune_similarity_records(
        [1.0, 0.0],
        [
            {'id': 'exact', 'embedding': '1.0,0.0'},
            {'id': 'above-threshold', 'embedding': '0.8,0.6'},
            {'id': 'below-threshold', 'embedding': '0.0,1.0'},
            {'id': 'missing', 'embedding': None},
        ],
        min_score=0.5,
    )

    assert [result['id'] for result in results] == ['exact', 'above-threshold']
    assert results[0]['score'] == pytest.approx(1.0)
    assert results[1]['score'] == pytest.approx(0.8)


def test_score_neptune_similarity_records_rejects_wrong_dimension():
    with pytest.raises(ValueError, match='has dimension 1; expected 2'):
        score_neptune_similarity_records(
            [1.0, 0.0],
            [{'id': 'bad', 'embedding': '1.0'}],
            min_score=0.5,
        )


def test_score_neptune_similarity_records_rejects_trailing_delimiter():
    with pytest.raises(ValueError, match='could not convert string to float'):
        score_neptune_similarity_records(
            [1.0, 0.0],
            [{'id': 'bad', 'embedding': '1.0,0.0,'}],
            min_score=0.5,
        )


def test_score_neptune_edge_match_records_preserves_order_and_identifiers():
    results = score_neptune_edge_match_records(
        [
            {
                'id': 7,
                'source_embedding': '1.0,0.0',
                'target_embedding': [1.0, 0.0],
                'search_edge_uuid': 'exact',
            },
            {
                'id': 8,
                'source_embedding': '0.0,1.0',
                'target_embedding': [1.0, 0.0],
                'search_edge_uuid': 'filtered',
            },
        ],
        min_score=0.5,
    )

    assert [result['id'] for result in results] == [7]
    assert [result['uuid'] for result in results] == ['exact']
    assert results[0]['score'] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_neptune_similarity_scoring_does_not_block_event_loop(monkeypatch):
    records = [{'id': str(i), 'embedding': '1.0,0.0'} for i in range(40)]
    executor = MagicMock()
    executor.execute_query = AsyncMock(return_value=(records, None, None))

    def slow_similarity_scoring(search_vector, records, min_score):
        time.sleep(0.08)
        return []

    monkeypatch.setattr(
        neptune_search_ops,
        'score_neptune_similarity_records',
        slow_similarity_scoring,
    )
    tick = asyncio.Event()

    async def mark_event_loop_progress():
        await asyncio.sleep(0.01)
        tick.set()

    search_task = asyncio.create_task(
        NeptuneSearchOperations().node_similarity_search(
            executor,
            [1.0, 0.0],
            SearchFilters(),
            min_score=1.0,
        )
    )
    marker_task = asyncio.create_task(mark_event_loop_progress())

    await asyncio.wait_for(tick.wait(), timeout=0.03)
    assert not search_task.done()
    assert await search_task == []
    await marker_task


@pytest.mark.asyncio
async def test_node_similarity_search_neptune_queries_knn_index_then_fetches_by_uuid():
    """node_similarity_search's NEPTUNE branch is k-NN backed: no client-side embedding
    scan. It queries the node_name_embedding vector index for candidate uuids, then
    fetches the matching nodes from Neptune in uuid order, preserving the knn score
    order."""
    driver = MagicMock()
    driver.provider = GraphProvider.NEPTUNE
    driver.search_interface = None
    driver.run_aoss_knn_query = AsyncMock(
        return_value=[{'id': 'node-2', 'score': 0.95}, {'id': 'node-1', 'score': 0.4}]
    )
    driver.execute_query = AsyncMock(
        return_value=(
            [
                {
                    'uuid': 'node-2',
                    'name': 'n2',
                    'group_id': 'g1',
                    'created_at': datetime(2026, 1, 1, tzinfo=timezone.utc),
                    'summary': '',
                    'labels': ['Entity'],
                    'attributes': {},
                },
                {
                    'uuid': 'node-1',
                    'name': 'n1',
                    'group_id': 'g1',
                    'created_at': datetime(2026, 1, 1, tzinfo=timezone.utc),
                    'summary': '',
                    'labels': ['Entity'],
                    'attributes': {},
                },
            ],
            None,
            None,
        )
    )

    results = await search_utils.node_similarity_search(
        driver,
        [1.0, 0.0],
        SearchFilters(),
        group_ids=['g1'],
        limit=10,
        min_score=0.3,
    )

    driver.run_aoss_knn_query.assert_awaited_once_with(
        'node_name_embedding', [1.0, 0.0], 10, 0.3, ['g1']
    )
    assert driver.execute_query.await_count == 1
    fetch_kwargs = driver.execute_query.await_args.kwargs
    assert fetch_kwargs['ids'] == [{'id': 'node-2', 'score': 0.95}, {'id': 'node-1', 'score': 0.4}]
    assert [n.uuid for n in results] == ['node-2', 'node-1']


@pytest.mark.asyncio
async def test_node_similarity_search_neptune_skips_fetch_when_knn_query_has_no_matches():
    driver = MagicMock()
    driver.provider = GraphProvider.NEPTUNE
    driver.search_interface = None
    driver.run_aoss_knn_query = AsyncMock(return_value=[])
    driver.execute_query = AsyncMock()

    results = await search_utils.node_similarity_search(
        driver, [1.0, 0.0], SearchFilters(), min_score=0.3
    )

    assert results == []
    driver.execute_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_edge_similarity_search_neptune_queries_knn_index_then_fetches_by_uuid():
    driver = MagicMock()
    driver.provider = GraphProvider.NEPTUNE
    driver.search_interface = None
    driver.run_aoss_knn_query = AsyncMock(return_value=[{'id': 'edge-1', 'score': 0.9}])
    driver.execute_query = AsyncMock(
        return_value=(
            [
                {
                    'uuid': 'edge-1',
                    'source_node_uuid': 'n1',
                    'target_node_uuid': 'n2',
                    'group_id': 'g1',
                    'name': 'REL',
                    'fact': 'a fact',
                    'episodes': [],
                    'created_at': datetime(2026, 1, 1, tzinfo=timezone.utc),
                    'expired_at': None,
                    'valid_at': None,
                    'invalid_at': None,
                    'reference_time': None,
                    'attributes': {},
                }
            ],
            None,
            None,
        )
    )

    results = await search_utils.edge_similarity_search(
        driver,
        [1.0, 0.0],
        None,
        None,
        SearchFilters(),
        group_ids=['g1'],
        limit=5,
        min_score=0.3,
    )

    driver.run_aoss_knn_query.assert_awaited_once_with(
        'edge_fact_embedding', [1.0, 0.0], 5, 0.3, ['g1'], None
    )
    assert [e.uuid for e in results] == ['edge-1']


@pytest.mark.asyncio
async def test_edge_similarity_source_filter_without_group_uses_exact_neptune_scan():
    driver = MagicMock()
    driver.provider = GraphProvider.NEPTUNE
    driver.search_interface = None
    driver.vector_search_enabled = True
    driver.run_aoss_knn_query = AsyncMock()
    driver.execute_query = AsyncMock(return_value=([], None, None))

    results = await search_utils.edge_similarity_search(
        driver,
        [1.0, 0.0],
        'source-node',
        None,
        SearchFilters(),
        group_ids=None,
        limit=5,
        min_score=0.3,
    )

    assert results == []
    driver.run_aoss_knn_query.assert_not_awaited()
    query = driver.execute_query.await_args.args[0]
    assert 'n.uuid = $source_uuid' in query
    assert driver.execute_query.await_args.kwargs['source_uuid'] == 'source-node'


def test_episode_aoss_write_uses_external_generation_when_supplied():
    driver = object.__new__(NeptuneDriver)
    driver.aoss_client = MagicMock()
    with (
        patch(
            'graphiti_core.driver.neptune_driver.aoss_indices',
            [
                {
                    'index_name': 'episode_content',
                    'body': {'mappings': {'properties': {'content': {}}}},
                }
            ],
        ),
        patch('graphiti_core.driver.neptune_driver.helpers.bulk', return_value=(1, 0)) as bulk,
    ):
        assert (
            driver.save_to_aoss(
                'episode_content',
                [{'uuid': 'episode-id', 'content': '', '_version': 42}],
            )
            == 1
        )

    assert bulk.call_args.args[1] == [
        {
            '_index': 'episode_content',
            '_id': 'episode-id',
            '_version': 42,
            '_version_type': 'external_gte',
            'content': '',
        }
    ]


@pytest.mark.asyncio
async def test_single_episode_write_uses_external_generation():
    driver = MagicMock()
    driver.provider = GraphProvider.NEPTUNE
    driver.graph_operations_interface = None
    driver.execute_query = AsyncMock(return_value=([{'uuid': 'episode-id'}], None, None))

    await _episode().save(driver)

    payload = driver.save_to_aoss.call_args.args[1][0]
    assert payload['_version'] == EPISODE_AOSS_WRITE_VERSION


@pytest.mark.asyncio
async def test_bulk_episode_write_uses_external_generation():
    driver = MagicMock()
    driver.provider = GraphProvider.NEPTUNE
    driver.graph_operations_interface = None
    tx = MagicMock()
    tx.run = AsyncMock(return_value=([{'uuid': 'episode-id'}], None, None))

    await add_nodes_and_edges_bulk_tx(
        tx,
        [_episode()],
        [],
        [],
        [],
        MagicMock(),
        driver,
    )

    payload = driver.save_to_aoss.call_args_list[0].args[1][0]
    assert payload['_version'] == EPISODE_AOSS_WRITE_VERSION


@pytest.mark.asyncio
async def test_legacy_edge_save_does_not_project_an_unpersisted_edge():
    driver = MagicMock()
    driver.provider = GraphProvider.NEPTUNE
    driver.graph_operations_interface = None
    driver.execute_query = AsyncMock(return_value=([], None, None))
    driver.save_vector_to_aoss_async = AsyncMock()
    edge = EntityEdge(
        uuid='edge-1',
        name='RELATES_TO',
        fact='a fact',
        fact_embedding=[1.0, 0.0],
        group_id='g1',
        source_node_uuid='source',
        target_node_uuid='target',
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    with pytest.raises(NodeGroupMismatchError, match='already owned by another group'):
        await edge.save(driver)

    driver.save_to_aoss.assert_not_called()
    driver.save_vector_to_aoss_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_bulk_node_write_rejects_partial_neptune_result():
    driver = MagicMock()
    driver.provider = GraphProvider.NEPTUNE
    driver.graph_operations_interface = None
    driver.save_vector_to_aoss_async = AsyncMock()
    tx = MagicMock()
    tx.run = AsyncMock(
        side_effect=[
            ([], None, None),
            ([{'uuid': 'node-1', 'projection_version': 7}], None, None),
        ]
    )
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    nodes = [
        EntityNode(
            uuid='node-1',
            name='persisted',
            name_embedding=[1.0, 0.0],
            group_id='g1',
            created_at=now,
        ),
        EntityNode(
            uuid='node-2',
            name='rejected',
            name_embedding=[0.0, 1.0],
            group_id='g1',
            created_at=now,
        ),
    ]

    with pytest.raises(NodeGroupMismatchError):
        await add_nodes_and_edges_bulk_tx(tx, [], [], nodes, [], MagicMock(), driver)

    assert tx.run.await_count == 2
    driver.save_to_aoss.assert_not_called()
    driver.save_vector_to_aoss_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_bulk_edge_write_rejects_partial_neptune_result():
    driver = MagicMock()
    driver.provider = GraphProvider.NEPTUNE
    driver.graph_operations_interface = None
    driver.save_vector_to_aoss_async = AsyncMock(return_value=1)
    driver.delete_from_aoss_async = AsyncMock()
    tx = MagicMock()
    tx.run = AsyncMock(
        side_effect=[
            ([], None, None),
            ([], None, None),
            ([], None, None),
            ([{'uuid': 'edge-1', 'projection_version': 7}], None, None),
            ([], None, None),
        ]
    )
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    edges = [
        EntityEdge(
            uuid='edge-1',
            name='RELATES_TO',
            fact='persisted',
            fact_embedding=[1.0, 0.0],
            group_id='g1',
            source_node_uuid='source',
            target_node_uuid='target',
            created_at=now,
        ),
        EntityEdge(
            uuid='edge-2',
            name='RELATES_TO',
            fact='missing endpoint',
            fact_embedding=[0.0, 1.0],
            group_id='g1',
            source_node_uuid='missing',
            target_node_uuid='target',
            created_at=now,
        ),
    ]

    with pytest.raises(NodeGroupMismatchError):
        await add_nodes_and_edges_bulk_tx(tx, [], [], [], edges, MagicMock(), driver)

    driver.save_to_aoss.assert_not_called()
    driver.save_vector_to_aoss_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_bulk_write_with_projection_disabled_keeps_generation_pending():
    driver = MagicMock()
    driver.provider = GraphProvider.NEPTUNE
    driver.graph_operations_interface = None
    driver.vector_projection_enabled = False
    driver.save_vector_to_aoss_async = AsyncMock()
    tx = MagicMock()
    tx.run = AsyncMock(
        side_effect=[
            ([], None, None),
            ([{'uuid': 'node-1', 'projection_version': 7}], None, None),
            ([], None, None),
            ([], None, None),
        ]
    )
    node = EntityNode(
        uuid='node-1',
        name='node',
        name_embedding=[1.0, 0.0],
        group_id='g1',
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    await add_nodes_and_edges_bulk_tx(tx, [], [], [node], [], MagicMock(), driver)

    assert tx.run.await_count == 4
    driver.save_to_aoss.assert_called_once_with(
        'node_name_and_summary',
        [{'uuid': 'node-1', 'name': 'node', 'summary': '', 'group_id': 'g1'}],
    )
    driver.save_vector_to_aoss_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_disabled_legacy_neptune_bulk_operations_keep_fulltext_indexing():
    driver = MagicMock()
    driver.provider = GraphProvider.NEPTUNE
    driver.vector_projection_enabled = False
    driver.graph_operations_interface = MagicMock()
    driver.graph_operations_interface.episodic_node_save_bulk = AsyncMock()
    driver.graph_operations_interface.node_save_bulk = AsyncMock()
    driver.graph_operations_interface.episodic_edge_save_bulk = AsyncMock()
    driver.graph_operations_interface.edge_save_bulk = AsyncMock()
    driver.save_vector_to_aoss_async = AsyncMock()
    tx = MagicMock()
    node = EntityNode(
        uuid='node-1',
        name='node',
        name_embedding=[1.0, 0.0],
        group_id='g1',
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    await add_nodes_and_edges_bulk_tx(tx, [], [], [node], [], MagicMock(), driver)

    driver.save_to_aoss.assert_called_once_with(
        'node_name_and_summary',
        [{'uuid': 'node-1', 'name': 'node', 'summary': '', 'group_id': 'g1'}],
    )
    driver.save_vector_to_aoss_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_bulk_ingest_rejects_projection_reserved_attributes_before_graph_write():
    driver = MagicMock()
    driver.provider = GraphProvider.NEPTUNE
    driver.graph_operations_interface = None
    tx = MagicMock()
    tx.run = AsyncMock()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    node = EntityNode(
        uuid='node-1',
        name='node',
        name_embedding=[1.0, 0.0],
        group_id='g1',
        created_at=now,
        attributes={'_graphiti_vector_delete_pending': True},
    )

    with pytest.raises(ValueError, match='_graphiti_vector_delete_pending'):
        await add_nodes_and_edges_bulk_tx(tx, [], [], [node], [], MagicMock(), driver)

    tx.run.assert_not_awaited()


class TestNeptuneClientTimeouts:
    """A boto3 client with no explicit Config falls back to 60s connect/read
    timeouts and minimal retries, which turns a slow-but-alive Neptune query
    into a dropped connection instead of a clean, retryable timeout."""

    def test_database_client_sets_boto_config(self):
        with patch('graphiti_core.driver.neptune_driver.boto3.Session') as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session

            NeptuneDatabaseClient('neptune.example.com', 8182)

            mock_session.client.assert_called_once_with(
                'neptunedata',
                endpoint_url='https://neptune.example.com:8182',
                config=NEPTUNE_BOTO_CONFIG,
            )

    def test_analytics_client_sets_boto_config(self):
        with patch('graphiti_core.driver.neptune_driver.boto3.Session') as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session

            NeptuneAnalyticsClient('g-example')

            mock_session.client.assert_called_once_with('neptune-graph', config=NEPTUNE_BOTO_CONFIG)

    def test_config_read_timeout_exceeds_default_neptune_query_timeout(self):
        # neptune_query_timeout cluster parameter defaults to 120s; our client
        # read_timeout must be longer so Neptune's own timeout fires first.
        assert NEPTUNE_BOTO_CONFIG.read_timeout > 120
        assert NEPTUNE_BOTO_CONFIG.retries['mode'] == 'standard'
        assert NEPTUNE_BOTO_CONFIG.retries['max_attempts'] >= 1


def test_graphiti_rejects_neptune_embedder_dimension_mismatch():
    driver = MagicMock()
    driver.provider = GraphProvider.NEPTUNE
    driver.embedding_dim = 2
    embedder = MagicMock()
    embedder.config.embedding_dim = 3

    with pytest.raises(
        ValueError,
        match=r'NeptuneDriver embedding_dim \(2\).*embedder dimension \(3\)',
    ):
        Graphiti(
            graph_driver=driver,
            llm_client=MagicMock(),
            embedder=embedder,
        )


class TestCosineSimilarityFromKnnScore:
    """The vector indexes use the faiss engine's cosinesimil space, where
    OpenSearch scores as (1 + cosine_similarity) / 2."""

    @pytest.mark.parametrize(
        ('cosine', 'expected_score'),
        [(1.0, 1.0), (0.0, 0.5), (-1.0, 0.0)],
    )
    def test_round_trips_reference_cosine_values(self, cosine, expected_score):
        score = (1 + cosine) / 2
        assert score == pytest.approx(expected_score)
        assert cosine_similarity_from_knn_score(score) == pytest.approx(cosine)

    def test_zero_score_does_not_raise(self):
        assert cosine_similarity_from_knn_score(0.0) == -1.0


class TestNeptuneKnnQuery:
    def _driver(self) -> NeptuneDriver:
        driver = object.__new__(NeptuneDriver)
        driver.embedding_dim = 2
        driver.vector_search_enabled = True
        driver._execute_aoss_search = AsyncMock(  # type: ignore[method-assign]
            return_value={'hits': {'hits': []}}
        )
        return driver

    @pytest.mark.asyncio
    async def test_zero_limit_does_not_query_aoss(self):
        driver = self._driver()

        result = await driver.run_aoss_knn_query('node_name_embedding', [1.0, 0.0], 0, 0.5)

        assert result == []
        driver._execute_aoss_search.assert_not_awaited()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('group_ids', 'uuids'),
        [([], None), (None, [])],
    )
    async def test_empty_filter_does_not_query_aoss(self, group_ids, uuids):
        driver = self._driver()

        result = await driver.run_aoss_knn_query(
            'edge_fact_embedding',
            [1.0, 0.0],
            10,
            0.5,
            group_ids=group_ids,
            uuids=uuids,
        )

        assert result == []
        driver._execute_aoss_search.assert_not_awaited()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_group_and_uuid_filters_are_pushed_into_knn_query(self):
        driver = self._driver()
        driver._execute_aoss_search.return_value = {  # type: ignore[attr-defined]
            'hits': {'hits': [{'_score': 0.75, '_source': {'uuid': 'edge-1'}}]}
        }

        result = await driver.run_aoss_knn_query(
            'edge_fact_embedding',
            [1.0, 0.0],
            5,
            0.4,
            group_ids=['group-1'],
            uuids=['edge-1', 'edge-2'],
        )

        assert result == [{'id': 'edge-1', 'score': pytest.approx(0.5)}]
        driver._execute_aoss_search.assert_awaited_once_with(  # type: ignore[attr-defined]
            'edge_fact_embedding',
            {
                'size': 5,
                '_source': ['uuid'],
                'query': {
                    'knn': {
                        'embedding': {
                            'vector': [1.0, 0.0],
                            'k': 5,
                            'filter': {
                                'bool': {
                                    'filter': [
                                        {'terms': {'group_id': ['group-1']}},
                                        {'terms': {'uuid': ['edge-1', 'edge-2']}},
                                    ]
                                }
                            },
                        }
                    }
                },
            },
            vector=True,
        )

    @pytest.mark.asyncio
    async def test_query_dimension_must_match_vector_index(self):
        driver = self._driver()

        with pytest.raises(ValueError, match='dimension 3; expected 2'):
            await driver.run_aoss_knn_query('node_name_embedding', [1.0, 0.0, 0.0], 10, 0.5)

        driver._execute_aoss_search.assert_not_awaited()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_disabled_vector_search_cannot_query_aoss(self):
        driver = self._driver()
        driver.vector_search_enabled = False

        with pytest.raises(RuntimeError, match='vector search is disabled'):
            await driver.run_aoss_knn_query('node_name_embedding', [1.0, 0.0], 10, 0.5)


class TestVectorIndexUnsupportedError:
    """A SEARCH type AOSS collection rejects knn_vector mappings. Index creation
    must surface one specific, actionable error instead of silently degrading."""

    def test_vector_index_rejection_raises_named_error(self):
        driver = object.__new__(NeptuneDriver)
        driver._aoss_host = 'aoss.example.com'
        driver.aoss_client = MagicMock()
        driver.aoss_client.indices.exists.return_value = False
        driver.aoss_client.indices.create.side_effect = Exception(
            'illegal_argument_exception: mapper [embedding] of type [knn_vector] is not supported'
        )

        with pytest.raises(VectorIndexUnsupportedError) as exc_info:
            driver._create_aoss_index(
                {
                    'index_name': 'node_name_embedding',
                    'body': {
                        'mappings': {
                            'properties': {'embedding': {'type': 'knn_vector', 'dimension': 4}}
                        }
                    },
                }
            )

        message = str(exc_info.value)
        assert 'aoss.example.com' in message
        assert 'node_name_embedding' in message
        assert 'VECTORSEARCH' in message

    def test_text_index_creation_failure_is_not_reclassified(self):
        driver = object.__new__(NeptuneDriver)
        driver._aoss_host = 'aoss.example.com'
        driver.aoss_client = MagicMock()
        driver.aoss_client.indices.exists.return_value = False
        driver.aoss_client.indices.create.side_effect = Exception('boom')

        with pytest.raises(Exception, match='boom'):
            driver._create_aoss_index(
                {
                    'index_name': 'node_name_and_summary',
                    'body': {'mappings': {'properties': {'name': {'type': 'text'}}}},
                }
            )

    def test_vector_index_authorization_failure_is_not_reclassified(self):
        driver = object.__new__(NeptuneDriver)
        driver._aoss_host = 'aoss.example.com'
        driver.aoss_client = MagicMock()
        driver.aoss_client.indices.exists.return_value = False
        driver.aoss_client.indices.create.side_effect = PermissionError(
            'authorization_exception while creating knn_vector index'
        )
        vector_index = neptune_driver_module._vector_aoss_indices(4)[0]

        with pytest.raises(PermissionError, match='authorization_exception'):
            driver._create_aoss_index(vector_index)

    def test_existing_incompatible_vector_index_fails_validation(self):
        driver = object.__new__(NeptuneDriver)
        driver._aoss_host = 'aoss.example.com'
        driver.aoss_client = MagicMock()
        driver.aoss_client.indices.exists.return_value = True
        driver.aoss_client.indices.get_mapping.return_value = {
            'node_name_embedding': {
                'mappings': {
                    'properties': {
                        'uuid': {'type': 'keyword'},
                        'group_id': {'type': 'keyword'},
                        'embedding': {
                            'type': 'knn_vector',
                            'dimension': 8,
                            'method': {
                                'name': 'hnsw',
                                'engine': 'faiss',
                                'space_type': 'cosinesimil',
                            },
                        },
                    }
                }
            }
        }
        driver.aoss_client.indices.get_settings.return_value = {
            'node_name_embedding': {'settings': {'index': {'knn': 'true'}}}
        }

        with pytest.raises(VectorIndexConfigurationError, match=r'dimension=8.*expected 4'):
            driver._create_aoss_index(neptune_driver_module._vector_aoss_indices(4)[0])

        driver.aoss_client.indices.create.assert_not_called()

    def test_existing_index_is_not_recreated(self):
        driver = object.__new__(NeptuneDriver)
        driver._aoss_host = 'aoss.example.com'
        driver.aoss_client = MagicMock()
        driver.aoss_client.indices.exists.return_value = True

        driver._create_aoss_index(
            {'index_name': 'node_name_embedding', 'body': {'mappings': {'properties': {}}}}
        )

        driver.aoss_client.indices.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_vector_aoss_indices_only_touches_vector_indices(self):
        driver = object.__new__(NeptuneDriver)
        driver._aoss_host = 'aoss.example.com'
        driver.aoss_client = MagicMock()
        driver.aoss_client.indices.exists.return_value = False

        await driver.create_vector_aoss_indices()

        created_names = {
            call.kwargs['index'] for call in driver.aoss_client.indices.create.call_args_list
        }
        assert created_names == {'node_name_embedding', 'edge_fact_embedding'}

    @pytest.mark.asyncio
    async def test_disabled_startup_creates_only_text_indices(self):
        driver = object.__new__(NeptuneDriver)
        driver.vector_search_enabled = False
        driver.vector_projection_enabled = False
        driver._create_aoss_index = MagicMock()  # type: ignore[method-assign]

        with patch.object(neptune_driver_module.asyncio, 'sleep', new=AsyncMock()) as sleep:
            await driver.create_aoss_indices()

        created_names = {
            call.args[0]['index_name']
            for call in driver._create_aoss_index.call_args_list  # type: ignore[attr-defined]
        }
        assert created_names == {
            index['index_name'] for index in neptune_driver_module.text_aoss_indices
        }
        sleep.assert_awaited_once_with(60)


class TestVectorProjectionDeletion:
    def test_uuid_delete_writes_strict_versioned_tombstones(self):
        driver = object.__new__(NeptuneDriver)
        driver.aoss_client = MagicMock()

        with patch.object(neptune_driver_module.helpers, 'bulk', return_value=(2, [])) as bulk:
            deleted = driver.delete_from_aoss(
                'node_name_embedding',
                uuids=['node-1', 'node-2'],
                versions={'node-1': 4, 'node-2': 9},
            )

        assert deleted == 2
        assert bulk.call_args.args[1] == [
            {
                '_op_type': 'index',
                '_index': 'node_name_embedding',
                '_id': 'node-1',
                '_version': neptune_driver_module.vector_aoss_external_version(4),
                '_version_type': 'external',
                'uuid': 'node-1',
                'projection_deleted': True,
            },
            {
                '_op_type': 'index',
                '_index': 'node_name_embedding',
                '_id': 'node-2',
                '_version': neptune_driver_module.vector_aoss_external_version(9),
                '_version_type': 'external',
                'uuid': 'node-2',
                'projection_deleted': True,
            },
        ]

    def test_older_save_version_conflict_is_a_superseded_success(self):
        driver = object.__new__(NeptuneDriver)
        driver.vector_projection_enabled = True
        driver._vector_aoss_indices = neptune_driver_module._vector_aoss_indices(2)
        driver._aoss_indices = driver._vector_aoss_indices
        driver.aoss_client = MagicMock()
        conflict = {
            'index': {
                '_id': 'node-1',
                'status': 409,
                'error': {'type': 'version_conflict_engine_exception'},
            }
        }

        with patch.object(
            neptune_driver_module.helpers, 'bulk', return_value=(0, [conflict])
        ) as bulk:
            assert (
                driver.save_vector_to_aoss(
                    'node_name_embedding',
                    [
                        {
                            'uuid': 'node-1',
                            'group_id': 'g1',
                            'embedding': [1.0, 0.0],
                            '_version': 3,
                        }
                    ],
                )
                == 1
            )

        assert bulk.call_args.args[1][0]['_version_type'] == 'external'

    def test_older_tombstone_version_conflict_is_a_superseded_success(self):
        driver = object.__new__(NeptuneDriver)
        driver.aoss_client = MagicMock()
        conflict = {
            'index': {
                '_id': 'node-1',
                'status': 409,
                'error': {'type': 'version_conflict_engine_exception'},
            }
        }

        with patch.object(neptune_driver_module.helpers, 'bulk', return_value=(0, [conflict])):
            assert (
                driver.delete_from_aoss(
                    'node_name_embedding',
                    uuids=['node-1'],
                    versions={'node-1': 3},
                )
                == 1
            )

    def test_unrelated_409_is_not_treated_as_a_version_conflict(self):
        driver = object.__new__(NeptuneDriver)
        driver.vector_projection_enabled = True
        driver._vector_aoss_indices = neptune_driver_module._vector_aoss_indices(2)
        driver._aoss_indices = driver._vector_aoss_indices
        driver.aoss_client = MagicMock()
        unrelated = {
            'index': {
                '_id': 'node-1',
                'status': 409,
                'error': {'type': 'resource_already_exists_exception'},
            }
        }

        with (
            patch.object(
                neptune_driver_module.helpers,
                'bulk',
                return_value=(0, [unrelated]),
            ),
            pytest.raises(AossProjectionError, match='indexed 0/1'),
        ):
            driver.save_vector_to_aoss(
                'node_name_embedding',
                [
                    {
                        'uuid': 'node-1',
                        'group_id': 'g1',
                        'embedding': [1.0, 0.0],
                        '_version': 3,
                    }
                ],
            )

    @pytest.mark.asyncio
    async def test_disabled_reset_does_not_delete_vector_indices(self):
        driver = object.__new__(NeptuneDriver)
        driver.vector_projection_enabled = False
        driver._aoss_indices = (
            neptune_driver_module.text_aoss_indices + neptune_driver_module._vector_aoss_indices(2)
        )
        driver._vector_aoss_indices = neptune_driver_module._vector_aoss_indices(2)
        driver.aoss_client = MagicMock()
        driver.vector_aoss_client = MagicMock()
        driver.aoss_client.indices.exists.return_value = True
        driver.vector_aoss_client.indices.exists.return_value = True

        await driver.delete_aoss_indices()

        deleted_text_indices = {
            call.kwargs['index'] for call in driver.aoss_client.indices.delete.call_args_list
        }
        assert deleted_text_indices == {
            index['index_name'] for index in neptune_driver_module.text_aoss_indices
        }
        driver.vector_aoss_client.indices.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_vector_reset_uses_vector_collection(self):
        driver = object.__new__(NeptuneDriver)
        driver._vector_aoss_indices = neptune_driver_module._vector_aoss_indices(2)
        driver.aoss_client = MagicMock()
        driver.vector_aoss_client = MagicMock()
        driver.vector_aoss_client.indices.exists.return_value = True

        await driver.delete_vector_aoss_indices()

        deleted_vector_indices = {
            call.kwargs['index'] for call in driver.vector_aoss_client.indices.delete.call_args_list
        }
        assert deleted_vector_indices == {'node_name_embedding', 'edge_fact_embedding'}
        driver.aoss_client.indices.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_vector_create_search_save_and_delete_use_separate_client(self):
        driver = object.__new__(NeptuneDriver)
        driver.embedding_dim = 2
        driver.vector_search_enabled = True
        driver.vector_projection_enabled = True
        driver._vector_aoss_indices = neptune_driver_module._vector_aoss_indices(2)
        driver._aoss_indices = neptune_driver_module.text_aoss_indices + driver._vector_aoss_indices
        driver._aoss_host = 'text.example.com'
        driver._vector_aoss_host = 'vector.example.com'
        driver.aoss_client = MagicMock()
        driver.vector_aoss_client = MagicMock()
        driver.vector_aoss_client.indices.exists.return_value = False
        driver.vector_aoss_client.search.return_value = {
            'hits': {'hits': [{'_score': 1.0, '_source': {'uuid': 'node-1'}}]}
        }

        await driver.create_vector_aoss_indices()
        driver.vector_aoss_client.indices.exists.return_value = True
        result = await driver.run_aoss_knn_query('node_name_embedding', [1.0, 0.0], 1, 0.5)
        with patch.object(
            neptune_driver_module.helpers,
            'bulk',
            side_effect=[(1, []), (1, [])],
        ) as bulk:
            assert (
                driver.save_vector_to_aoss(
                    'node_name_embedding',
                    [
                        {
                            'uuid': 'node-1',
                            'group_id': 'g1',
                            'embedding': [1.0, 0.0],
                            '_version': 1,
                        }
                    ],
                )
                == 1
            )
            assert (
                driver.delete_from_aoss(
                    'node_name_embedding',
                    uuids=['node-1'],
                    versions={'node-1': 2},
                )
                == 1
            )

        assert result == [{'id': 'node-1', 'score': 1.0}]
        assert {
            call.kwargs['index'] for call in driver.vector_aoss_client.indices.create.call_args_list
        } == {'node_name_embedding', 'edge_fact_embedding'}
        driver.vector_aoss_client.search.assert_called_once()
        assert [call.args[0] for call in bulk.call_args_list] == [
            driver.vector_aoss_client,
            driver.vector_aoss_client,
        ]
        driver.aoss_client.search.assert_not_called()
        driver.aoss_client.indices.create.assert_not_called()
