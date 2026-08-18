"""Tests for Neptune driver async behavior."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.neptune_driver import (
    NEPTUNE_BOTO_CONFIG,
    NeptuneAnalyticsClient,
    NeptuneDatabaseClient,
    NeptuneDriver,
)
from graphiti_core.helpers import EPISODE_AOSS_WRITE_VERSION
from graphiti_core.nodes import EpisodeType, EpisodicNode
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
