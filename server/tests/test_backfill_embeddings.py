"""Tests for graph_service.backfill_embeddings.

Batching/cursor logic is covered with a fake Neptune query source (real
Neptune is not reachable from a test environment). Bulk-indexing logic is
covered against a real OpenSearch instance the same way
tests/driver/test_neptune_knn_similarity_int.py is: skipped if OpenSearch is
not reachable on OPENSEARCH_TEST_PORT (default 9201).
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from graph_service import backfill_embeddings
from graph_service.backfill_embeddings import BackfillError

OPENSEARCH_TEST_PORT = int(os.getenv('OPENSEARCH_TEST_PORT', '9201'))


class FakeNeptuneQuerySource:
    """Simulates the ORDER BY uuid DESC / `AND uuid < $cursor` / LIMIT pagination
    the real Neptune queries use, over an in-memory record list."""

    def __init__(self, records: list[dict[str, Any]]):
        # Records must already be in descending uuid order, matching the real
        # `ORDER BY n.uuid DESC` / `ORDER BY e.uuid DESC` queries.
        self.records = records
        self.calls: list[dict[str, Any]] = []
        self.queries: list[str] = []
        self.completed: list[list[dict[str, Any]]] = []

    async def execute_query(self, query: str, **kwargs: Any):
        self.queries.append(query)
        self.calls.append(kwargs)
        if 'completed' in kwargs:
            self.completed.append(kwargs['completed'])
            return [], None, None
        cursor = kwargs.get('cursor')
        batch_size = kwargs['batch_size']
        page = self.records if cursor is None else [r for r in self.records if r['uuid'] < cursor]
        return page[:batch_size], None, None


class FakeAossSink:
    def __init__(self, success_count: int | None = None, embedding_dim: int = 2):
        """success_count=None means "report full success"."""
        self.success_count = success_count
        self.embedding_dim = embedding_dim
        self.write_calls: list[tuple[str, list[dict[str, Any]]]] = []

    def save_vector_to_aoss(self, name: str, data: list[dict[str, Any]]) -> int:
        self.write_calls.append((name, data))
        return len(data) if self.success_count is None else self.success_count

    async def save_vector_to_aoss_async(self, name: str, data: list[dict[str, Any]]) -> int:
        return self.save_vector_to_aoss(name, data)

    async def delete_from_aoss_async(
        self,
        name: str,
        *,
        uuids: list[str] | None = None,
        group_ids: list[str] | None = None,
        versions: dict[str, int] | None = None,
    ) -> int:
        del name, versions
        return len(uuids or group_ids or [])


class FakeDriver(FakeNeptuneQuerySource, FakeAossSink):
    def __init__(self, records: list[dict[str, Any]], success_count: int | None = None):
        FakeNeptuneQuerySource.__init__(self, records)
        FakeAossSink.__init__(self, success_count)


def _records(uuids: list[str], group_id: str = 'g1') -> list[dict[str, Any]]:
    return [
        {
            'uuid': u,
            'group_id': group_id,
            'embedding': [0.1, 0.2],
            'projection_version': 1,
        }
        for u in uuids
    ]


class TestIterBatches:
    @pytest.mark.asyncio
    async def test_paginates_by_cursor_and_preserves_order(self):
        # Already-descending uuids, matching ORDER BY uuid DESC.
        records = _records(['u9', 'u8', 'u7', 'u6', 'u5', 'u4', 'u3', 'u2', 'u1', 'u0'])
        source = FakeNeptuneQuerySource(records)

        batches = [batch async for batch in backfill_embeddings.iter_node_batches(source, 'g1', 4)]

        assert [len(b) for b in batches] == [4, 4, 2]
        assert [r['uuid'] for batch in batches for r in batch] == [r['uuid'] for r in records]
        # Second call's cursor is the last uuid of the first batch.
        assert source.calls[1]['cursor'] == batches[0][-1]['uuid']
        assert source.calls[0].get('cursor') is None

    @pytest.mark.asyncio
    async def test_makes_one_terminating_call_when_last_page_is_exactly_full(self):
        records = _records(['u3', 'u2', 'u1', 'u0'])
        source = FakeNeptuneQuerySource(records)

        batches = [batch async for batch in backfill_embeddings.iter_node_batches(source, 'g1', 4)]

        assert [len(b) for b in batches] == [4]
        # One call for the full page, one more that comes back empty and stops.
        assert len(source.calls) == 2

    @pytest.mark.asyncio
    async def test_no_records_yields_no_batches(self):
        source = FakeNeptuneQuerySource([])

        batches = [batch async for batch in backfill_embeddings.iter_node_batches(source, 'g1', 4)]

        assert batches == []
        assert len(source.calls) == 1

    @pytest.mark.asyncio
    async def test_edge_batches_use_edge_uuid_cursor(self):
        records = _records(['e2', 'e1', 'e0'])
        source = FakeNeptuneQuerySource(records)

        batches = [batch async for batch in backfill_embeddings.iter_edge_batches(source, 'g1', 2)]

        assert [len(b) for b in batches] == [2, 1]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('iterator_name', 'non_null_clause', 'non_empty_clause'),
        [
            (
                'iter_node_batches',
                'n.name_embedding IS NOT NULL',
                "n.name_embedding <> ''",
            ),
            (
                'iter_edge_batches',
                'e.fact_embedding IS NOT NULL',
                "e.fact_embedding <> ''",
            ),
        ],
    )
    async def test_queries_exclude_null_and_empty_embeddings(
        self, iterator_name, non_null_clause, non_empty_clause
    ):
        source = FakeNeptuneQuerySource([])
        iterator = getattr(backfill_embeddings, iterator_name)

        batches = [batch async for batch in iterator(source, 'g1', 4)]

        assert batches == []
        assert non_null_clause in source.queries[0]
        assert non_empty_clause in source.queries[0]
        assert '_graphiti_vector_delete_pending' in source.queries[0]

    @pytest.mark.asyncio
    async def test_all_groups_query_omits_group_filter_and_parameter(self):
        source = FakeNeptuneQuerySource([])

        batches = [batch async for batch in backfill_embeddings.iter_node_batches(source, None, 4)]

        assert batches == []
        assert 'n.group_id = $group_id' not in source.queries[0]
        assert 'group_id' not in source.calls[0]

    @pytest.mark.asyncio
    async def test_pending_query_includes_tombstones_and_exact_generation_filter(self):
        source = FakeNeptuneQuerySource([])

        batches = [
            batch
            async for batch in backfill_embeddings.iter_node_batches(
                source,
                None,
                4,
                pending_only=True,
            )
        ]

        assert batches == []
        query = source.queries[0]
        assert 'n.name_embedding IS NOT NULL' not in query
        assert (
            'n._graphiti_vector_sync_pending = coalesce(n._graphiti_projection_version, 0)'
        ) in query
        assert 'CASE WHEN n.name_embedding IS NULL' in query

    @pytest.mark.asyncio
    @pytest.mark.parametrize('batch_size', [0, -1])
    async def test_rejects_non_positive_batch_size_before_query(self, batch_size):
        source = FakeNeptuneQuerySource(_records(['u1']))

        with pytest.raises(ValueError, match='batch_size must be a positive integer'):
            _ = [
                batch
                async for batch in backfill_embeddings.iter_node_batches(source, 'g1', batch_size)
            ]

        assert source.calls == []


class TestIndexBatch:
    def test_empty_batch_does_not_call_save_to_aoss(self):
        sink = FakeAossSink()

        result = backfill_embeddings.index_batch(sink, 'node_name_embedding', [])

        assert result == 0
        assert sink.write_calls == []

    def test_success_indexes_expected_documents(self):
        sink = FakeAossSink()
        batch = _records(['n1', 'n2'])

        result = backfill_embeddings.index_batch(sink, 'node_name_embedding', batch)

        assert result == 2
        name, docs = sink.write_calls[0]
        assert name == 'node_name_embedding'
        assert docs == [
            {'uuid': 'n1', 'group_id': 'g1', 'embedding': [0.1, 0.2], '_version': 1},
            {'uuid': 'n2', 'group_id': 'g1', 'embedding': [0.1, 0.2], '_version': 1},
        ]

    def test_partial_failure_raises_backfill_error(self):
        sink = FakeAossSink(success_count=1)
        batch = _records(['n1', 'n2'])

        with pytest.raises(BackfillError, match='node_name_embedding'):
            backfill_embeddings.index_batch(sink, 'node_name_embedding', batch)

    def test_missing_embedding_becomes_versioned_tombstone(self):
        sink = FakeAossSink()
        batch = [
            {
                'uuid': 'n1',
                'group_id': 'g1',
                'embedding': None,
                'projection_version': 7,
            }
        ]

        result = backfill_embeddings.index_batch(sink, 'node_name_embedding', batch)

        assert result == 1
        assert sink.write_calls == [
            ('node_name_embedding', [{'uuid': 'n1', 'group_id': 'g1', '_version': 7}])
        ]

    @pytest.mark.parametrize(
        'embedding',
        ['0.1,0.2', [0.1, None], [0.1, float('nan')], [0.1, float('inf')]],
    )
    def test_malformed_embedding_rejected_before_aoss_write(self, embedding):
        sink = FakeAossSink(embedding_dim=2)
        batch = [
            _records(['valid'])[0],
            {
                'uuid': 'bad',
                'group_id': 'g1',
                'embedding': embedding,
                'projection_version': 1,
            },
        ]

        with pytest.raises(BackfillError, match='bad.*malformed embedding'):
            backfill_embeddings.index_batch(sink, 'node_name_embedding', batch)

        assert sink.write_calls == []

    @pytest.mark.parametrize('embedding', [[0.1], [0.1, 0.2, 0.3]])
    def test_wrong_dimension_rejected_before_aoss_write(self, embedding):
        sink = FakeAossSink(embedding_dim=2)
        batch = [
            _records(['valid'])[0],
            {
                'uuid': 'bad',
                'group_id': 'g1',
                'embedding': embedding,
                'projection_version': 1,
            },
        ]

        with pytest.raises(BackfillError, match=r'bad.*dimension.*expected 2'):
            backfill_embeddings.index_batch(sink, 'edge_fact_embedding', batch)

        assert sink.write_calls == []


class TestRunBackfill:
    @pytest.mark.asyncio
    async def test_aggregates_counts_across_all_batches(self, monkeypatch):
        driver = FakeDriver(records=[])
        # run_backfill calls iter_node_batches/iter_edge_batches against the same
        # driver; give each its own record set via distinct fakes wired through.
        node_records = _records(['n2', 'n1', 'n0'])
        edge_records = _records(['e1', 'e0'])

        call_log: list[str] = []

        async def fake_iter_node_batches(d, group_id, batch_size, pending_only=False):
            call_log.append('nodes')
            for i in range(0, len(node_records), batch_size):
                yield node_records[i : i + batch_size]

        async def fake_iter_edge_batches(d, group_id, batch_size, pending_only=False):
            call_log.append('edges')
            for i in range(0, len(edge_records), batch_size):
                yield edge_records[i : i + batch_size]

        monkeypatch.setattr(backfill_embeddings, 'iter_node_batches', fake_iter_node_batches)
        monkeypatch.setattr(backfill_embeddings, 'iter_edge_batches', fake_iter_edge_batches)

        stats = await backfill_embeddings.run_backfill(driver, 'g1', batch_size=2)

        assert stats.nodes_indexed == 3
        assert stats.edges_indexed == 2
        assert call_log == ['nodes', 'edges']
        indexed_names = {name for name, _ in driver.write_calls}
        assert indexed_names == {'node_name_embedding', 'edge_fact_embedding'}
        assert driver.completed == [
            [
                {'uuid': 'n2', 'projection_version': 1},
                {'uuid': 'n1', 'projection_version': 1},
            ],
            [{'uuid': 'n0', 'projection_version': 1}],
            [
                {'uuid': 'e1', 'projection_version': 1},
                {'uuid': 'e0', 'projection_version': 1},
            ],
        ]

    @pytest.mark.asyncio
    async def test_pending_only_delegates_to_shared_reconciler(self, monkeypatch):
        driver = FakeDriver(records=[])
        reconciliation = SimpleNamespace(
            nodes_saved=2,
            edges_saved=3,
            nodes_deleted=4,
            edges_deleted=5,
            failures=6,
        )
        reconcile_pending = AsyncMock(return_value=reconciliation)
        monkeypatch.setattr(
            backfill_embeddings,
            'reconcile_pending_projections',
            reconcile_pending,
        )

        stats = await backfill_embeddings.run_backfill(
            driver,
            'g1',
            batch_size=17,
            pending_only=True,
        )

        reconcile_pending.assert_awaited_once_with(driver, group_id='g1', batch_size=17)
        assert stats.nodes_indexed == 2
        assert stats.edges_indexed == 3
        assert stats.nodes_deleted == 4
        assert stats.edges_deleted == 5
        assert stats.failures == 6
        assert driver.calls == []
        assert driver.write_calls == []

    @pytest.mark.asyncio
    async def test_stops_and_raises_on_first_bulk_failure(self, monkeypatch):
        driver = FakeDriver(records=[], success_count=0)

        async def fake_iter_node_batches(d, group_id, batch_size, pending_only=False):
            yield _records(['n0'])

        async def fake_iter_edge_batches(d, group_id, batch_size, pending_only=False):
            pytest.fail('edges should not be processed after a node batch failure')
            yield  # pragma: no cover

        monkeypatch.setattr(backfill_embeddings, 'iter_node_batches', fake_iter_node_batches)
        monkeypatch.setattr(backfill_embeddings, 'iter_edge_batches', fake_iter_edge_batches)

        with pytest.raises(BackfillError):
            await backfill_embeddings.run_backfill(driver, 'g1', batch_size=500)

    @pytest.mark.asyncio
    async def test_rejects_non_positive_batch_size_before_processing(self, monkeypatch):
        driver = FakeDriver(records=[])

        async def fail_if_iterated(*args, **kwargs):
            pytest.fail('iterators must not run for an invalid batch size')
            yield  # pragma: no cover

        monkeypatch.setattr(backfill_embeddings, 'iter_node_batches', fail_if_iterated)

        with pytest.raises(ValueError, match='batch_size must be a positive integer'):
            await backfill_embeddings.run_backfill(driver, 'g1', batch_size=0)

        assert driver.write_calls == []


def test_parse_args_rejects_non_positive_batch_size(capsys):
    with pytest.raises(SystemExit) as exc_info:
        backfill_embeddings._parse_args(
            [
                '--group-id',
                'g1',
                '--batch-size',
                '0',
                '--acknowledge-ingestion-and-deletion-quiesced',
            ]
        )

    assert exc_info.value.code == 2
    assert 'must be a positive integer' in capsys.readouterr().err


def test_parse_args_requires_explicit_quiescence_acknowledgement(capsys):
    with pytest.raises(SystemExit) as exc_info:
        backfill_embeddings._parse_args(['--group-id', 'g1'])

    assert exc_info.value.code == 2
    assert '--acknowledge-ingestion-and-deletion-quiesced' in capsys.readouterr().err


def test_parse_args_requires_exactly_one_scope(capsys):
    acknowledgement = ['--acknowledge-ingestion-and-deletion-quiesced']
    with pytest.raises(SystemExit) as missing_scope:
        backfill_embeddings._parse_args(acknowledgement)
    with pytest.raises(SystemExit) as both_scopes:
        backfill_embeddings._parse_args(['--group-id', 'g1', '--all-groups', *acknowledgement])

    assert missing_scope.value.code == 2
    assert both_scopes.value.code == 2
    assert 'not allowed with argument' in capsys.readouterr().err


def test_parse_args_allows_index_reset_only_for_all_groups(capsys):
    with pytest.raises(SystemExit) as exc_info:
        backfill_embeddings._parse_args(
            [
                '--group-id',
                'g1',
                '--reset-vector-indices',
                '--acknowledge-ingestion-and-deletion-quiesced',
            ]
        )

    assert exc_info.value.code == 2
    assert '--reset-vector-indices requires --all-groups' in capsys.readouterr().err


@pytest.mark.parametrize(
    ('args', 'message'),
    [
        (
            ['--all-groups', '--acknowledge-ingestion-and-deletion-quiesced'],
            '--all-groups exact backfill requires --reset-vector-indices',
        ),
        (
            ['--group-id', 'g1', '--acknowledge-ingestion-and-deletion-quiesced'],
            '--group-id exact repair requires --reset-group-vector-documents',
        ),
    ],
)
def test_parse_args_requires_an_exact_scope_reset(args, message, capsys):
    with pytest.raises(SystemExit) as exc_info:
        backfill_embeddings._parse_args(args)

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err


def test_parse_args_allows_pending_repair_without_quiescence_acknowledgement():
    args = backfill_embeddings._parse_args(['--all-groups', '--pending-only'])

    assert args.pending_only is True


class FakeCliDriver:
    def __init__(self) -> None:
        self.create_vector_aoss_indices = AsyncMock()
        self.delete_vector_aoss_indices = AsyncMock()
        self.purge_vector_aoss_group_documents_async = AsyncMock()
        self.close = AsyncMock()


def _cli_settings() -> SimpleNamespace:
    return SimpleNamespace(
        db_backend='neptune',
        neptune_host='neptune-db://example',
        neptune_port=8182,
        aoss_host='text.example',
        aoss_port=443,
        vector_aoss_host='vector.example',
        vector_aoss_port=443,
    )


@pytest.mark.asyncio
async def test_single_group_cli_reconciles_without_authorizing_global_reads(monkeypatch, caplog):
    from graph_service import config

    caplog.set_level('INFO')
    driver = FakeCliDriver()
    monkeypatch.setattr(config, 'get_settings', _cli_settings)
    monkeypatch.setattr(backfill_embeddings, '_build_driver', lambda **kwargs: driver)
    run_backfill = AsyncMock(
        side_effect=[
            SimpleNamespace(nodes_indexed=5, edges_indexed=3),
            SimpleNamespace(nodes_indexed=5, edges_indexed=3),
        ]
    )
    monkeypatch.setattr(backfill_embeddings, 'run_backfill', run_backfill)

    exit_code = await backfill_embeddings._main_async(
        [
            '--group-id',
            'g1',
            '--reset-group-vector-documents',
            '--batch-size',
            '10',
            '--acknowledge-ingestion-and-deletion-quiesced',
        ]
    )

    assert exit_code == 0
    driver.delete_vector_aoss_indices.assert_not_awaited()
    driver.create_vector_aoss_indices.assert_awaited_once_with(wait_for_propagation=True)
    driver.purge_vector_aoss_group_documents_async.assert_awaited_once_with('g1')
    assert run_backfill.await_count == 2
    assert run_backfill.await_args_list[0].args == (driver, 'g1', 10)
    assert run_backfill.await_args_list[1].args == (driver, 'g1', 10)
    driver.close.assert_awaited_once_with()
    assert 'Do not enable driver-wide vector reads' in caplog.text
    assert 'driver-wide vector reads may now be enabled' not in caplog.text


@pytest.mark.asyncio
async def test_all_groups_cli_resets_reconciles_then_authorizes_global_reads(monkeypatch, caplog):
    from graph_service import config

    caplog.set_level('INFO')
    driver = FakeCliDriver()
    monkeypatch.setattr(config, 'get_settings', _cli_settings)
    monkeypatch.setattr(backfill_embeddings, '_build_driver', lambda **kwargs: driver)
    completed = SimpleNamespace(nodes_indexed=5, edges_indexed=3)
    run_backfill = AsyncMock(side_effect=[completed, completed])
    monkeypatch.setattr(backfill_embeddings, 'run_backfill', run_backfill)

    exit_code = await backfill_embeddings._main_async(
        [
            '--all-groups',
            '--reset-vector-indices',
            '--batch-size',
            '10',
            '--acknowledge-ingestion-and-deletion-quiesced',
        ]
    )

    assert exit_code == 0
    driver.delete_vector_aoss_indices.assert_awaited_once_with()
    driver.create_vector_aoss_indices.assert_awaited_once_with(wait_for_propagation=True)
    driver.purge_vector_aoss_group_documents_async.assert_not_awaited()
    assert run_backfill.await_args_list[0].args == (driver, None, 10)
    assert run_backfill.await_args_list[1].args == (driver, None, 10)
    assert 'driver-wide vector reads may now be enabled' in caplog.text
    driver.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_all_groups_reset_happens_before_create_and_both_backfill_passes(monkeypatch):
    from graph_service import config

    events: list[str] = []
    driver = FakeCliDriver()
    driver.delete_vector_aoss_indices.side_effect = lambda: events.append('delete')
    driver.create_vector_aoss_indices.side_effect = lambda **_kwargs: events.append('create')
    monkeypatch.setattr(config, 'get_settings', _cli_settings)
    monkeypatch.setattr(backfill_embeddings, '_build_driver', lambda **kwargs: driver)
    completed = SimpleNamespace(nodes_indexed=1, edges_indexed=1)

    async def run_backfill(*_args, **_kwargs):
        events.append('backfill')
        return completed

    monkeypatch.setattr(backfill_embeddings, 'run_backfill', run_backfill)

    exit_code = await backfill_embeddings._main_async(
        [
            '--all-groups',
            '--reset-vector-indices',
            '--acknowledge-ingestion-and-deletion-quiesced',
        ]
    )

    assert exit_code == 0
    assert events == ['delete', 'create', 'backfill', 'backfill']


@pytest.mark.asyncio
async def test_pending_cli_runs_one_safe_reconciliation_pass(monkeypatch, caplog):
    from graph_service import config

    caplog.set_level('INFO')
    driver = FakeCliDriver()
    monkeypatch.setattr(config, 'get_settings', _cli_settings)
    monkeypatch.setattr(backfill_embeddings, '_build_driver', lambda **kwargs: driver)
    completed = SimpleNamespace(
        nodes_indexed=2,
        edges_indexed=1,
        nodes_deleted=3,
        edges_deleted=4,
        failures=0,
    )
    run_backfill = AsyncMock(return_value=completed)
    monkeypatch.setattr(backfill_embeddings, 'run_backfill', run_backfill)

    exit_code = await backfill_embeddings._main_async(['--all-groups', '--pending-only'])

    assert exit_code == 0
    driver.delete_vector_aoss_indices.assert_not_awaited()
    driver.purge_vector_aoss_group_documents_async.assert_not_awaited()
    driver.create_vector_aoss_indices.assert_awaited_once_with(wait_for_propagation=True)
    run_backfill.assert_awaited_once_with(driver, None, 500, pending_only=True)
    driver.close.assert_awaited_once_with()
    assert 'saved=2 nodes/1 edges, deleted=3 nodes/4 edges, failures=0' in caplog.text


@pytest.mark.asyncio
async def test_pending_cli_reports_failures_and_exits_nonzero(monkeypatch, caplog):
    from graph_service import config

    caplog.set_level('INFO')
    driver = FakeCliDriver()
    monkeypatch.setattr(config, 'get_settings', _cli_settings)
    monkeypatch.setattr(backfill_embeddings, '_build_driver', lambda **kwargs: driver)
    incomplete = SimpleNamespace(
        nodes_indexed=2,
        edges_indexed=1,
        nodes_deleted=3,
        edges_deleted=4,
        failures=5,
    )
    run_backfill = AsyncMock(return_value=incomplete)
    monkeypatch.setattr(backfill_embeddings, 'run_backfill', run_backfill)

    exit_code = await backfill_embeddings._main_async(['--group-id', 'g1', '--pending-only'])

    assert exit_code == 1
    run_backfill.assert_awaited_once_with(driver, 'g1', 500, pending_only=True)
    driver.close.assert_awaited_once_with()
    assert 'saved=2 nodes/1 edges, deleted=3 nodes/4 edges, failures=5' in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize('failing_pass', ['initial', 'reconciliation'])
async def test_cli_fails_closed_when_either_pass_fails(monkeypatch, caplog, failing_pass: str):
    from graph_service import config

    driver = FakeCliDriver()
    monkeypatch.setattr(config, 'get_settings', _cli_settings)
    monkeypatch.setattr(backfill_embeddings, '_build_driver', lambda **kwargs: driver)
    completed = SimpleNamespace(nodes_indexed=5, edges_indexed=3)
    side_effect = (
        [BackfillError('initial failed')]
        if failing_pass == 'initial'
        else [completed, BackfillError('reconciliation failed')]
    )
    run_backfill = AsyncMock(side_effect=side_effect)
    monkeypatch.setattr(backfill_embeddings, 'run_backfill', run_backfill)

    exit_code = await backfill_embeddings._main_async(
        [
            '--group-id',
            'g1',
            '--reset-group-vector-documents',
            '--acknowledge-ingestion-and-deletion-quiesced',
        ]
    )

    assert exit_code == 1
    assert run_backfill.await_count == (1 if failing_pass == 'initial' else 2)
    driver.close.assert_awaited_once_with()
    assert 'keep vector reads disabled' in caplog.text
    assert 'driver-wide vector reads may now be enabled' not in caplog.text


@pytest.mark.asyncio
async def test_cli_failure_log_omits_exception_details(monkeypatch, caplog):
    from graph_service import config

    class SecretBackfillFailure(RuntimeError):
        pass

    driver = FakeCliDriver()
    secret = 'backfill-transport-credential-secret-6d30'
    monkeypatch.setattr(config, 'get_settings', _cli_settings)
    monkeypatch.setattr(backfill_embeddings, '_build_driver', lambda **kwargs: driver)
    monkeypatch.setattr(
        backfill_embeddings,
        'run_backfill',
        AsyncMock(side_effect=SecretBackfillFailure(secret)),
    )
    caplog.set_level('ERROR')

    exit_code = await backfill_embeddings._main_async(
        [
            '--group-id',
            'g1',
            '--reset-group-vector-documents',
            '--acknowledge-ingestion-and-deletion-quiesced',
        ]
    )

    assert exit_code == 1
    assert 'error_type=SecretBackfillFailure' in caplog.text
    assert secret not in caplog.text


class TestIndexBatchAgainstRealOpenSearch:
    """Bulk-indexing correctness and failure detection against a real OpenSearch
    instance. Run OpenSearch locally with the same docker command documented in
    tests/driver/test_neptune_knn_similarity_int.py."""

    @pytest.fixture(scope='class')
    def opensearch_client(self):
        try:
            from opensearchpy import OpenSearch  # pyright: ignore[reportMissingImports]
        except ModuleNotFoundError:
            pytest.skip(
                'opensearch-py is not installed; run against the graphiti-core[neptune] env'
            )

        client = OpenSearch(
            hosts=[{'host': 'localhost', 'port': OPENSEARCH_TEST_PORT}],
            use_ssl=False,
            verify_certs=False,
            timeout=2,
        )
        try:
            reachable = client.ping()
        except Exception:
            reachable = False
        if not reachable:
            pytest.skip(f'OpenSearch is not reachable on localhost:{OPENSEARCH_TEST_PORT}')
        yield client

    @pytest.fixture
    def driver_and_index(self, opensearch_client):
        import uuid

        import graphiti_core.driver.neptune_driver as neptune_driver_module
        from graphiti_core.driver.neptune_driver import NeptuneDriver

        d = object.__new__(NeptuneDriver)
        d.aoss_client = opensearch_client
        d.vector_aoss_client = opensearch_client
        d.embedding_dim = 2
        index_name = f'test_backfill_node_embedding_{uuid.uuid4().hex[:8]}'
        index_body = neptune_driver_module._vector_index_body(2)
        d._vector_aoss_indices = [{'index_name': index_name, 'body': index_body}]
        d._aoss_indices = d._vector_aoss_indices
        d.vector_projection_enabled = True
        opensearch_client.indices.create(index=index_name, body=index_body)
        yield d, index_name
        opensearch_client.indices.delete(index=index_name, ignore=[404])

    def test_index_batch_writes_queryable_documents(self, driver_and_index):
        driver, index_name = driver_and_index
        batch = _records(['n1', 'n2'])

        result = backfill_embeddings.index_batch(driver, index_name, batch)

        assert result == 2
        driver.aoss_client.indices.refresh(index=index_name)
        stored = driver.aoss_client.get(index=index_name, id='n1')
        assert stored['_source']['embedding'] == [0.1, 0.2]
        assert stored['_source']['group_id'] == 'g1'

    def test_index_batch_raises_on_real_dimension_mismatch(self, driver_and_index):
        driver, index_name = driver_and_index
        # index_name's vector field is dimension 2; a 3-element vector is a real
        # OpenSearch bulk failure, not a simulated one.
        batch = [
            {
                'uuid': 'bad',
                'group_id': 'g1',
                'embedding': [0.1, 0.2, 0.3],
                'projection_version': 1,
            }
        ]

        with pytest.raises(BackfillError):
            backfill_embeddings.index_batch(driver, index_name, batch)
