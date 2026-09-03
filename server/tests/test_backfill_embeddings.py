"""Tests for graph_service.backfill_embeddings.

Batching/cursor logic is covered with a fake Neptune query source (real
Neptune is not reachable from a test environment). Bulk-indexing logic is
covered against a real OpenSearch instance the same way
tests/driver/test_neptune_knn_similarity_int.py is: skipped if OpenSearch is
not reachable on OPENSEARCH_TEST_PORT (default 9201).
"""

from __future__ import annotations

import os
from typing import Any

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

    async def execute_query(self, query: str, **kwargs: Any):
        self.calls.append(kwargs)
        cursor = kwargs.get('cursor')
        batch_size = kwargs['batch_size']
        page = self.records if cursor is None else [r for r in self.records if r['uuid'] < cursor]
        return page[:batch_size], None, None


class FakeAossSink:
    def __init__(self, success_count: int | None = None):
        """success_count=None means "report full success"."""
        self.success_count = success_count
        self.write_calls: list[tuple[str, list[dict[str, Any]]]] = []

    def save_to_aoss(self, name: str, data: list[dict[str, Any]]) -> int:
        self.write_calls.append((name, data))
        return len(data) if self.success_count is None else self.success_count


class FakeDriver(FakeNeptuneQuerySource, FakeAossSink):
    def __init__(self, records: list[dict[str, Any]], success_count: int | None = None):
        FakeNeptuneQuerySource.__init__(self, records)
        FakeAossSink.__init__(self, success_count)


def _records(uuids: list[str], group_id: str = 'g1') -> list[dict[str, Any]]:
    return [{'uuid': u, 'group_id': group_id, 'embedding': [0.1, 0.2]} for u in uuids]


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
            {'uuid': 'n1', 'group_id': 'g1', 'embedding': [0.1, 0.2]},
            {'uuid': 'n2', 'group_id': 'g1', 'embedding': [0.1, 0.2]},
        ]

    def test_partial_failure_raises_backfill_error(self):
        sink = FakeAossSink(success_count=1)
        batch = _records(['n1', 'n2'])

        with pytest.raises(BackfillError, match='node_name_embedding'):
            backfill_embeddings.index_batch(sink, 'node_name_embedding', batch)


class TestRunBackfill:
    @pytest.mark.asyncio
    async def test_aggregates_counts_across_all_batches(self, monkeypatch):
        driver = FakeDriver(records=[])
        # run_backfill calls iter_node_batches/iter_edge_batches against the same
        # driver; give each its own record set via distinct fakes wired through.
        node_records = _records(['n2', 'n1', 'n0'])
        edge_records = _records(['e1', 'e0'])

        call_log: list[str] = []

        async def fake_iter_node_batches(d, group_id, batch_size):
            call_log.append('nodes')
            for i in range(0, len(node_records), batch_size):
                yield node_records[i : i + batch_size]

        async def fake_iter_edge_batches(d, group_id, batch_size):
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

    @pytest.mark.asyncio
    async def test_stops_and_raises_on_first_bulk_failure(self, monkeypatch):
        driver = FakeDriver(records=[], success_count=0)

        async def fake_iter_node_batches(d, group_id, batch_size):
            yield _records(['n0'])

        async def fake_iter_edge_batches(d, group_id, batch_size):
            pytest.fail('edges should not be processed after a node batch failure')
            yield  # pragma: no cover

        monkeypatch.setattr(backfill_embeddings, 'iter_node_batches', fake_iter_node_batches)
        monkeypatch.setattr(backfill_embeddings, 'iter_edge_batches', fake_iter_edge_batches)

        with pytest.raises(BackfillError):
            await backfill_embeddings.run_backfill(driver, 'g1', batch_size=500)


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
        index_name = f'test_backfill_node_embedding_{uuid.uuid4().hex[:8]}'
        opensearch_client.indices.create(
            index=index_name, body=neptune_driver_module._vector_index_body(2)
        )
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
        batch = [{'uuid': 'bad', 'group_id': 'g1', 'embedding': [0.1, 0.2, 0.3]}]

        with pytest.raises(BackfillError):
            backfill_embeddings.index_batch(driver, index_name, batch)
