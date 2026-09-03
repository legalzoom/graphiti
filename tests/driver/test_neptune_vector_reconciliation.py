import asyncio
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock

import pytest

from graphiti_core.driver.neptune import vector_reconciliation as reconciliation
from graphiti_core.driver.neptune.operations.entity_edge_ops import (
    NeptuneEntityEdgeOperations,
)


class ReconciliationDriver:
    embedding_dim = 2

    def __init__(self) -> None:
        self.vector_projection_enabled = True
        self.pending_edge_deletes: list[dict[str, Any]] = []
        self.pending_node_deletes: list[dict[str, Any]] = []
        self.pending_node_saves: list[dict[str, Any]] = []
        self.pending_edge_saves: list[dict[str, Any]] = []
        self.incident_edge_batches: list[list[dict[str, Any]]] = []
        self.query_calls: list[tuple[str, dict[str, Any]]] = []
        self.save_calls: list[tuple[str, list[dict[str, Any]]]] = []
        self.delete_calls: list[
            tuple[str, list[str] | None, list[str] | None, dict[str, int] | None]
        ] = []
        self.save_result: Callable[[str, list[dict[str, Any]]], int] = lambda _name, documents: len(
            documents
        )
        self.save_error: BaseException | None = None
        self.delete_result: Callable[[list[str] | None, list[str] | None], int] = (
            lambda uuids, group_ids: len(uuids or group_ids or [])
        )

    def _take(self, attribute: str) -> list[dict[str, Any]]:
        records = getattr(self, attribute)
        setattr(self, attribute, [])
        return records

    async def execute_query(self, query: str, **kwargs: Any) -> Any:
        self.query_calls.append((query, kwargs))
        if 'RETURN DISTINCT edge.uuid AS uuid' in query:
            records = self.incident_edge_batches.pop(0) if self.incident_edge_batches else []
            return records, None, None
        if 'RETURN projection.uuid AS uuid, projection.group_id AS group_id' in query:
            attribute = (
                'pending_node_saves'
                if 'MATCH (projection:Entity)' in query
                else 'pending_edge_saves'
            )
            return self._take(attribute), None, None
        if 'RETURN projection.uuid AS uuid,' in query:
            attribute = (
                'pending_node_deletes'
                if 'MATCH (projection:Entity)' in query
                else 'pending_edge_deletes'
            )
            return self._take(attribute), None, None
        return [], None, None

    async def save_vector_to_aoss_async(
        self,
        name: str,
        data: list[dict[str, Any]],
    ) -> int:
        self.save_calls.append((name, data))
        if self.save_error is not None:
            raise self.save_error
        return self.save_result(name, data)

    async def delete_from_aoss_async(
        self,
        name: str,
        *,
        uuids: list[str] | None = None,
        group_ids: list[str] | None = None,
        versions: dict[str, int] | None = None,
    ) -> int:
        self.delete_calls.append((name, uuids, group_ids, versions))
        return self.delete_result(uuids, group_ids)


def pending_save(
    uuid: str,
    version: int,
    *,
    group_id: str = 'group-1',
    embedding: object = None,
) -> dict[str, Any]:
    return {
        'uuid': uuid,
        'group_id': group_id,
        'embedding': embedding,
        'projection_version': version,
    }


def acknowledgement_payloads(driver: ReconciliationDriver) -> list[list[dict[str, Any]]]:
    return [
        kwargs['completed']
        for query, kwargs in driver.query_calls
        if 'UNWIND $completed AS completed' in query
    ]


@pytest.mark.asyncio
async def test_reconcile_resumes_pending_edge_delete_at_exact_generation():
    driver = ReconciliationDriver()
    driver.pending_edge_deletes = [{'uuid': 'edge-1', 'projection_version': 7}]

    stats = await reconciliation.reconcile_pending_projections(driver, batch_size=10)

    assert stats == reconciliation.ProjectionReconciliationStats(edges_deleted=1)
    assert driver.delete_calls == [('edge_fact_embedding', ['edge-1'], None, {'edge-1': 7})]
    finalize_query, finalize_kwargs = next(
        (query, kwargs) for query, kwargs in driver.query_calls if 'DELETE projection' in query
    )
    assert 'coalesce(projection._graphiti_vector_delete_pending, false) = true' in finalize_query
    assert 'projection._graphiti_projection_version = deletion.projection_version' in finalize_query
    assert finalize_kwargs == {'deletions': [{'uuid': 'edge-1', 'projection_version': 7}]}


@pytest.mark.asyncio
async def test_reconcile_node_delete_drains_incident_edges_before_exact_finalization(
    monkeypatch,
):
    driver = ReconciliationDriver()
    driver.pending_node_deletes = [{'uuid': 'node-1', 'projection_version': 9}]
    driver.incident_edge_batches = [
        [{'uuid': 'edge-1'}, {'uuid': 'edge-1'}],
        [],
    ]
    edge_delete_calls: list[tuple[Any, list[str], int]] = []

    async def delete_edges(
        _operations,
        executor,
        uuids: list[str],
        tx=None,
        batch_size: int = 100,
    ) -> None:
        assert tx is None
        edge_delete_calls.append((executor, uuids, batch_size))

    monkeypatch.setattr(NeptuneEntityEdgeOperations, 'delete_by_uuids', delete_edges)

    stats = await reconciliation.reconcile_pending_projections(driver, batch_size=10)

    assert stats == reconciliation.ProjectionReconciliationStats(nodes_deleted=1)
    assert edge_delete_calls == [(driver, ['edge-1'], 10)]
    assert driver.delete_calls == [('node_name_embedding', ['node-1'], None, {'node-1': 9})]
    incident_query, incident_kwargs = next(
        (query, kwargs)
        for query, kwargs in driver.query_calls
        if 'RETURN DISTINCT edge.uuid AS uuid' in query
    )
    assert 'node._graphiti_projection_version = deletion.projection_version' in incident_query
    assert incident_kwargs == {
        'deletions': [{'uuid': 'node-1', 'projection_version': 9}],
        'batch_size': 10,
    }
    finalize_query, finalize_kwargs = next(
        (query, kwargs)
        for query, kwargs in driver.query_calls
        if 'DETACH DELETE projection' in query
    )
    assert 'projection._graphiti_projection_version = deletion.projection_version' in finalize_query
    assert finalize_kwargs == {'deletions': [{'uuid': 'node-1', 'projection_version': 9}]}


@pytest.mark.asyncio
async def test_reconcile_node_delete_keeps_marker_when_incident_edge_uuid_is_invalid():
    driver = ReconciliationDriver()
    driver.pending_node_deletes = [{'uuid': 'node-1', 'projection_version': 9}]
    driver.incident_edge_batches = [[{'uuid': ''}]]

    stats = await reconciliation.reconcile_pending_projections(driver, batch_size=10)

    assert stats == reconciliation.ProjectionReconciliationStats(failures=1)
    assert driver.delete_calls == []
    assert all('DETACH DELETE projection' not in query for query, _ in driver.query_calls)


@pytest.mark.asyncio
async def test_reconcile_isolates_poison_save_and_only_clears_good_markers():
    driver = ReconciliationDriver()
    driver.pending_node_saves = [
        pending_save('node-good-1', 1, embedding=[1.0, 0.0]),
        pending_save('node-bad', 2, embedding='malformed'),
        pending_save('node-good-2', 3, embedding=[0.0, 1.0]),
    ]

    stats = await reconciliation.reconcile_pending_projections(driver, batch_size=10)

    assert stats == reconciliation.ProjectionReconciliationStats(nodes_saved=2, failures=1)
    assert [[document['uuid'] for document in documents] for _, documents in driver.save_calls] == [
        ['node-good-1'],
        ['node-good-2'],
    ]
    assert acknowledgement_payloads(driver) == [
        [{'uuid': 'node-good-1', 'projection_version': 1}],
        [{'uuid': 'node-good-2', 'projection_version': 3}],
    ]


@pytest.mark.asyncio
async def test_reconcile_isolates_partial_aoss_batch_before_clearing_markers():
    driver = ReconciliationDriver()
    driver.pending_node_saves = [
        pending_save('node-1', 1, embedding=[1.0, 0.0]),
        pending_save('node-2', 2, embedding=[0.0, 1.0]),
        pending_save('node-3', 3, embedding=None),
    ]
    driver.save_result = (
        lambda _name, documents: len(documents) if len(documents) == 1 else len(documents) - 1
    )

    stats = await reconciliation.reconcile_pending_projections(driver, batch_size=10)

    assert stats == reconciliation.ProjectionReconciliationStats(nodes_saved=3)
    assert [len(documents) for _, documents in driver.save_calls] == [3, 1, 2, 1, 1]
    assert acknowledgement_payloads(driver) == [
        [{'uuid': 'node-1', 'projection_version': 1}],
        [{'uuid': 'node-2', 'projection_version': 2}],
        [{'uuid': 'node-3', 'projection_version': 3}],
    ]


@pytest.mark.asyncio
async def test_reconcile_recovers_save_with_empty_group_id():
    driver = ReconciliationDriver()
    driver.pending_edge_saves = [pending_save('edge-1', 4, group_id='', embedding=[1.0, 0.0])]

    stats = await reconciliation.reconcile_pending_projections(driver, batch_size=10)

    assert stats == reconciliation.ProjectionReconciliationStats(edges_saved=1)
    assert driver.save_calls == [
        (
            'edge_fact_embedding',
            [
                {
                    'uuid': 'edge-1',
                    'group_id': '',
                    '_version': 4,
                    'embedding': [1.0, 0.0],
                }
            ],
        )
    ]
    assert acknowledgement_payloads(driver) == [[{'uuid': 'edge-1', 'projection_version': 4}]]


@pytest.mark.asyncio
async def test_disabled_projection_still_repairs_deletes_but_skips_pending_saves():
    driver = ReconciliationDriver()
    driver.vector_projection_enabled = False
    driver.pending_edge_deletes = [{'uuid': 'edge-delete', 'projection_version': 5}]
    driver.pending_node_saves = [pending_save('node-save', 6, embedding=[1.0, 0.0])]

    stats = await reconciliation.reconcile_pending_projections(driver, batch_size=10)

    assert stats == reconciliation.ProjectionReconciliationStats(edges_deleted=1)
    assert driver.delete_calls == [
        ('edge_fact_embedding', ['edge-delete'], None, {'edge-delete': 5})
    ]
    assert driver.save_calls == []
    assert acknowledgement_payloads(driver) == []
    assert [record['uuid'] for record in driver.pending_node_saves] == ['node-save']


@pytest.mark.asyncio
async def test_reconcile_propagates_cancellation_without_clearing_marker():
    driver = ReconciliationDriver()
    driver.pending_node_saves = [pending_save('node-1', 1, embedding=[1.0, 0.0])]
    driver.save_error = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await reconciliation.reconcile_pending_projections(driver, batch_size=10)

    assert acknowledgement_payloads(driver) == []


@pytest.mark.asyncio
async def test_periodic_reconciler_retries_sweep_failures_and_propagates_cancellation(
    monkeypatch,
):
    sweep = AsyncMock(
        side_effect=[
            RuntimeError('temporary Neptune failure'),
            reconciliation.ProjectionReconciliationStats(nodes_saved=1),
        ]
    )
    sleep_intervals: list[float] = []

    async def sleep(interval: float) -> None:
        sleep_intervals.append(interval)
        if len(sleep_intervals) == 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(reconciliation, 'reconcile_pending_projections', sweep)
    monkeypatch.setattr(reconciliation.asyncio, 'sleep', sleep)

    with pytest.raises(asyncio.CancelledError):
        await reconciliation.run_pending_projection_reconciler(
            ReconciliationDriver(),
            interval_seconds=2.5,
            batch_size=7,
        )

    assert sweep.await_count == 2
    assert all(call.kwargs == {'batch_size': 7} for call in sweep.await_args_list)
    assert sleep_intervals == [2.5, 2.5]
