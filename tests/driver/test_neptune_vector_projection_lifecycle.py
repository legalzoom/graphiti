"""Unit coverage for Neptune entity vector projection lifecycle."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, cast

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.neptune.operations.entity_edge_ops import NeptuneEntityEdgeOperations
from graphiti_core.driver.neptune.operations.entity_node_ops import NeptuneEntityNodeOperations
from graphiti_core.driver.neptune.operations.graph_ops import NeptuneGraphMaintenanceOperations
from graphiti_core.driver.neptune.projection_versions import defer_cancellation_until_complete
from graphiti_core.driver.neptune_driver import NeptuneDriver
from graphiti_core.driver.record_parsers import (
    entity_edge_from_neptune_record,
    entity_edge_from_record,
    entity_node_from_neptune_record,
    entity_node_from_record,
)
from graphiti_core.edges import Edge, EntityEdge, get_entity_edge_from_record
from graphiti_core.errors import NodeGroupMismatchError
from graphiti_core.nodes import EntityNode, Node, get_entity_node_from_record

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_deferred_cancellation_survives_inner_failure(caplog):
    started = asyncio.Event()
    fail_boundary = asyncio.Event()
    failure_secret = 'projection-failure-secret'

    @defer_cancellation_until_complete
    async def consistency_boundary() -> None:
        started.set()
        await fail_boundary.wait()
        raise RuntimeError(failure_secret)

    caplog.set_level(
        'ERROR',
        logger='graphiti_core.driver.neptune.projection_versions',
    )
    task = asyncio.create_task(consistency_boundary())
    await started.wait()

    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()

    fail_boundary.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert 'error_type=RuntimeError' in caplog.text
    assert failure_secret not in caplog.text


class QueryRecorder:
    def __init__(self, responses: list[Any] | None = None):
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._responses = list(responses or [])

    async def execute_query(self, query: str, **kwargs: Any) -> Any:
        return await self.run(query, **kwargs)

    async def run(self, query: str, **kwargs: Any) -> Any:
        self.calls.append((query, kwargs))
        if self._responses:
            return self._responses.pop(0)
        if 'projections' in kwargs:
            return (
                [
                    {
                        'uuid': projection['uuid'],
                        'projection_version': index + 1,
                    }
                    for index, projection in enumerate(kwargs['projections'])
                ],
                None,
                None,
            )
        if 'entity_data' in kwargs:
            return (
                [
                    {
                        'uuid': kwargs['entity_data']['uuid'],
                        'projection_version': 1,
                    }
                ],
                None,
                None,
            )
        if 'nodes' in kwargs:
            return (
                [
                    {'uuid': node['uuid'], 'projection_version': index + 1}
                    for index, node in enumerate(kwargs['nodes'])
                ],
                None,
                None,
            )
        if 'edge_data' in kwargs:
            return (
                [
                    {
                        'uuid': kwargs['edge_data']['uuid'],
                        'projection_version': 1,
                    }
                ],
                None,
                None,
            )
        if 'entity_edges' in kwargs:
            return (
                [
                    {'uuid': edge['uuid'], 'projection_version': index + 1}
                    for index, edge in enumerate(kwargs['entity_edges'])
                ],
                None,
                None,
            )
        if 'completed' in kwargs:
            return ([{'uuid': item['uuid']} for item in kwargs['completed']], None, None)
        return [], None, None


class ProjectionRecorder:
    def __init__(
        self,
        save_results: list[int] | None = None,
        delete_error: Exception | None = None,
    ):
        self.vector_projection_enabled = True
        self.save_calls: list[tuple[str, list[dict[str, Any]]]] = []
        self.text_save_calls: list[tuple[str, list[dict[str, Any]]]] = []
        self.delete_calls: list[
            tuple[str, list[str] | None, list[str] | None, dict[str, int] | None]
        ] = []
        self._save_results = list(save_results or [])
        self._delete_error = delete_error

    def save_to_aoss(self, name: str, data: list[dict[str, Any]]) -> int:
        self.text_save_calls.append((name, data))
        return len(data)

    def save_vector_to_aoss(self, name: str, data: list[dict[str, Any]]) -> int:
        self.save_calls.append((name, data))
        if self._save_results:
            return self._save_results.pop(0)
        return len(data)

    async def save_vector_to_aoss_async(self, name: str, data: list[dict[str, Any]]) -> int:
        return self.save_vector_to_aoss(name, data)

    def delete_from_aoss(
        self,
        name: str,
        *,
        uuids: list[str] | None = None,
        group_ids: list[str] | None = None,
        versions: dict[str, int] | None = None,
    ) -> int:
        self.delete_calls.append((name, uuids, group_ids, versions))
        if self._delete_error is not None:
            raise self._delete_error
        return len(uuids or group_ids or [])

    async def delete_from_aoss_async(
        self,
        name: str,
        *,
        uuids: list[str] | None = None,
        group_ids: list[str] | None = None,
        versions: dict[str, int] | None = None,
    ) -> int:
        return self.delete_from_aoss(
            name,
            uuids=uuids,
            group_ids=group_ids,
            versions=versions,
        )


class BlockingProjectionRecorder(ProjectionRecorder):
    def __init__(self) -> None:
        super().__init__()
        self.delete_started = asyncio.Event()
        self.allow_delete = asyncio.Event()

    async def delete_from_aoss_async(
        self,
        name: str,
        *,
        uuids: list[str] | None = None,
        group_ids: list[str] | None = None,
        versions: dict[str, int] | None = None,
    ) -> int:
        self.delete_started.set()
        await self.allow_delete.wait()
        return self.delete_from_aoss(
            name,
            uuids=uuids,
            group_ids=group_ids,
            versions=versions,
        )


def projection_driver(recorder: ProjectionRecorder) -> NeptuneDriver:
    return cast(NeptuneDriver, recorder)


def make_node(uuid: str, embedding: list[float] | None) -> EntityNode:
    return EntityNode(
        uuid=uuid,
        name=f'node-{uuid}',
        group_id='group-1',
        labels=['Person'],
        created_at=NOW,
        name_embedding=embedding,
    )


def make_edge(uuid: str, embedding: list[float] | None) -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        name='KNOWS',
        fact=f'fact-{uuid}',
        fact_embedding=embedding,
        group_id='group-1',
        source_node_uuid='source',
        target_node_uuid='target',
        created_at=NOW,
    )


class TestNodeVectorProjectionWrites:
    @pytest.mark.asyncio
    async def test_save_rejects_projection_reserved_attributes_before_graph_write(self):
        projection = ProjectionRecorder()
        ops = NeptuneEntityNodeOperations(projection_driver(projection))
        executor = QueryRecorder()
        node = make_node('node-1', [0.1])
        node.attributes['_graphiti_vector_delete_pending'] = True

        with pytest.raises(ValueError, match='_graphiti_vector_delete_pending'):
            await ops.save(executor, node)

        assert executor.calls == []
        assert projection.save_calls == []

    @pytest.mark.asyncio
    async def test_save_uses_transaction_then_projects_embedding(self):
        projection = ProjectionRecorder()
        ops = NeptuneEntityNodeOperations(projection_driver(projection))
        executor = QueryRecorder()
        transaction = QueryRecorder([([{'uuid': 'node-1', 'projection_version': 1}], None, None)])
        node = make_node('node-1', [0.1, 0.2])

        await ops.save(executor, node, tx=transaction)

        assert executor.calls == []
        assert len(transaction.calls) == 2
        acknowledgement_query, acknowledgement_kwargs = transaction.calls[1]
        assert (
            '_graphiti_projection_version = completed.projection_version' in acknowledgement_query
        )
        assert (
            '_graphiti_vector_sync_pending = completed.projection_version' in acknowledgement_query
        )
        assert 'REMOVE projection._graphiti_vector_sync_pending' in acknowledgement_query
        assert acknowledgement_kwargs == {
            'completed': [{'uuid': 'node-1', 'projection_version': 1}]
        }
        assert projection.save_calls == [
            (
                'node_name_embedding',
                [
                    {
                        'uuid': 'node-1',
                        'group_id': 'group-1',
                        'embedding': [0.1, 0.2],
                        '_version': 1,
                    }
                ],
            )
        ]
        assert projection.delete_calls == []

    @pytest.mark.asyncio
    async def test_save_without_embedding_removes_prior_projection(self):
        projection = ProjectionRecorder()
        ops = NeptuneEntityNodeOperations(projection_driver(projection))
        executor = QueryRecorder([([{'uuid': 'node-1', 'projection_version': 1}], None, None)])

        await ops.save(executor, make_node('node-1', None))

        assert projection.save_calls == [
            (
                'node_name_embedding',
                [{'uuid': 'node-1', 'group_id': 'group-1', '_version': 1}],
            )
        ]
        assert projection.delete_calls == []

    @pytest.mark.asyncio
    async def test_disabled_projection_keeps_graph_generation_pending_for_later_repair(self):
        projection = ProjectionRecorder()
        projection.vector_projection_enabled = False
        ops = NeptuneEntityNodeOperations(projection_driver(projection))
        executor = QueryRecorder()

        await ops.save(executor, make_node('node-1', [0.1]))

        assert len(executor.calls) == 1
        assert projection.save_calls == []
        assert projection.text_save_calls[0][0] == 'node_name_and_summary'

    @pytest.mark.asyncio
    async def test_save_raises_when_projection_write_is_short(self):
        projection = ProjectionRecorder(save_results=[0])
        ops = NeptuneEntityNodeOperations(projection_driver(projection))
        executor = QueryRecorder([([{'uuid': 'node-1', 'projection_version': 1}], None, None)])

        with pytest.raises(RuntimeError, match=r'indexed 0/1 documents'):
            await ops.save(executor, make_node('node-1', [0.1]))

    @pytest.mark.asyncio
    async def test_save_bulk_projects_embedded_and_removes_unembedded(self):
        projection = ProjectionRecorder()
        ops = NeptuneEntityNodeOperations(projection_driver(projection))
        executor = QueryRecorder()
        nodes = [
            make_node('node-1', [0.1]),
            make_node('node-2', None),
            make_node('node-3', [0.3]),
        ]

        await ops.save_bulk(executor, nodes, batch_size=2)

        assert len(executor.calls) == 4
        assert [kwargs['completed'] for _, kwargs in executor.calls[2:]] == [
            [
                {'uuid': 'node-1', 'projection_version': 1},
                {'uuid': 'node-2', 'projection_version': 2},
            ],
            [{'uuid': 'node-3', 'projection_version': 1}],
        ]
        assert projection.save_calls == [
            (
                'node_name_embedding',
                [
                    {
                        'uuid': 'node-1',
                        'group_id': 'group-1',
                        '_version': 1,
                        'embedding': [0.1],
                    },
                    {'uuid': 'node-2', 'group_id': 'group-1', '_version': 2},
                ],
            ),
            (
                'node_name_embedding',
                [
                    {
                        'uuid': 'node-3',
                        'group_id': 'group-1',
                        '_version': 1,
                        'embedding': [0.3],
                    },
                ],
            ),
        ]
        assert projection.delete_calls == []

    @pytest.mark.asyncio
    async def test_save_bulk_rejects_projection_reserved_attributes(self):
        projection = ProjectionRecorder()
        ops = NeptuneEntityNodeOperations(projection_driver(projection))
        executor = QueryRecorder()
        node = make_node('node-1', [0.1])
        node.attributes['_graphiti_injected'] = 'unsafe'

        with pytest.raises(ValueError, match='_graphiti_injected'):
            await ops.save_bulk(executor, [node])

        assert executor.calls == []

    @pytest.mark.asyncio
    async def test_save_bulk_rejects_partial_neptune_write(self):
        projection = ProjectionRecorder()
        ops = NeptuneEntityNodeOperations(projection_driver(projection))
        executor = QueryRecorder([([{'uuid': 'node-1', 'projection_version': 1}], None, None)])

        with pytest.raises(NodeGroupMismatchError):
            await ops.save_bulk(
                executor,
                [make_node('node-1', [0.1]), make_node('node-2', [0.2])],
            )

        assert projection.text_save_calls == []
        assert projection.save_calls == []

    @pytest.mark.asyncio
    async def test_save_bulk_collapses_duplicate_uuids_before_both_writes(self):
        projection = ProjectionRecorder()
        ops = NeptuneEntityNodeOperations(projection_driver(projection))
        executor = QueryRecorder()
        first = make_node('node-1', [0.1])
        first.name = 'first'
        last = make_node('node-1', [0.9])
        last.name = 'last'

        await ops.save_bulk(executor, [first, last])

        assert len(executor.calls) == 2
        persisted_node = executor.calls[0][1]['nodes'][0]
        assert {key: value for key, value in persisted_node.items() if key != 'labels'} == {
            'uuid': 'node-1',
            'name': 'last',
            'group_id': 'group-1',
            'summary': '',
            'created_at': NOW,
            'name_embedding': [0.9],
        }
        assert set(persisted_node['labels']) == {'Entity', 'Person'}
        assert executor.calls[1][1] == {'completed': [{'uuid': 'node-1', 'projection_version': 1}]}
        assert projection.save_calls == [
            (
                'node_name_embedding',
                [
                    {
                        'uuid': 'node-1',
                        'group_id': 'group-1',
                        'embedding': [0.9],
                        '_version': 1,
                    }
                ],
            )
        ]


class TestEdgeVectorProjectionWrites:
    @pytest.mark.asyncio
    async def test_save_rejects_projection_reserved_attributes_before_graph_write(self):
        projection = ProjectionRecorder()
        ops = NeptuneEntityEdgeOperations(projection_driver(projection))
        executor = QueryRecorder()
        edge = make_edge('edge-1', [0.1])
        edge.attributes['_graphiti_vector_delete_pending'] = True

        with pytest.raises(ValueError, match='_graphiti_vector_delete_pending'):
            await ops.save(executor, edge)

        assert executor.calls == []
        assert projection.save_calls == []

    @pytest.mark.asyncio
    async def test_save_and_save_bulk_persist_reference_time(self):
        projection = ProjectionRecorder()
        ops = NeptuneEntityEdgeOperations(projection_driver(projection))
        executor = QueryRecorder()
        direct = make_edge('edge-1', [0.1])
        direct.reference_time = NOW
        bulk = make_edge('edge-2', [0.2])
        bulk.reference_time = NOW

        await ops.save(executor, direct)
        await ops.save_bulk(executor, [bulk])

        assert executor.calls[0][1]['edge_data']['reference_time'] == NOW
        assert executor.calls[2][1]['entity_edges'][0]['reference_time'] == NOW

    @pytest.mark.asyncio
    async def test_disabled_projection_keeps_graph_generation_pending_for_later_repair(self):
        projection = ProjectionRecorder()
        projection.vector_projection_enabled = False
        ops = NeptuneEntityEdgeOperations(projection_driver(projection))
        executor = QueryRecorder()

        await ops.save(executor, make_edge('edge-1', [0.1]))

        assert len(executor.calls) == 1
        assert projection.save_calls == []
        assert projection.text_save_calls[0][0] == 'edge_name_and_fact'

    @pytest.mark.asyncio
    async def test_save_bulk_projects_embedded_and_removes_unembedded(self):
        projection = ProjectionRecorder()
        ops = NeptuneEntityEdgeOperations(projection_driver(projection))
        executor = QueryRecorder(
            [
                (
                    [
                        {'uuid': 'edge-1', 'projection_version': 1},
                        {'uuid': 'edge-2', 'projection_version': 2},
                    ],
                    None,
                    None,
                ),
                ([{'uuid': 'edge-3', 'projection_version': 1}], None, None),
            ]
        )
        edges = [
            make_edge('edge-1', [0.1]),
            make_edge('edge-2', None),
            make_edge('edge-3', [0.3]),
        ]

        await ops.save_bulk(executor, edges, batch_size=2)

        assert len(executor.calls) == 4
        assert [kwargs['completed'] for _, kwargs in executor.calls[2:]] == [
            [
                {'uuid': 'edge-1', 'projection_version': 1},
                {'uuid': 'edge-2', 'projection_version': 2},
            ],
            [{'uuid': 'edge-3', 'projection_version': 1}],
        ]
        assert projection.save_calls == [
            (
                'edge_fact_embedding',
                [
                    {
                        'uuid': 'edge-1',
                        'group_id': 'group-1',
                        '_version': 1,
                        'embedding': [0.1],
                    },
                    {'uuid': 'edge-2', 'group_id': 'group-1', '_version': 2},
                ],
            ),
            (
                'edge_fact_embedding',
                [
                    {
                        'uuid': 'edge-3',
                        'group_id': 'group-1',
                        '_version': 1,
                        'embedding': [0.3],
                    },
                ],
            ),
        ]

    @pytest.mark.asyncio
    async def test_save_raises_when_projection_write_is_short(self):
        projection = ProjectionRecorder(save_results=[0])
        ops = NeptuneEntityEdgeOperations(projection_driver(projection))
        executor = QueryRecorder([([{'uuid': 'edge-1', 'projection_version': 1}], None, None)])

        with pytest.raises(RuntimeError, match=r'indexed 0/1 documents'):
            await ops.save(executor, make_edge('edge-1', [0.1]))

    @pytest.mark.asyncio
    async def test_save_does_not_project_when_edge_was_not_persisted(self):
        projection = ProjectionRecorder()
        ops = NeptuneEntityEdgeOperations(projection_driver(projection))
        executor = QueryRecorder([([], None, None)])

        with pytest.raises(NodeGroupMismatchError):
            await ops.save(executor, make_edge('edge-1', [0.1]))

        assert projection.save_calls == []
        assert projection.delete_calls == []

    @pytest.mark.asyncio
    async def test_save_bulk_rejects_partial_neptune_write(self):
        projection = ProjectionRecorder()
        ops = NeptuneEntityEdgeOperations(projection_driver(projection))
        executor = QueryRecorder(
            [
                (
                    [
                        {'uuid': 'edge-1', 'projection_version': 1},
                        {'uuid': 'edge-3', 'projection_version': 3},
                    ],
                    None,
                    None,
                )
            ]
        )
        edges = [
            make_edge('edge-1', [0.1]),
            make_edge('edge-2', [0.2]),
            make_edge('edge-3', None),
        ]

        with pytest.raises(NodeGroupMismatchError):
            await ops.save_bulk(executor, edges, batch_size=10)

        assert projection.text_save_calls == []
        assert projection.save_calls == []
        assert projection.delete_calls == []

    @pytest.mark.asyncio
    async def test_save_bulk_rejects_projection_reserved_attributes(self):
        projection = ProjectionRecorder()
        ops = NeptuneEntityEdgeOperations(projection_driver(projection))
        executor = QueryRecorder()
        edge = make_edge('edge-1', [0.1])
        edge.attributes['_graphiti_injected'] = 'unsafe'

        with pytest.raises(ValueError, match='_graphiti_injected'):
            await ops.save_bulk(executor, [edge])

        assert executor.calls == []

    @pytest.mark.asyncio
    async def test_save_bulk_collapses_duplicate_uuids_before_both_writes(self):
        projection = ProjectionRecorder()
        ops = NeptuneEntityEdgeOperations(projection_driver(projection))
        executor = QueryRecorder()
        first = make_edge('edge-1', [0.1])
        first.fact = 'first'
        last = make_edge('edge-1', [0.9])
        last.fact = 'last'

        await ops.save_bulk(executor, [first, last])

        assert len(executor.calls) == 2
        assert executor.calls[0][1]['entity_edges'][0]['fact'] == 'last'
        assert executor.calls[0][1]['entity_edges'][0]['fact_embedding'] == [0.9]
        assert len(executor.calls[0][1]['entity_edges']) == 1
        assert executor.calls[1][1] == {'completed': [{'uuid': 'edge-1', 'projection_version': 1}]}
        assert projection.save_calls == [
            (
                'edge_fact_embedding',
                [
                    {
                        'uuid': 'edge-1',
                        'group_id': 'group-1',
                        'embedding': [0.9],
                        '_version': 1,
                    }
                ],
            )
        ]


class TestProjectionAttributeParsing:
    @pytest.mark.parametrize(
        'parser',
        [
            entity_node_from_record,
            lambda record: get_entity_node_from_record(record, GraphProvider.NEO4J),
            lambda record: get_entity_node_from_record(record, GraphProvider.FALKORDB),
        ],
    )
    def test_non_neptune_node_reads_preserve_graphiti_prefixed_user_attributes(self, parser):
        record = {
            'uuid': 'node-1',
            'name': 'node',
            'group_id': 'group-1',
            'name_embedding': [0.1],
            'summary': '',
            'created_at': NOW,
            'labels': ['Entity'],
            'attributes': {
                'uuid': 'node-1',
                '_graphiti_user_attribute': 'preserved',
            },
        }

        node = parser(record)

        assert node.attributes == {'_graphiti_user_attribute': 'preserved'}

    @pytest.mark.parametrize(
        'parser',
        [
            entity_edge_from_record,
            lambda record: get_entity_edge_from_record(record, GraphProvider.NEO4J),
            lambda record: get_entity_edge_from_record(record, GraphProvider.FALKORDB),
        ],
    )
    def test_non_neptune_edge_reads_preserve_graphiti_prefixed_user_attributes(self, parser):
        record = {
            'uuid': 'edge-1',
            'source_node_uuid': 'source',
            'target_node_uuid': 'target',
            'fact': 'fact',
            'fact_embedding': [0.1],
            'name': 'KNOWS',
            'group_id': 'group-1',
            'episodes': [],
            'created_at': NOW,
            'expired_at': None,
            'valid_at': None,
            'invalid_at': None,
            'reference_time': None,
            'attributes': {
                'uuid': 'edge-1',
                '_graphiti_user_attribute': 'preserved',
            },
        }

        edge = parser(record)

        assert edge.attributes == {'_graphiti_user_attribute': 'preserved'}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'parser',
        [
            entity_node_from_neptune_record,
            lambda record: get_entity_node_from_record(record, GraphProvider.NEPTUNE),
        ],
    )
    async def test_node_read_modify_save_hides_internal_projection_attributes(self, parser):
        attributes = {
            'uuid': 'node-1',
            'name': 'node',
            'group_id': 'group-1',
            'name_embedding': '0.1',
            'summary': '',
            'created_at': NOW,
            '_graphiti_projection_version': 4,
            '_graphiti_vector_delete_pending': False,
            '_graphiti_vector_sync_pending': 4,
            'custom': 'visible',
        }
        record = {
            'uuid': 'node-1',
            'name': 'node',
            'group_id': 'group-1',
            'name_embedding': [0.1],
            'summary': '',
            'created_at': NOW,
            'labels': ['Entity', 'Person'],
            'attributes': attributes,
        }

        node = parser(record)

        assert node.attributes == {'custom': 'visible'}
        assert attributes['_graphiti_projection_version'] == 4
        projection = ProjectionRecorder()
        await NeptuneEntityNodeOperations(projection_driver(projection)).save(
            QueryRecorder(),
            node,
        )
        assert projection.save_calls

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'parser',
        [
            entity_edge_from_neptune_record,
            lambda record: get_entity_edge_from_record(record, GraphProvider.NEPTUNE),
        ],
    )
    async def test_edge_read_modify_save_hides_internal_projection_attributes(self, parser):
        attributes = {
            'uuid': 'edge-1',
            'source_node_uuid': 'source',
            'target_node_uuid': 'target',
            'fact': 'fact',
            'fact_embedding': '0.1',
            'name': 'KNOWS',
            'group_id': 'group-1',
            'episodes': '',
            'created_at': NOW,
            '_graphiti_projection_version': 4,
            '_graphiti_vector_delete_pending': False,
            '_graphiti_vector_sync_pending': 4,
            'custom': 'visible',
        }
        record = {
            'uuid': 'edge-1',
            'source_node_uuid': 'source',
            'target_node_uuid': 'target',
            'fact': 'fact',
            'fact_embedding': [0.1],
            'name': 'KNOWS',
            'group_id': 'group-1',
            'episodes': [],
            'created_at': NOW,
            'expired_at': None,
            'valid_at': None,
            'invalid_at': None,
            'reference_time': None,
            'attributes': attributes,
        }

        edge = parser(record)

        assert edge.attributes == {'custom': 'visible'}
        assert attributes['_graphiti_projection_version'] == 4
        projection = ProjectionRecorder()
        await NeptuneEntityEdgeOperations(projection_driver(projection)).save(
            QueryRecorder(),
            edge,
        )
        assert projection.save_calls
        assert projection.delete_calls == []


class TestVectorProjectionDeletes:
    @pytest.mark.asyncio
    async def test_node_delete_removes_node_and_incident_edge_vectors(self):
        projection = ProjectionRecorder()
        ops = NeptuneEntityNodeOperations(projection_driver(projection))
        executor = QueryRecorder()
        transaction = QueryRecorder(
            [
                ([{'uuid': 'node-1', 'projection_version': 1}], None, None),
                ([{'uuid': 'node-1'}], None, None),
                ([{'uuid': 'edge-1'}, {'uuid': 'edge-2'}], None, None),
                (
                    [
                        {'uuid': 'edge-1', 'projection_version': 1},
                        {'uuid': 'edge-2', 'projection_version': 2},
                    ],
                    None,
                    None,
                ),
                ([], None, None),
                ([], None, None),
                ([], None, None),
                ([], None, None),
            ]
        )

        await ops.delete(executor, make_node('node-1', [0.1]), tx=transaction)

        assert executor.calls == []
        assert len(transaction.calls) == 8
        reserve_query, reserve_kwargs = transaction.calls[0]
        prepare_query, prepare_kwargs = transaction.calls[1]
        incident_query, incident_kwargs = transaction.calls[2]
        edge_reserve_query, edge_reserve_kwargs = transaction.calls[3]
        delete_query, delete_kwargs = transaction.calls[7]
        assert 'GraphitiProjectionVersion' in reserve_query
        assert reserve_kwargs['projections'] == [{'uuid': 'node-1', 'projection_id': 'node:node-1'}]
        assert '_graphiti_vector_delete_pending = true' in prepare_query
        assert prepare_kwargs == {'deletions': [{'uuid': 'node-1', 'projection_version': 1}]}
        assert 'RETURN DISTINCT e.uuid AS uuid' in incident_query
        assert incident_kwargs == {
            'deletions': [{'uuid': 'node-1', 'projection_version': 1}],
            'batch_size': 100,
        }
        assert 'GraphitiProjectionVersion' in edge_reserve_query
        assert edge_reserve_kwargs['projections'] == [
            {'uuid': 'edge-1', 'projection_id': 'edge:edge-1'},
            {'uuid': 'edge-2', 'projection_id': 'edge:edge-2'},
        ]
        assert 'DETACH DELETE n' in delete_query
        assert delete_kwargs == {'deletions': [{'uuid': 'node-1', 'projection_version': 1}]}
        assert projection.delete_calls == [
            (
                'edge_fact_embedding',
                ['edge-1', 'edge-2'],
                None,
                {'edge-1': 1, 'edge-2': 2},
            ),
            ('node_name_embedding', ['node-1'], None, {'node-1': 1}),
        ]

    @pytest.mark.asyncio
    async def test_node_delete_does_not_delete_graph_when_projection_cleanup_fails(self):
        projection = ProjectionRecorder(delete_error=RuntimeError('AOSS unavailable'))
        ops = NeptuneEntityNodeOperations(projection_driver(projection))
        transaction = QueryRecorder(
            [
                ([{'uuid': 'node-1', 'projection_version': 1}], None, None),
                ([{'uuid': 'node-1'}], None, None),
                ([{'uuid': 'edge-1'}], None, None),
                ([{'uuid': 'edge-1', 'projection_version': 1}], None, None),
                ([], None, None),
            ]
        )

        with pytest.raises(RuntimeError, match='AOSS unavailable'):
            await ops.delete(QueryRecorder(), make_node('node-1', [0.1]), tx=transaction)

        assert len(transaction.calls) == 5
        assert all('DETACH DELETE n' not in query for query, _ in transaction.calls)
        assert all('DELETE e' not in query for query, _ in transaction.calls)

    @pytest.mark.asyncio
    async def test_node_delete_by_uuids_deduplicates_incident_edge_vectors(self):
        projection = ProjectionRecorder()
        ops = NeptuneEntityNodeOperations(projection_driver(projection))
        executor = QueryRecorder(
            [
                (
                    [
                        {'uuid': 'node-1', 'projection_version': 1},
                        {'uuid': 'node-2', 'projection_version': 2},
                    ],
                    None,
                    None,
                ),
                ([{'uuid': 'node-1'}, {'uuid': 'node-2'}], None, None),
                (
                    [{'uuid': 'edge-1'}, {'uuid': 'edge-shared'}, {'uuid': 'edge-2'}],
                    None,
                    None,
                ),
                (
                    [
                        {'uuid': 'edge-1', 'projection_version': 1},
                        {'uuid': 'edge-shared', 'projection_version': 2},
                        {'uuid': 'edge-2', 'projection_version': 3},
                    ],
                    None,
                    None,
                ),
                ([], None, None),
                ([], None, None),
                ([], None, None),
                ([], None, None),
            ]
        )

        await ops.delete_by_uuids(executor, ['node-1', 'node-2'])

        assert projection.delete_calls == [
            (
                'edge_fact_embedding',
                ['edge-1', 'edge-shared', 'edge-2'],
                None,
                {'edge-1': 1, 'edge-shared': 2, 'edge-2': 3},
            ),
            (
                'node_name_embedding',
                ['node-1', 'node-2'],
                None,
                {'node-1': 1, 'node-2': 2},
            ),
        ]

    @pytest.mark.asyncio
    async def test_node_delete_by_group_removes_both_vector_indices(self):
        projection = ProjectionRecorder()
        ops = NeptuneEntityNodeOperations(projection_driver(projection))
        executor = QueryRecorder(
            [
                ([{'uuid': 'node-1'}], None, None),
                ([{'uuid': 'node-1', 'projection_version': 1}], None, None),
                ([{'uuid': 'node-1'}], None, None),
                ([{'uuid': 'edge-1'}], None, None),
                ([{'uuid': 'edge-1', 'projection_version': 1}], None, None),
                ([], None, None),
                ([], None, None),
                ([], None, None),
                ([], None, None),
                ([], None, None),
            ]
        )

        await ops.delete_by_group_id(executor, 'group-1')

        assert projection.delete_calls == [
            ('edge_fact_embedding', ['edge-1'], None, {'edge-1': 1}),
            ('node_name_embedding', ['node-1'], None, {'node-1': 1}),
        ]

    @pytest.mark.asyncio
    async def test_edge_delete_paths_remove_vector_documents(self):
        projection = ProjectionRecorder()
        ops = NeptuneEntityEdgeOperations(projection_driver(projection))
        executor = QueryRecorder()

        await ops.delete(executor, make_edge('edge-1', [0.1]))
        await ops.delete_by_uuids(executor, ['edge-2', 'edge-3'])

        assert projection.delete_calls == [
            ('edge_fact_embedding', ['edge-1'], None, {'edge-1': 1}),
            (
                'edge_fact_embedding',
                ['edge-2', 'edge-3'],
                None,
                {'edge-2': 1, 'edge-3': 2},
            ),
        ]

    @pytest.mark.asyncio
    async def test_node_delete_by_uuids_chunks_graph_and_vector_work(self):
        projection = ProjectionRecorder()
        ops = NeptuneEntityNodeOperations(projection_driver(projection))
        executor = QueryRecorder()

        await ops.delete_by_uuids(
            executor,
            ['node-1', 'node-2', 'node-3', 'node-4', 'node-5'],
            batch_size=2,
        )

        assert [call[1]['projections'] for call in executor.calls[::4]] == [
            [
                {'uuid': 'node-1', 'projection_id': 'node:node-1'},
                {'uuid': 'node-2', 'projection_id': 'node:node-2'},
            ],
            [
                {'uuid': 'node-3', 'projection_id': 'node:node-3'},
                {'uuid': 'node-4', 'projection_id': 'node:node-4'},
            ],
            [{'uuid': 'node-5', 'projection_id': 'node:node-5'}],
        ]
        assert [call[1] for call in projection.delete_calls] == [
            ['node-1', 'node-2'],
            ['node-3', 'node-4'],
            ['node-5'],
        ]

    @pytest.mark.asyncio
    async def test_edge_delete_by_uuids_chunks_graph_and_vector_work(self):
        projection = ProjectionRecorder()
        ops = NeptuneEntityEdgeOperations(projection_driver(projection))
        executor = QueryRecorder()

        await ops.delete_by_uuids(
            executor,
            ['edge-1', 'edge-2', 'edge-3', 'edge-4', 'edge-5'],
            batch_size=2,
        )

        assert [call[1]['projections'] for call in executor.calls[::3]] == [
            [
                {'uuid': 'edge-1', 'projection_id': 'edge:edge-1'},
                {'uuid': 'edge-2', 'projection_id': 'edge:edge-2'},
            ],
            [
                {'uuid': 'edge-3', 'projection_id': 'edge:edge-3'},
                {'uuid': 'edge-4', 'projection_id': 'edge:edge-4'},
            ],
            [{'uuid': 'edge-5', 'projection_id': 'edge:edge-5'}],
        ]
        assert [call[1] for call in projection.delete_calls] == [
            ['edge-1', 'edge-2'],
            ['edge-3', 'edge-4'],
            ['edge-5'],
        ]

    @pytest.mark.asyncio
    async def test_cancellation_waits_for_graph_finalization_before_raising(self):
        projection = BlockingProjectionRecorder()
        ops = NeptuneEntityEdgeOperations(projection_driver(projection))
        executor = QueryRecorder()

        task = asyncio.create_task(ops.delete_by_uuids(executor, ['edge-1']))
        await projection.delete_started.wait()
        task.cancel()
        await asyncio.sleep(0)

        assert not task.done()
        assert all('DELETE e' not in query for query, _ in executor.calls)

        projection.allow_delete.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert 'DELETE e' in executor.calls[-1][0]


class DeleteRecorder:
    def __init__(self) -> None:
        self.uuid_calls: list[tuple[list[str], int]] = []
        self.group_calls: list[tuple[str, int]] = []

    async def delete_by_uuids(
        self,
        _executor: Any,
        uuids: list[str],
        *,
        batch_size: int,
    ) -> None:
        self.uuid_calls.append((uuids, batch_size))

    async def delete_by_group_id(
        self,
        _executor: Any,
        group_id: str,
        *,
        batch_size: int,
    ) -> None:
        self.group_calls.append((group_id, batch_size))


class ClearDriver:
    def __init__(self, entity_node_ops: DeleteRecorder) -> None:
        self.entity_node_ops = entity_node_ops
        self.vector_index_reset = False

    async def delete_vector_aoss_indices(self) -> None:
        self.vector_index_reset = True


class TestGraphClearLifecycle:
    @pytest.mark.asyncio
    async def test_global_clear_preserves_generation_ledger_and_batches_nodes(self):
        deletes = DeleteRecorder()
        driver = ClearDriver(deletes)
        operations = NeptuneGraphMaintenanceOperations(cast(NeptuneDriver, driver))
        executor = QueryRecorder(
            [
                ([{'uuid': 'node-1'}, {'uuid': 'node-2'}], None, None),
                ([], None, None),
                ([{'id': 'remaining-1'}, {'id': 'remaining-2'}], None, None),
                ([], None, None),
                ([{'id': 'remaining-3'}], None, None),
                ([], None, None),
                ([], None, None),
            ]
        )

        await operations.clear_data(executor, batch_size=2)

        assert deletes.uuid_calls == [(['node-1', 'node-2'], 2)]
        remaining_queries = [query for query, _ in executor.calls[2:]]
        select_queries = [query for query in remaining_queries if 'RETURN id(n) AS id' in query]
        delete_queries = [query for query in remaining_queries if 'DETACH DELETE n' in query]
        assert len(select_queries) == 3
        assert len(delete_queries) == 2
        assert all('LIMIT $batch_size' in query for query in select_queries)
        assert all('NOT n:GraphitiProjectionVersion' in query for query in select_queries)
        assert all('NOT n:Entity' in query for query in select_queries)
        assert driver.vector_index_reset is False


class GenericGraphOpsRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str] | None, int]] = []

    async def clear_data(
        self,
        _executor: Any,
        group_ids: list[str] | None,
        *,
        batch_size: int,
    ) -> None:
        self.calls.append((group_ids, batch_size))


class GenericNeptuneDriver:
    provider = GraphProvider.NEPTUNE
    graph_operations_interface = None

    def __init__(self) -> None:
        self.entity_node_ops = DeleteRecorder()
        self.entity_edge_ops = DeleteRecorder()
        self.graph_ops = GenericGraphOpsRecorder()
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute_query(self, query: str, **kwargs: Any) -> Any:
        self.calls.append((query, kwargs))
        if 'MATCH (n:Entity)' in query and 'RETURN n.uuid AS uuid' in query:
            return (
                [{'uuid': uuid} for uuid in kwargs['uuids'] if uuid.startswith('node')],
                None,
                None,
            )
        if 'MATCH ()-[e:RELATES_TO]->()' in query and 'RETURN e.uuid AS uuid' in query:
            return (
                [{'uuid': uuid} for uuid in kwargs['uuids'] if uuid.startswith('edge')],
                None,
                None,
            )
        return [], None, None


class TestGenericNeptuneDeleteRouting:
    @pytest.mark.asyncio
    async def test_generic_node_group_delete_forwards_batch_size(self):
        driver = GenericNeptuneDriver()

        await Node.delete_by_group_id(cast(Any, driver), 'group-1', batch_size=7)

        assert driver.graph_ops.calls == [(['group-1'], 7)]

    @pytest.mark.asyncio
    async def test_generic_node_uuid_delete_routes_entities_in_bounded_chunks(self):
        driver = GenericNeptuneDriver()
        uuids = ['node-1', 'episode-1', 'node-2', 'community-1', 'node-3']

        await Node.delete_by_uuids(cast(Any, driver), uuids, batch_size=2)

        assert driver.entity_node_ops.uuid_calls == [
            (['node-1'], 2),
            (['node-2'], 2),
            (['node-3'], 2),
        ]
        query_chunks = [kwargs['uuids'] for _, kwargs in driver.calls]
        assert query_chunks == [
            ['node-1', 'episode-1'],
            ['node-1', 'episode-1'],
            ['node-1', 'episode-1'],
            ['node-2', 'community-1'],
            ['node-2', 'community-1'],
            ['node-2', 'community-1'],
            ['node-3'],
            ['node-3'],
            ['node-3'],
        ]

    @pytest.mark.asyncio
    async def test_generic_edge_uuid_delete_routes_relates_to_in_bounded_chunks(self):
        driver = GenericNeptuneDriver()
        uuids = ['edge-1', 'mention-1', 'edge-2', 'member-1', 'edge-3']

        await Edge.delete_by_uuids(cast(Any, driver), uuids, batch_size=2)

        assert driver.entity_edge_ops.uuid_calls == [
            (['edge-1'], 2),
            (['edge-2'], 2),
            (['edge-3'], 2),
        ]
        assert [kwargs['uuids'] for _, kwargs in driver.calls] == [
            ['edge-1', 'mention-1'],
            ['edge-1', 'mention-1'],
            ['edge-2', 'member-1'],
            ['edge-2', 'member-1'],
            ['edge-3'],
            ['edge-3'],
        ]
