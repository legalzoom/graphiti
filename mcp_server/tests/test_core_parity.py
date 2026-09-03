#!/usr/bin/env python3
"""Unit tests for the MCP <-> graphiti-core parity wiring.

These tests exercise the pure helper functions and the queue-service argument
threading without requiring a live database or LLM. They run as part of the
default (non-integration) suite.
"""

import asyncio
import inspect
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from graphiti_core import Graphiti
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.falkordb.operations.has_episode_edge_ops import (
    FalkorHasEpisodeEdgeOperations,
)
from graphiti_core.driver.falkordb.operations.next_episode_edge_ops import (
    FalkorNextEpisodeEdgeOperations,
)
from graphiti_core.driver.kuzu.operations.has_episode_edge_ops import KuzuHasEpisodeEdgeOperations
from graphiti_core.driver.kuzu.operations.next_episode_edge_ops import KuzuNextEpisodeEdgeOperations
from graphiti_core.driver.neo4j.operations.has_episode_edge_ops import Neo4jHasEpisodeEdgeOperations
from graphiti_core.driver.neo4j.operations.next_episode_edge_ops import (
    Neo4jNextEpisodeEdgeOperations,
)
from graphiti_core.driver.neptune.operations.has_episode_edge_ops import (
    NeptuneHasEpisodeEdgeOperations,
)
from graphiti_core.driver.neptune.operations.next_episode_edge_ops import (
    NeptuneNextEpisodeEdgeOperations,
)
from graphiti_core.edges import EntityEdge, HasEpisodeEdge, NextEpisodeEdge
from graphiti_core.errors import NodeGroupMismatchError
from graphiti_core.models.edges.edge_db_queries import (
    HAS_EPISODE_EDGE_SAVE,
    NEXT_EPISODE_EDGE_SAVE,
)
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.search.search_filters import ComparisonOperator, SearchFilters

# Add the src directory to the path (mirrors the other unit tests)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config.schema import (  # noqa: E402
    EdgeTypeConfig,
    EdgeTypeMapEntry,
    EntityTypeConfig,
)
from models.edge_types import EDGE_TYPES  # noqa: E402
from models.entity_types import ENTITY_TYPES  # noqa: E402
from services.queue_service import QueueService  # noqa: E402
from utils.type_config import (  # noqa: E402
    build_edge_type_map,
    build_edge_types,
    build_entity_types,
    build_fact_search_filters,
    coerce_group_ids,
    parse_reference_time,
)


class TestParseReferenceTime:
    def test_none_returns_none(self):
        assert parse_reference_time(None) is None

    def test_naive_value_is_coerced_to_utc(self):
        result = parse_reference_time('2025-01-15T10:30:00')
        assert result is not None
        assert result.tzinfo is timezone.utc
        assert result == datetime(2025, 1, 15, 10, 30, tzinfo=timezone.utc)

    def test_trailing_z_is_treated_as_utc(self):
        result = parse_reference_time('2025-01-15T10:30:00Z')
        assert result == datetime(2025, 1, 15, 10, 30, tzinfo=timezone.utc)

    def test_explicit_offset_is_converted_to_utc(self):
        result = parse_reference_time('2025-01-15T10:30:00+02:00')
        assert result is not None
        assert result.tzinfo is timezone.utc
        assert result == datetime(2025, 1, 15, 8, 30, tzinfo=timezone.utc)

    def test_invalid_string_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_reference_time('not-a-timestamp')


class TestBuildEntityTypes:
    def test_none_when_empty(self):
        assert build_entity_types(None) is None
        assert build_entity_types([]) is None

    def test_registered_model_is_preferred(self):
        cfg = [EntityTypeConfig(name='Preference', description='ignored description')]
        result = build_entity_types(cfg)
        assert result is not None
        # The rich registered model, not a doc-only stub, must be used.
        assert result['Preference'] is ENTITY_TYPES['Preference']

    def test_unknown_name_falls_back_to_doc_only_model(self):
        cfg = [EntityTypeConfig(name='Widget', description='A made-up type')]
        result = build_entity_types(cfg)
        assert result is not None
        model = result['Widget']
        assert model.__name__ == 'Widget'
        assert model.__doc__ == 'A made-up type'
        assert model not in ENTITY_TYPES.values()


class TestBuildEdgeTypes:
    def test_none_when_empty(self):
        assert build_edge_types(None) is None
        assert build_edge_types([]) is None

    def test_registered_model_is_preferred(self):
        cfg = [EdgeTypeConfig(name='WorksFor', description='ignored')]
        result = build_edge_types(cfg)
        assert result is not None
        assert result['WorksFor'] is EDGE_TYPES['WorksFor']

    def test_unknown_name_falls_back_to_doc_only_model(self):
        cfg = [EdgeTypeConfig(name='CustomEdge', description='custom relation')]
        result = build_edge_types(cfg)
        assert result is not None
        assert result['CustomEdge'].__doc__ == 'custom relation'


class TestBuildEdgeTypeMap:
    def test_none_when_empty(self):
        assert build_edge_type_map(None) is None
        assert build_edge_type_map([]) is None

    def test_entries_become_tuple_keyed_map(self):
        entries = [
            EdgeTypeMapEntry(source='Person', target='Organization', edge_types=['WorksFor']),
            EdgeTypeMapEntry(edge_types=['RelatesTo']),  # defaults to Entity/Entity
        ]
        result = build_edge_type_map(entries)
        assert result == {
            ('Person', 'Organization'): ['WorksFor'],
            ('Entity', 'Entity'): ['RelatesTo'],
        }


class TestBuildFactSearchFilters:
    def test_none_when_no_criteria(self):
        assert build_fact_search_filters() is None

    def test_edge_types_only(self):
        sf = build_fact_search_filters(edge_types=['WorksFor'])
        assert isinstance(sf, SearchFilters)
        assert sf.edge_types == ['WorksFor']
        assert sf.valid_at is None
        assert sf.invalid_at is None

    def test_valid_at_range_is_and_group(self):
        sf = build_fact_search_filters(
            valid_at_after='2025-01-01T00:00:00Z',
            valid_at_before='2025-02-01T00:00:00Z',
        )
        assert isinstance(sf, SearchFilters)
        assert sf.valid_at is not None
        # One OR group containing two AND-ed conditions (>= and <=).
        assert len(sf.valid_at) == 1
        operators = {cond.comparison_operator for cond in sf.valid_at[0]}
        assert operators == {
            ComparisonOperator.greater_than_equal,
            ComparisonOperator.less_than_equal,
        }

    def test_invalid_date_raises_value_error(self):
        with pytest.raises(ValueError):
            build_fact_search_filters(valid_at_after='garbage')


class TestQueueServiceThreading:
    """The queue service must forward every parity param to Graphiti.add_episode."""

    @pytest.mark.asyncio
    async def test_add_episode_forwards_all_params(self, monkeypatch):
        client = AsyncMock(spec=Graphiti)
        scoped_client = AsyncMock(spec=Graphiti)
        scope_client = AsyncMock(return_value=scoped_client)
        monkeypatch.setattr('services.queue_service.graphiti_for_group', scope_client)
        service = QueueService()
        await service.initialize(client)

        ref_time = datetime(2024, 6, 1, tzinfo=timezone.utc)
        edge_types = {'WorksFor': EDGE_TYPES['WorksFor']}
        edge_type_map = {('Entity', 'Entity'): ['WorksFor']}

        await service.add_episode(
            group_id='g1',
            name='ep',
            content='body',
            source_description='desc',
            episode_type='text',
            entity_types={'Preference': ENTITY_TYPES['Preference']},
            uuid='ep-uuid',
            reference_time=ref_time,
            edge_types=edge_types,
            edge_type_map=edge_type_map,
            excluded_entity_types=['Object'],
            previous_episode_uuids=['prev-uuid'],
            custom_extraction_instructions='extra',
            update_communities=True,
            saga='my-saga',
            saga_previous_episode_uuid='saga-prev',
        )

        # The worker runs the queued coroutine in the background; wait for it.
        await service._episode_queues['g1'].join()

        scope_client.assert_awaited_once_with(client, 'g1')
        scoped_client.add_episode.assert_awaited_once()
        client.add_episode.assert_not_awaited()
        kwargs = scoped_client.add_episode.await_args.kwargs
        assert kwargs['reference_time'] == ref_time
        assert kwargs['edge_types'] == edge_types
        assert kwargs['edge_type_map'] == edge_type_map
        assert kwargs['excluded_entity_types'] == ['Object']
        assert kwargs['previous_episode_uuids'] == ['prev-uuid']
        assert kwargs['custom_extraction_instructions'] == 'extra'
        assert kwargs['update_communities'] is True
        assert kwargs['saga'] == 'my-saga'
        assert kwargs['saga_previous_episode_uuid'] == 'saga-prev'
        assert kwargs['uuid'] == 'ep-uuid'

        worker = service._worker_tasks['g1']
        worker.cancel()
        await worker

    @pytest.mark.asyncio
    async def test_add_episode_defaults_reference_time_to_now(self, monkeypatch):
        client = AsyncMock(spec=Graphiti)
        scoped_client = AsyncMock(spec=Graphiti)
        monkeypatch.setattr(
            'services.queue_service.graphiti_for_group', AsyncMock(return_value=scoped_client)
        )
        service = QueueService()
        await service.initialize(client)

        before = datetime.now(timezone.utc)
        await service.add_episode(
            group_id='g2',
            name='ep',
            content='body',
            source_description='desc',
            episode_type='text',
            entity_types=None,
            uuid=None,
        )
        await service._episode_queues['g2'].join()
        after = datetime.now(timezone.utc)

        kwargs = scoped_client.add_episode.await_args.kwargs
        assert before <= kwargs['reference_time'] <= after

        worker = service._worker_tasks['g2']
        worker.cancel()
        await worker

    @pytest.mark.asyncio
    async def test_falkor_logical_default_does_not_trigger_an_inner_core_clone(self, monkeypatch):
        client = AsyncMock(spec=Graphiti)
        scoped_client = SimpleNamespace(
            driver=SimpleNamespace(default_group_id='_'),
            add_episode=AsyncMock(),
        )
        scope_client = AsyncMock(return_value=scoped_client)
        monkeypatch.setattr('services.queue_service.graphiti_for_group', scope_client)
        service = QueueService()
        await service.initialize(client)

        for episode_uuid in ('ep-1', 'ep-2'):
            await service.add_episode(
                group_id='_',
                name=episode_uuid,
                content='body',
                source_description='desc',
                episode_type='text',
                entity_types=None,
                uuid=episode_uuid,
            )
        await service._episode_queues['_'].join()

        assert [call.kwargs['group_id'] for call in scoped_client.add_episode.await_args_list] == [
            None,
            None,
        ]
        assert scope_client.await_count == 2

        worker = service._worker_tasks['_']
        worker.cancel()
        await worker

    @pytest.mark.asyncio
    async def test_concurrent_same_group_enqueues_use_one_sequential_worker(self):
        service = QueueService()
        active = 0
        max_active = 0
        processed = 0

        async def process_episode() -> None:
            nonlocal active, max_active, processed
            active += 1
            max_active = max(max_active, active)
            try:
                await asyncio.sleep(0.01)
                processed += 1
            finally:
                active -= 1

        await asyncio.gather(
            *(service.add_episode_task('same-group', process_episode) for _ in range(8))
        )
        await service._episode_queues['same-group'].join()

        assert processed == 8
        assert max_active == 1
        assert service.is_worker_running('same-group')

        worker = service._worker_tasks['same-group']
        worker.cancel()
        await worker
        assert not service.is_worker_running('same-group')

    @pytest.mark.asyncio
    async def test_shutdown_stops_admission_and_drains_before_workers_exit(self):
        service = QueueService()
        started = asyncio.Event()
        release = asyncio.Event()
        completed: list[str] = []

        async def process_episode() -> None:
            started.set()
            await release.wait()
            completed.append('episode')

        await service.add_episode_task('group', process_episode)
        await started.wait()

        shutdown = asyncio.create_task(service.shutdown(timeout_seconds=1))
        await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match='not accepting new work'):
            await service.add_episode_task('group', process_episode)

        assert not shutdown.done()
        release.set()
        assert await shutdown is True

        assert completed == ['episode']
        assert not service.is_worker_running('group')
        assert service.get_queue_size('group') == 0

    @pytest.mark.asyncio
    async def test_shutdown_is_bounded_when_worker_suppresses_cancellation(self, caplog):
        service = QueueService()
        started = asyncio.Event()
        cancellation_seen = asyncio.Event()
        release = asyncio.Event()

        async def cancellation_suppressing_job() -> None:
            started.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancellation_seen.set()

        await service.add_episode_task('stuck-group', cancellation_suppressing_job)
        await started.wait()
        worker = service._worker_tasks['stuck-group']
        caplog.set_level('CRITICAL', logger='services.queue_service')

        all_workers_stopped = await asyncio.wait_for(
            service.shutdown(timeout_seconds=0.01), timeout=0.25
        )

        assert all_workers_stopped is False
        assert cancellation_seen.is_set()
        assert not worker.done()
        assert 'detaching them and leaving their shared graph client open' in caplog.text
        assert not service.is_worker_running('stuck-group')
        assert service.get_queue_size('stuck-group') == 0

        # Let the deliberately non-cooperative job finish, then stop the detached worker at its
        # next queue wait so this test does not leak a task into the event loop.
        release.set()
        await asyncio.sleep(0)
        worker.cancel()
        await asyncio.wait_for(worker, timeout=1)


class TestEpisodeContextGroupIntegrity:
    @staticmethod
    def graphiti_stub():
        driver = SimpleNamespace(provider=GraphProvider.NEO4J, _database='team-a')
        graphiti: Any = object.__new__(Graphiti)
        graphiti.driver = driver
        graphiti.clients = SimpleNamespace(driver=driver)
        graphiti.tracer = MagicMock()
        return graphiti

    @pytest.mark.asyncio
    async def test_core_rejects_cross_group_explicit_context(self, monkeypatch):
        graphiti = self.graphiti_stub()
        get_previous = AsyncMock(return_value=[SimpleNamespace(group_id='team-b')])
        monkeypatch.setattr(EpisodicNode, 'get_by_uuids', get_previous)

        with pytest.raises(NodeGroupMismatchError):
            await graphiti.add_episode(
                name='episode',
                episode_body='body',
                source_description='test',
                reference_time=datetime.now(timezone.utc),
                source=EpisodeType.text,
                group_id='team-a',
                previous_episode_uuids=['cross-group'],
            )

        get_previous.assert_awaited_once_with(graphiti.driver, ['cross-group'])

    @pytest.mark.asyncio
    async def test_core_rejects_cross_group_saga_predecessor(self, monkeypatch):
        graphiti = self.graphiti_stub()
        monkeypatch.setattr(EpisodicNode, 'get_by_uuids', AsyncMock(return_value=[]))
        get_saga_previous = AsyncMock(return_value=SimpleNamespace(group_id='team-b'))
        monkeypatch.setattr(EpisodicNode, 'get_by_uuid', get_saga_previous)

        with pytest.raises(NodeGroupMismatchError):
            await graphiti.add_episode(
                name='episode',
                episode_body='body',
                source_description='test',
                reference_time=datetime.now(timezone.utc),
                source=EpisodeType.text,
                group_id='team-a',
                previous_episode_uuids=[],
                saga='saga',
                saga_previous_episode_uuid='cross-group',
            )

        get_saga_previous.assert_awaited_once_with(graphiti.driver, 'cross-group')

    def test_saga_edge_writes_match_endpoints_by_uuid_and_group(self):
        assert 'MATCH (saga:Saga {uuid: $saga_uuid, group_id: $group_id})' in HAS_EPISODE_EDGE_SAVE
        assert (
            'MATCH (episode:Episodic {uuid: $episode_uuid, group_id: $group_id})'
            in HAS_EPISODE_EDGE_SAVE
        )
        assert (
            'MATCH (source_episode:Episodic '
            '{uuid: $source_episode_uuid, group_id: $group_id})' in NEXT_EPISODE_EDGE_SAVE
        )
        assert (
            'MATCH (target_episode:Episodic '
            '{uuid: $target_episode_uuid, group_id: $group_id})' in NEXT_EPISODE_EDGE_SAVE
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('edge_type', [HasEpisodeEdge, NextEpisodeEdge])
    async def test_legacy_saga_edge_writes_fail_closed_on_missing_endpoints(self, edge_type):
        driver = SimpleNamespace(
            graph_operations_interface=None,
            provider=GraphProvider.NEO4J,
            execute_query=AsyncMock(return_value=([], [], None)),
        )
        edge = edge_type(
            source_node_uuid='source',
            target_node_uuid='target',
            group_id='team-a',
            created_at=datetime.now(timezone.utc),
        )

        with pytest.raises(NodeGroupMismatchError):
            await edge.save(driver)

        driver.execute_query.return_value = ([{'uuid': edge.uuid}], [], None)
        await edge.save(driver)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('operation', 'edge_type'),
        [
            (Neo4jHasEpisodeEdgeOperations(), HasEpisodeEdge),
            (Neo4jNextEpisodeEdgeOperations(), NextEpisodeEdge),
            (NeptuneHasEpisodeEdgeOperations(), HasEpisodeEdge),
            (NeptuneNextEpisodeEdgeOperations(), NextEpisodeEdge),
            (FalkorHasEpisodeEdgeOperations(), HasEpisodeEdge),
            (FalkorNextEpisodeEdgeOperations(), NextEpisodeEdge),
            (KuzuHasEpisodeEdgeOperations(), HasEpisodeEdge),
            (KuzuNextEpisodeEdgeOperations(), NextEpisodeEdge),
        ],
    )
    async def test_provider_saga_edge_writes_fail_closed_on_missing_endpoints(
        self, operation, edge_type
    ):
        executor = SimpleNamespace(execute_query=AsyncMock(return_value=([], [], None)))
        edge = edge_type(
            source_node_uuid='source',
            target_node_uuid='target',
            group_id='team-a',
            created_at=datetime.now(timezone.utc),
        )

        with pytest.raises(NodeGroupMismatchError):
            await operation.save(executor, edge)

        executor.execute_query.return_value = ([{'uuid': edge.uuid}], [], None)
        await operation.save(executor, edge)


class TestCoreSignatureCompatibility:
    """Guard against drift between the params we send and graphiti-core's API."""

    def test_queue_service_kwargs_are_accepted_by_add_episode(self):
        params = set(inspect.signature(Graphiti.add_episode).parameters)
        sent = {
            'name',
            'episode_body',
            'source_description',
            'source',
            'group_id',
            'reference_time',
            'entity_types',
            'edge_types',
            'edge_type_map',
            'excluded_entity_types',
            'previous_episode_uuids',
            'custom_extraction_instructions',
            'update_communities',
            'saga',
            'saga_previous_episode_uuid',
            'uuid',
        }
        assert sent <= params

    def test_core_exposes_parity_methods(self):
        # The new tools depend on these methods; guard against being pointed at a
        # graphiti-core too old to support them (e.g. pre-0.29 lacks summarize_saga).
        for method in (
            'remove_episode',
            'summarize_saga',
            'build_communities',
            'add_triplet',
            'get_nodes_and_edges_by_episode',
        ):
            assert hasattr(Graphiti, method), f'graphiti-core is missing {method}'

    def test_triplet_objects_construct(self):
        """The shapes add_triplet builds must satisfy EntityNode/EntityEdge."""
        now = datetime.now(timezone.utc)
        source = EntityNode(uuid='s', name='Alice', group_id='g', created_at=now)
        target = EntityNode(uuid='t', name='Acme', group_id='g', created_at=now)
        edge = EntityEdge(
            name='WORKS_FOR',
            fact='Alice works for Acme',
            group_id='g',
            source_node_uuid=source.uuid,
            target_node_uuid=target.uuid,
            created_at=now,
        )
        assert edge.source_node_uuid == 's'
        assert edge.target_node_uuid == 't'


class TestEntityTypeRegistration:
    """Configured entity types must be registerable with graphiti-core."""

    def test_configured_entity_types_avoid_reserved_field_names(self):
        # graphiti-core rejects custom entity-type fields that collide with
        # EntityNode's own fields (e.g. 'name'); such a clash silently fails every
        # episode ingest. Guard the registered models against reintroducing one.
        from models.entity_types import ENTITY_TYPES

        reserved = set(EntityNode.model_fields.keys())
        for type_name, model in ENTITY_TYPES.items():
            clashes = set(model.model_fields.keys()) & reserved
            assert not clashes, (
                f'entity type {type_name} uses reserved EntityNode field(s): {sorted(clashes)}'
            )


class TestCoerceGroupIds:
    """Read tools accept a scalar group_id or a list (graphiti-core wants a list)."""

    def test_scalar_string_becomes_one_element_list(self):
        assert coerce_group_ids('g1') == ['g1']

    def test_list_passes_through(self):
        assert coerce_group_ids(['g1', 'g2']) == ['g1', 'g2']

    def test_duplicate_groups_are_deduplicated_in_order(self):
        assert coerce_group_ids(['g2', 'g1', 'g2']) == ['g2', 'g1']

    def test_none_passes_through(self):
        assert coerce_group_ids(None) is None

    def test_blank_string_is_treated_as_omitted(self):
        # A blank scalar falls back to the default group (not group '').
        assert coerce_group_ids('') is None
