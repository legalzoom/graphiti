from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.driver.neptune.operations.entity_edge_ops import (
    NeptuneEntityEdgeOperations,
)
from graphiti_core.driver.neptune.operations.entity_node_ops import (
    NeptuneEntityNodeOperations,
)
from graphiti_core.driver.neptune.operations.graph_ops import NeptuneGraphMaintenanceOperations
from graphiti_core.driver.neptune.operations.search_ops import NeptuneSearchOperations
from graphiti_core.driver.query_executor import QueryExecutor
from graphiti_core.edges import EntityEdge
from graphiti_core.errors import EdgeNotFoundError, NodeNotFoundError
from graphiti_core.nodes import EntityNode
from graphiti_core.search import search_utils
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance import community_operations


def _empty_executor() -> MagicMock:
    executor = MagicMock(spec=QueryExecutor)
    executor.execute_query = AsyncMock(return_value=([], None, None))
    return executor


def _pending_delete_predicate(alias: str) -> str:
    return f'coalesce({alias}._graphiti_vector_delete_pending, false) = false'


def _assert_edge_and_endpoint_predicates(query: str) -> None:
    assert _pending_delete_predicate('e') in query
    assert _pending_delete_predicate('n') in query
    assert _pending_delete_predicate('m') in query


def _assert_variable_path_predicates(query: str) -> None:
    assert 'MATCH path =' in query
    assert 'all(path_relationship IN relationships(path) WHERE' in query
    assert _pending_delete_predicate('path_relationship') in query
    assert 'all(path_node IN nodes(path) WHERE' in query
    assert _pending_delete_predicate('path_node') in query


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('method_name', 'args', 'missing_error'),
    [
        pytest.param('get_by_uuid', ('node-1',), NodeNotFoundError, id='get-by-uuid'),
        pytest.param('get_by_uuids', (['node-1'],), None, id='get-by-uuids'),
        pytest.param('get_by_group_ids', (['group-1'],), None, id='get-by-group-ids'),
        pytest.param(
            'load_embeddings',
            (SimpleNamespace(uuid='node-1'),),
            NodeNotFoundError,
            id='load-embeddings',
        ),
        pytest.param(
            'load_embeddings_bulk',
            ([SimpleNamespace(uuid='node-1')],),
            None,
            id='load-embeddings-bulk',
        ),
    ],
)
async def test_entity_node_public_reads_exclude_pending_deletes(
    method_name: str,
    args: tuple[object, ...],
    missing_error: type[Exception] | None,
) -> None:
    executor = _empty_executor()
    operation = getattr(NeptuneEntityNodeOperations(), method_name)

    if missing_error is None:
        await operation(executor, *args)
    else:
        with pytest.raises(missing_error):
            await operation(executor, *args)

    executor.execute_query.assert_awaited_once()
    query = executor.execute_query.await_args.args[0]
    assert _pending_delete_predicate('n') in query


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('method_name', 'args', 'missing_error'),
    [
        pytest.param('get_by_uuid', ('edge-1',), EdgeNotFoundError, id='get-by-uuid'),
        pytest.param('get_by_uuids', (['edge-1'],), None, id='get-by-uuids'),
        pytest.param('get_by_group_ids', (['group-1'],), None, id='get-by-group-ids'),
        pytest.param(
            'get_between_nodes',
            ('source-1', 'target-1'),
            None,
            id='get-between-nodes',
        ),
        pytest.param('get_by_node_uuid', ('node-1',), None, id='get-by-node-uuid'),
        pytest.param(
            'load_embeddings',
            (SimpleNamespace(uuid='edge-1'),),
            EdgeNotFoundError,
            id='load-embeddings',
        ),
        pytest.param(
            'load_embeddings_bulk',
            ([SimpleNamespace(uuid='edge-1')],),
            None,
            id='load-embeddings-bulk',
        ),
    ],
)
async def test_entity_edge_public_reads_exclude_pending_edge_and_endpoints(
    method_name: str,
    args: tuple[object, ...],
    missing_error: type[Exception] | None,
) -> None:
    executor = _empty_executor()
    operation = getattr(NeptuneEntityEdgeOperations(), method_name)

    if missing_error is None:
        await operation(executor, *args)
    else:
        with pytest.raises(missing_error):
            await operation(executor, *args)

    executor.execute_query.assert_awaited_once()
    query = executor.execute_query.await_args.args[0]
    _assert_edge_and_endpoint_predicates(query)


@pytest.mark.asyncio
async def test_legacy_entity_node_api_routes_neptune_reads_through_guarded_operations() -> None:
    node = EntityNode(name='node', group_id='group')
    operations = SimpleNamespace(
        get_by_uuid=AsyncMock(return_value=node),
        get_by_uuids=AsyncMock(return_value=[node]),
        get_by_group_ids=AsyncMock(return_value=[node]),
        load_embeddings=AsyncMock(),
        load_embeddings_bulk=AsyncMock(),
    )
    driver = SimpleNamespace(
        provider=GraphProvider.NEPTUNE,
        graph_operations_interface=None,
        entity_node_ops=operations,
        execute_query=AsyncMock(side_effect=AssertionError('legacy query path used')),
    )

    assert await EntityNode.get_by_uuid(driver, node.uuid) is node  # type: ignore[arg-type]
    assert await EntityNode.get_by_uuids(driver, [node.uuid]) == [node]  # type: ignore[arg-type]
    assert await EntityNode.get_by_group_ids(  # type: ignore[arg-type]
        driver,
        ['group'],
        with_embeddings=True,
    ) == [node]
    await node.load_name_embedding(driver)  # type: ignore[arg-type]

    operations.get_by_uuid.assert_awaited_once_with(driver, node.uuid)
    operations.get_by_uuids.assert_awaited_once_with(driver, [node.uuid])
    operations.get_by_group_ids.assert_awaited_once_with(driver, ['group'], None, None)
    operations.load_embeddings_bulk.assert_awaited_once_with(driver, [node])
    operations.load_embeddings.assert_awaited_once_with(driver, node)
    driver.execute_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_entity_edge_api_routes_neptune_reads_through_guarded_operations() -> None:
    edge = EntityEdge(
        name='RELATES_TO',
        fact='source relates to target',
        group_id='group',
        source_node_uuid='source',
        target_node_uuid='target',
        created_at=datetime.now(timezone.utc),
    )
    operations = SimpleNamespace(
        get_by_uuid=AsyncMock(return_value=edge),
        get_by_uuids=AsyncMock(return_value=[edge]),
        get_by_group_ids=AsyncMock(return_value=[edge]),
        get_between_nodes=AsyncMock(return_value=[edge]),
        get_by_node_uuid=AsyncMock(return_value=[edge]),
        load_embeddings=AsyncMock(),
        load_embeddings_bulk=AsyncMock(),
    )
    driver = SimpleNamespace(
        provider=GraphProvider.NEPTUNE,
        graph_operations_interface=None,
        entity_edge_ops=operations,
        execute_query=AsyncMock(side_effect=AssertionError('legacy query path used')),
    )

    assert await EntityEdge.get_by_uuid(driver, edge.uuid) is edge  # type: ignore[arg-type]
    assert await EntityEdge.get_by_uuids(driver, [edge.uuid]) == [edge]  # type: ignore[arg-type]
    assert await EntityEdge.get_by_group_ids(  # type: ignore[arg-type]
        driver,
        ['group'],
        with_embeddings=True,
    ) == [edge]
    assert await EntityEdge.get_between_nodes(  # type: ignore[arg-type]
        driver,
        'source',
        'target',
    ) == [edge]
    assert await EntityEdge.get_by_node_uuid(driver, 'source') == [edge]  # type: ignore[arg-type]
    await edge.load_fact_embedding(driver)  # type: ignore[arg-type]

    operations.get_by_uuid.assert_awaited_once_with(driver, edge.uuid)
    operations.get_by_uuids.assert_awaited_once_with(driver, [edge.uuid])
    operations.get_by_group_ids.assert_awaited_once_with(driver, ['group'], None, None)
    operations.load_embeddings_bulk.assert_awaited_once_with(driver, [edge])
    operations.get_between_nodes.assert_awaited_once_with(driver, 'source', 'target')
    operations.get_by_node_uuid.assert_awaited_once_with(driver, 'source')
    operations.load_embeddings.assert_awaited_once_with(driver, edge)
    driver.execute_query.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('subject', 'method_name', 'args'),
    [
        pytest.param(
            EntityNode(name='node', group_id='group'), 'load_name_embedding', (), id='node-load'
        ),
        pytest.param(EntityNode, 'get_by_uuid', ('node-1',), id='node-get-by-uuid'),
        pytest.param(EntityNode, 'get_by_uuids', (['node-1'],), id='node-get-by-uuids'),
        pytest.param(EntityNode, 'get_by_group_ids', (['group'],), id='node-get-by-group'),
        pytest.param(
            EntityEdge(
                name='RELATES_TO',
                fact='source relates to target',
                group_id='group',
                source_node_uuid='source',
                target_node_uuid='target',
                created_at=datetime.now(timezone.utc),
            ),
            'load_fact_embedding',
            (),
            id='edge-load',
        ),
        pytest.param(EntityEdge, 'get_by_uuid', ('edge-1',), id='edge-get-by-uuid'),
        pytest.param(EntityEdge, 'get_by_uuids', (['edge-1'],), id='edge-get-by-uuids'),
        pytest.param(EntityEdge, 'get_by_group_ids', (['group'],), id='edge-get-by-group'),
        pytest.param(
            EntityEdge,
            'get_between_nodes',
            ('source', 'target'),
            id='edge-get-between-nodes',
        ),
        pytest.param(EntityEdge, 'get_by_node_uuid', ('node-1',), id='edge-get-by-node'),
    ],
)
async def test_legacy_interface_cannot_bypass_projection_read_guards(
    subject: object,
    method_name: str,
    args: tuple[object, ...],
) -> None:
    legacy_interface = MagicMock()
    driver = SimpleNamespace(
        provider=GraphProvider.NEPTUNE,
        vector_projection_enabled=True,
        graph_operations_interface=legacy_interface,
    )

    with pytest.raises(RuntimeError, match='graph_operations_interface'):
        await getattr(subject, method_name)(driver, *args)

    assert legacy_interface.mock_calls == []


@pytest.mark.asyncio
async def test_neptune_community_maintenance_excludes_pending_entities() -> None:
    executor = _empty_executor()
    operations = NeptuneGraphMaintenanceOperations()

    assert await operations.get_community_clusters(executor, ['group']) == []
    cluster_query = executor.execute_query.await_args.args[0]
    assert _pending_delete_predicate('n') in cluster_query

    executor.execute_query.reset_mock()
    await operations.determine_entity_community(
        executor,
        EntityNode(name='node', group_id='group'),
    )
    determine_queries = [call.args[0] for call in executor.execute_query.await_args_list]
    assert len(determine_queries) == 2
    assert _pending_delete_predicate('n') in determine_queries[0]
    assert _pending_delete_predicate('n') in determine_queries[1]
    assert _pending_delete_predicate('m') in determine_queries[1]
    assert _pending_delete_predicate('e') in determine_queries[1]

    executor.execute_query.reset_mock()
    assert (
        await operations.get_mentioned_nodes(
            executor,
            [SimpleNamespace(uuid='episode')],
        )
        == []
    )
    assert _pending_delete_predicate('n') in executor.execute_query.await_args.args[0]

    executor.execute_query.reset_mock()
    assert (
        await operations.get_communities_by_nodes(
            executor,
            [EntityNode(name='node', group_id='group')],
        )
        == []
    )
    assert _pending_delete_predicate('m') in executor.execute_query.await_args.args[0]


@pytest.mark.asyncio
async def test_shared_graphiti_helpers_exclude_pending_neptune_entities() -> None:
    node = EntityNode(name='node', group_id='group')
    node_operations = SimpleNamespace(
        get_by_group_ids=AsyncMock(return_value=[node]),
        get_by_uuids=AsyncMock(return_value=[node]),
    )
    driver = SimpleNamespace(
        provider=GraphProvider.NEPTUNE,
        graph_operations_interface=None,
        entity_node_ops=node_operations,
        execute_query=AsyncMock(return_value=([], None, None)),
    )

    assert (
        await search_utils.get_mentioned_nodes(  # type: ignore[arg-type]
            driver,
            [SimpleNamespace(uuid='episode')],
        )
        == []
    )
    assert _pending_delete_predicate('n') in driver.execute_query.await_args.args[0]

    driver.execute_query.reset_mock()
    assert await search_utils.get_communities_by_nodes(driver, [node]) == []  # type: ignore[arg-type]
    assert _pending_delete_predicate('m') in driver.execute_query.await_args.args[0]

    driver.execute_query.reset_mock()
    assert await community_operations.get_community_clusters(  # type: ignore[arg-type]
        driver,
        ['group'],
    ) == [[node]]
    neighbor_query = driver.execute_query.await_args.args[0]
    _assert_edge_and_endpoint_predicates(neighbor_query)

    driver.execute_query.reset_mock()
    assert await community_operations.determine_entity_community(  # type: ignore[arg-type]
        driver,
        node,
    ) == (None, False)
    determine_queries = [call.args[0] for call in driver.execute_query.await_args_list]
    assert _pending_delete_predicate('n') in determine_queries[0]
    _assert_edge_and_endpoint_predicates(determine_queries[1])


def _shared_neptune_driver() -> MagicMock:
    driver = MagicMock(spec=GraphDriver)
    driver.provider = GraphProvider.NEPTUNE
    driver.search_interface = None
    driver.execute_query = AsyncMock(return_value=([], None, None))
    return driver


@pytest.mark.asyncio
@pytest.mark.parametrize('implementation', ['shared', 'unbound'])
async def test_neptune_node_bfs_excludes_pending_results_and_variable_paths(
    implementation: str,
) -> None:
    if implementation == 'shared':
        executor = _shared_neptune_driver()
        result = await search_utils.node_bfs_search(
            executor,
            ['origin-1'],
            SearchFilters(),
            3,
        )
    else:
        executor = _empty_executor()
        result = await NeptuneSearchOperations().node_bfs_search(
            executor,
            ['origin-1'],
            SearchFilters(),
            3,
        )

    assert result == []
    executor.execute_query.assert_awaited_once()
    query = executor.execute_query.await_args.args[0]
    assert _pending_delete_predicate('n') in query
    assert _pending_delete_predicate('origin') in query
    _assert_variable_path_predicates(query)


@pytest.mark.asyncio
@pytest.mark.parametrize('implementation', ['shared', 'unbound'])
async def test_neptune_edge_bfs_excludes_pending_results_endpoints_and_variable_paths(
    implementation: str,
) -> None:
    if implementation == 'shared':
        executor = _shared_neptune_driver()
        result = await search_utils.edge_bfs_search(
            executor,
            ['origin-1'],
            3,
            SearchFilters(),
        )
    else:
        executor = _empty_executor()
        result = await NeptuneSearchOperations().edge_bfs_search(
            executor,
            ['origin-1'],
            3,
            SearchFilters(),
        )

    assert result == []
    executor.execute_query.assert_awaited_once()
    query = executor.execute_query.await_args.args[0]
    _assert_edge_and_endpoint_predicates(query)
    _assert_variable_path_predicates(query)
