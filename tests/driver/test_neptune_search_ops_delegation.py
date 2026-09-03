from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.driver.neptune.operations import search_ops as neptune_search_ops
from graphiti_core.driver.neptune.operations.search_ops import NeptuneSearchOperations
from graphiti_core.driver.query_executor import QueryExecutor
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from graphiti_core.search import search_utils
from graphiti_core.search.search_filters import SearchFilters


def aoss_hit(uuid: str) -> dict:
    return {
        'hits': {
            'total': {'value': 1},
            'hits': [{'_source': {'uuid': uuid}, '_score': 0.9}],
        }
    }


def shared_neptune_driver() -> MagicMock:
    driver = MagicMock()
    driver.provider = GraphProvider.NEPTUNE
    driver.search_interface = None
    driver.fulltext_syntax = ''
    driver.vector_search_enabled = True
    driver.run_aoss_query = AsyncMock(return_value=aoss_hit('candidate-1'))
    driver.run_aoss_knn_query = AsyncMock(return_value=[])
    driver.execute_query = AsyncMock(return_value=([], None, None))
    return driver


def assert_edge_and_endpoint_delete_filters(query: str, edge_alias: str) -> None:
    assert f'coalesce({edge_alias}._graphiti_vector_delete_pending, false) = false' in query
    assert 'coalesce(n._graphiti_vector_delete_pending, false) = false' in query
    assert 'coalesce(m._graphiti_vector_delete_pending, false) = false' in query


@pytest.mark.asyncio
async def test_node_similarity_search_delegates_all_arguments_and_result(monkeypatch):
    driver = MagicMock(spec=GraphDriver)
    executor = MagicMock(spec=QueryExecutor)
    search_filter = SearchFilters(node_labels=['Person'])
    expected = [MagicMock(spec=EntityNode)]
    delegate = AsyncMock(return_value=expected)
    monkeypatch.setattr(neptune_search_ops, 'shared_node_similarity_search', delegate)

    result = await NeptuneSearchOperations(driver=driver).node_similarity_search(
        executor,
        [0.25, 0.75],
        search_filter,
        group_ids=['group-1'],
        limit=7,
        min_score=0.82,
    )

    assert result is expected
    delegate.assert_awaited_once_with(
        driver,
        [0.25, 0.75],
        search_filter,
        ['group-1'],
        7,
        0.82,
    )


@pytest.mark.asyncio
async def test_edge_similarity_search_delegates_all_arguments_and_result(monkeypatch):
    driver = MagicMock(spec=GraphDriver)
    executor = MagicMock(spec=QueryExecutor)
    search_filter = SearchFilters(edge_types=['WORKS_WITH'])
    expected = [MagicMock(spec=EntityEdge)]
    delegate = AsyncMock(return_value=expected)
    monkeypatch.setattr(neptune_search_ops, 'shared_edge_similarity_search', delegate)

    result = await NeptuneSearchOperations(driver=driver).edge_similarity_search(
        executor,
        [0.6, 0.4],
        'source-node',
        'target-node',
        search_filter,
        group_ids=['group-2'],
        limit=11,
        min_score=0.71,
    )

    assert result is expected
    delegate.assert_awaited_once_with(
        driver,
        [0.6, 0.4],
        'source-node',
        'target-node',
        search_filter,
        ['group-2'],
        11,
        0.71,
    )


@pytest.mark.asyncio
async def test_bound_fulltext_hydration_excludes_pending_nodes_edges_and_endpoints():
    driver = shared_neptune_driver()
    executor = MagicMock(spec=QueryExecutor)
    executor.execute_query = AsyncMock(return_value=([], None, None))
    operations = NeptuneSearchOperations(driver=driver)

    assert await operations.node_fulltext_search(executor, 'query', SearchFilters()) == []
    assert await operations.edge_fulltext_search(executor, 'query', SearchFilters()) == []

    node_query = executor.execute_query.await_args_list[0].args[0]
    edge_query = executor.execute_query.await_args_list[1].args[0]
    assert 'coalesce(n._graphiti_vector_delete_pending, false) = false' in node_query
    assert_edge_and_endpoint_delete_filters(edge_query, 'e')


@pytest.mark.asyncio
async def test_shared_fulltext_hydration_excludes_pending_nodes_edges_and_endpoints():
    driver = shared_neptune_driver()

    assert await search_utils.node_fulltext_search(driver, 'query', SearchFilters()) == []
    assert await search_utils.edge_fulltext_search(driver, 'query', SearchFilters()) == []

    node_query = driver.execute_query.await_args_list[0].args[0]
    edge_query = driver.execute_query.await_args_list[1].args[0]
    assert 'coalesce(n._graphiti_vector_delete_pending, false) = false' in node_query
    assert_edge_and_endpoint_delete_filters(edge_query, 'e')


@pytest.mark.asyncio
async def test_shared_node_vector_scan_and_race_safe_hydration_exclude_pending_nodes():
    driver = shared_neptune_driver()
    driver.vector_search_enabled = False
    driver.execute_query = AsyncMock(
        side_effect=[
            ([{'id': 7, 'embedding': '1.0,0.0'}], None, None),
            ([], None, None),
        ]
    )

    result = await search_utils.node_similarity_search(
        driver,
        [1.0, 0.0],
        SearchFilters(),
    )

    assert result == []
    driver.run_aoss_knn_query.assert_not_awaited()
    scan_query = driver.execute_query.await_args_list[0].args[0]
    hydration_query = driver.execute_query.await_args_list[1].args[0]
    assert 'coalesce(n._graphiti_vector_delete_pending, false) = false' in scan_query
    assert 'coalesce(n._graphiti_vector_delete_pending, false) = false' in hydration_query


@pytest.mark.asyncio
async def test_shared_edge_vector_scan_and_race_safe_hydration_exclude_pending_graph():
    driver = shared_neptune_driver()
    driver.vector_search_enabled = False
    driver.execute_query = AsyncMock(
        side_effect=[
            ([{'id': 8, 'embedding': '1.0,0.0'}], None, None),
            ([], None, None),
        ]
    )

    result = await search_utils.edge_similarity_search(
        driver,
        [1.0, 0.0],
        None,
        None,
        SearchFilters(),
    )

    assert result == []
    driver.run_aoss_knn_query.assert_not_awaited()
    scan_query = driver.execute_query.await_args_list[0].args[0]
    hydration_query = driver.execute_query.await_args_list[1].args[0]
    assert_edge_and_endpoint_delete_filters(scan_query, 'e')
    assert_edge_and_endpoint_delete_filters(hydration_query, 'e')
    assert 'e.reference_time AS reference_time' in hydration_query


@pytest.mark.asyncio
async def test_shared_node_knn_hydration_excludes_pending_candidates():
    driver = shared_neptune_driver()
    driver.run_aoss_knn_query = AsyncMock(return_value=[{'id': 'node-1', 'score': 0.9}])

    result = await search_utils.node_similarity_search(
        driver,
        [1.0, 0.0],
        SearchFilters(),
    )

    assert result == []
    hydration_query = driver.execute_query.await_args.args[0]
    assert 'n._graphiti_vector_sync_pending IS NULL' in hydration_query
    assert 'coalesce(n._graphiti_vector_delete_pending, false) = false' in hydration_query


@pytest.mark.asyncio
async def test_shared_edge_knn_hydration_excludes_pending_edges_and_endpoints():
    driver = shared_neptune_driver()
    driver.run_aoss_knn_query = AsyncMock(return_value=[{'id': 'edge-1', 'score': 0.9}])

    result = await search_utils.edge_similarity_search(
        driver,
        [1.0, 0.0],
        None,
        None,
        SearchFilters(),
    )

    assert result == []
    hydration_query = driver.execute_query.await_args.args[0]
    assert 'e._graphiti_vector_sync_pending IS NULL' in hydration_query
    assert_edge_and_endpoint_delete_filters(hydration_query, 'e')


@pytest.mark.asyncio
async def test_unbound_node_similarity_preserves_query_executor_scan(monkeypatch):
    executor = MagicMock(spec=QueryExecutor)
    executor.execute_query = AsyncMock(
        side_effect=[
            ([{'id': 7, 'embedding': '1.0,0.0'}], None, None),
            ([], None, None),
        ]
    )
    search_filter = SearchFilters()
    delegate = AsyncMock(return_value=[])
    monkeypatch.setattr(neptune_search_ops, 'shared_node_similarity_search', delegate)

    result = await NeptuneSearchOperations().node_similarity_search(
        executor,
        [1.0, 0.0],
        search_filter,
    )

    assert result == []
    delegate.assert_not_awaited()
    assert executor.execute_query.await_count == 2
    scan_query = executor.execute_query.await_args_list[0].args[0]
    hydration_query = executor.execute_query.await_args_list[1].args[0]
    assert 'n.name_embedding AS embedding' in scan_query
    assert 'coalesce(n._graphiti_vector_delete_pending, false) = false' in scan_query
    assert 'coalesce(n._graphiti_vector_delete_pending, false) = false' in hydration_query


@pytest.mark.asyncio
async def test_unbound_edge_similarity_preserves_query_executor_scan(monkeypatch):
    executor = MagicMock(spec=QueryExecutor)
    executor.execute_query = AsyncMock(
        side_effect=[
            ([{'id': 8, 'embedding': '1.0,0.0'}], None, None),
            ([], None, None),
        ]
    )
    search_filter = SearchFilters()
    delegate = AsyncMock(return_value=[])
    monkeypatch.setattr(neptune_search_ops, 'shared_edge_similarity_search', delegate)

    result = await NeptuneSearchOperations().edge_similarity_search(
        executor,
        [1.0, 0.0],
        'source-node',
        'target-node',
        search_filter,
    )

    assert result == []
    delegate.assert_not_awaited()
    assert executor.execute_query.await_count == 2
    scan_query = executor.execute_query.await_args_list[0].args[0]
    hydration_query = executor.execute_query.await_args_list[1].args[0]
    kwargs = executor.execute_query.await_args_list[0].kwargs
    assert 'e.fact_embedding AS embedding' in scan_query
    assert kwargs == {'source_uuid': 'source-node', 'target_uuid': 'target-node'}
    assert_edge_and_endpoint_delete_filters(scan_query, 'e')
    assert_edge_and_endpoint_delete_filters(hydration_query, 'e')
    assert 'e.reference_time AS reference_time' in hydration_query
