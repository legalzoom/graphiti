"""Synthetic protocol records guard full-text limits without external services."""

from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from graphiti_core.driver.neptune.operations.search_ops import NeptuneSearchOperations
from graphiti_core.driver.neptune_driver import NeptuneDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.search.search import edge_search
from graphiti_core.search.search_config import EdgeSearchConfig, EdgeSearchMethod
from graphiti_core.search.search_filters import SearchFilters


@pytest.mark.asyncio
@pytest.mark.parametrize('result_limit', [1, 17])
@pytest.mark.parametrize('bound_operations', [False, True])
async def test_fulltext_limit_reaches_request_and_hydration(result_limit, bound_operations):
    candidates = [
        EntityEdge(
            source_node_uuid='source',
            target_node_uuid='target',
            name='RELATES_TO',
            fact=f'Synthetic assertion {i}',
            group_id='test-group',
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        for i in range(2 * result_limit)
    ]
    records = {edge.uuid: edge.model_dump(mode='json') for edge in candidates}
    driver = object.__new__(NeptuneDriver)
    driver.aoss_client = Mock()
    driver.client = Mock()
    driver.search_interface = None

    def search(*, body, index):
        assert index == 'edge_name_and_fact'
        selected = candidates[: body['size']]
        return {
            'hits': {
                'total': {'value': len(candidates)},
                'hits': [{'_source': {'uuid': edge.uuid}, '_score': 1.0} for edge in selected],
            }
        }

    def query(cypher, *, params):
        assert 'e.group_id IN $group_ids' in cypher
        assert params['group_ids'] == ['test-group']
        assert params['limit'] == 2 * result_limit
        return [records[item['id']] for item in params['ids']][: params['limit']]

    driver.aoss_client.search.side_effect = search
    driver.client.query.side_effect = query

    if bound_operations:
        result = await NeptuneSearchOperations(driver).edge_fulltext_search(
            driver,
            'synthetic query',
            SearchFilters(),
            ['test-group'],
            2 * result_limit,
        )
        assert len(result) == 2 * result_limit
    else:
        result, _ = await edge_search(
            driver=driver,
            cross_encoder=Mock(),
            query='synthetic query',
            query_vector=[],
            group_ids=['test-group'],
            config=EdgeSearchConfig(search_methods=[EdgeSearchMethod.bm25]),
            search_filter=SearchFilters(),
            limit=result_limit,
        )
        assert len(result) == result_limit

    request = driver.aoss_client.search.call_args.kwargs['body']
    assert request['size'] == 2 * result_limit
    assert request['query']['multi_match']['query'] == 'synthetic query'
    driver.aoss_client.search.assert_called_once()
    driver.client.query.assert_called_once()
