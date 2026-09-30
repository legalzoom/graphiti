"""Boundary tests use in-memory entities because no captured browse page exists yet."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode

import graphiti_mcp_server


@pytest.mark.asyncio
async def test_browse_group_pages_nodes_and_edges_with_their_endpoints(monkeypatch) -> None:
    driver = SimpleNamespace(provider=GraphProvider.NEO4J)
    monkeypatch.setattr(
        graphiti_mcp_server,
        'graphiti_service',
        SimpleNamespace(get_client=AsyncMock(return_value=SimpleNamespace(driver=driver))),
    )
    first = EntityNode(uuid='11111111-1111-4111-8111-111111111111', name='First', group_id='lz-org')
    second = EntityNode(
        uuid='22222222-2222-4222-8222-222222222222', name='Second', group_id='lz-org'
    )
    first_edge = EntityEdge(
        uuid='33333333-3333-4333-8333-333333333333',
        name='KNOWS',
        fact='First knows Second',
        source_node_uuid=first.uuid,
        target_node_uuid=second.uuid,
        group_id='lz-org',
        created_at=datetime.now(timezone.utc),
    )
    next_edge = first_edge.model_copy(update={'uuid': '44444444-4444-4444-8444-444444444444'})
    get_nodes = AsyncMock(return_value=[first, second])
    get_edges = AsyncMock(return_value=[first_edge, next_edge])
    get_endpoints = AsyncMock(return_value=[second])
    monkeypatch.setattr(EntityNode, 'get_by_group_ids', get_nodes)
    monkeypatch.setattr(EntityEdge, 'get_by_group_ids', get_edges)
    monkeypatch.setattr(EntityNode, 'get_by_uuids', get_endpoints)

    result = await graphiti_mcp_server.browse_group_graph('lz-org', max_nodes=1, max_facts=1)
    assert result['group_id'] == 'lz-org'
    assert [node['uuid'] for node in result['nodes']] == [first.uuid, second.uuid]
    assert [fact['uuid'] for fact in result['facts']] == [first_edge.uuid]
    assert result['next_node_cursor'] == first.uuid
    assert result['next_fact_cursor'] == first_edge.uuid
    get_nodes.assert_awaited_once_with(driver, ['lz-org'], limit=2, uuid_cursor=None)
    get_edges.assert_awaited_once_with(driver, ['lz-org'], limit=2, uuid_cursor=None)
    get_endpoints.assert_awaited_once_with(driver, [second.uuid], group_id='lz-org')

    get_endpoints.return_value = [second.model_copy(update={'group_id': 'another-group'})]
    denied = await graphiti_mcp_server.browse_group_graph('lz-org', max_nodes=1, max_facts=1)
    assert 'outside the requested groups' in denied['error']


@pytest.mark.asyncio
async def test_browse_group_skips_exhausted_streams_and_rejects_foreign_groups(monkeypatch) -> None:
    driver = SimpleNamespace(provider=GraphProvider.NEO4J)
    monkeypatch.setattr(
        graphiti_mcp_server,
        'graphiti_service',
        SimpleNamespace(get_client=AsyncMock(return_value=SimpleNamespace(driver=driver))),
    )
    foreign = EntityNode(name='Foreign', group_id='another-group')
    get_nodes = AsyncMock(return_value=[foreign])
    get_edges = AsyncMock(return_value=[])
    monkeypatch.setattr(EntityNode, 'get_by_group_ids', get_nodes)
    monkeypatch.setattr(EntityEdge, 'get_by_group_ids', get_edges)

    assert (await graphiti_mcp_server.browse_group_graph('lz-org', max_nodes=0, max_facts=1)) == {
        'group_id': 'lz-org',
        'nodes': [],
        'facts': [],
        'next_node_cursor': None,
        'next_fact_cursor': None,
    }
    get_nodes.assert_not_awaited()
    denied = await graphiti_mcp_server.browse_group_graph('lz-org', max_nodes=1, max_facts=1)
    assert 'outside the requested groups' in denied['error']
