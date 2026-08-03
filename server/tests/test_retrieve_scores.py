"""Unit tests for score and entity-node plumbing on the retrieve routes.

Two things graphiti_core has but FactResult did not expose. Reranker scores live
in SearchResults.edge_reranker_scores, a list parallel to SearchResults.edges,
not on EntityEdge. Entity identity lives on EntityEdge as source_node_uuid and
target_node_uuid, with the names only reachable through a node lookup. These
tests cover both seams, since getting either wrong drops data silently: the
score bug this replaced returned facts nobody could rank, and a bad join would
drop facts entirely.

No database required. The helpers are pure, the route tests override the
graphiti dependency with a fake that records what the routes send, and node
lookups are stubbed.
"""

from datetime import datetime, timezone
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from graphiti_core.edges import EntityEdge
from graphiti_core.search.search_config import EdgeReranker
from graphiti_core.search.search_config import SearchResults as CoreSearchResults
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

from graph_service import zep_graphiti
from graph_service.routers import retrieve as retrieve_router
from graph_service.routers.retrieve import _facts_from_results, _rrf_search_config
from graph_service.zep_graphiti import (
    ZepGraphiti,
    get_fact_result_from_edge,
    get_graphiti,
    resolve_node_names,
)

NODE_NAMES = {
    'source-uuid': 'compliance-platform',
    'target-uuid': 'authorization-service',
}


def _edge(
    fact: str = 'a depends on b',
    source_uuid: str = 'source-uuid',
    target_uuid: str = 'target-uuid',
) -> EntityEdge:
    return EntityEdge(
        group_id='test-group',
        source_node_uuid=source_uuid,
        target_node_uuid=target_uuid,
        created_at=datetime.now(timezone.utc),
        name='DEPENDS_ON',
        fact=fact,
    )


class _FakeNode:
    def __init__(self, uuid: str, name: str):
        self.uuid = uuid
        self.name = name


class _FakeGraphiti:
    """Records search_() kwargs and returns canned results.

    Deliberately does not subclass ZepGraphiti: the point is to assert what the
    routes send and serialize, without a graph database. driver is None so an
    unstubbed node lookup raises and exercises the degradation path.
    """

    driver = None

    def __init__(self, results: CoreSearchResults):
        self._results = results
        self.calls: list[dict] = []

    async def search_(self, **kwargs) -> CoreSearchResults:
        self.calls.append(kwargs)
        return self._results


def _no_nodes_graphiti() -> ZepGraphiti:
    """A stand-in with no usable driver, so node lookups degrade.

    cast rather than subclassing: inheriting ZepGraphiti would drag in database
    configuration these tests exist to avoid.
    """
    return cast(ZepGraphiti, _FakeGraphiti(CoreSearchResults()))


@pytest.fixture
def resolvable_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let node names resolve without a database."""

    async def _get_by_uuids(driver, uuids, group_id=None):
        return [_FakeNode(u, NODE_NAMES[u]) for u in uuids if u in NODE_NAMES]

    monkeypatch.setattr(zep_graphiti.EntityNode, 'get_by_uuids', _get_by_uuids)


class TestGetFactResultFromEdge:
    def test_score_defaults_to_zero(self):
        """Callers that did not rank (fetch-by-uuid) get 0.0, not an error."""
        assert get_fact_result_from_edge(_edge()).score == 0.0

    def test_score_is_passed_through(self):
        assert get_fact_result_from_edge(_edge(), 1.5).score == 1.5

    def test_node_uuids_always_populated(self):
        """The edge carries these directly, so they need no lookup."""
        result = get_fact_result_from_edge(_edge())
        assert result.source_node_uuid == 'source-uuid'
        assert result.target_node_uuid == 'target-uuid'

    def test_node_names_omitted_without_a_mapping(self):
        result = get_fact_result_from_edge(_edge())
        assert result.source_node is None
        assert result.target_node is None

    def test_node_names_populated_from_mapping(self):
        result = get_fact_result_from_edge(_edge(), 1.0, NODE_NAMES)
        assert result.source_node == 'compliance-platform'
        assert result.target_node == 'authorization-service'

    def test_partial_mapping_leaves_the_other_name_none(self):
        """A node missing from the graph must not blank out the one that resolved."""
        result = get_fact_result_from_edge(_edge(), 1.0, {'source-uuid': 'compliance-platform'})
        assert result.source_node == 'compliance-platform'
        assert result.target_node is None

    def test_other_fields_still_serialized(self):
        edge = _edge(fact='x publishes to y')
        result = get_fact_result_from_edge(edge, 0.5)
        assert result.uuid == edge.uuid
        assert result.name == 'DEPENDS_ON'
        assert result.fact == 'x publishes to y'
        assert result.expired_at is None


class TestResolveNodeNames:
    @pytest.mark.asyncio
    async def test_maps_uuid_to_name(self, resolvable_nodes: None):
        names = await resolve_node_names(_no_nodes_graphiti(), [_edge()])
        assert names == NODE_NAMES

    @pytest.mark.asyncio
    async def test_deduplicates_uuids_across_edges(self, resolvable_nodes: None):
        """Three edges over the same two nodes must not fan out to six lookups."""
        seen: list[list[str]] = []

        async def _record(driver, uuids, group_id=None):
            seen.append(list(uuids))
            return [_FakeNode(u, NODE_NAMES[u]) for u in uuids if u in NODE_NAMES]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(zep_graphiti.EntityNode, 'get_by_uuids', _record)
            await resolve_node_names(_no_nodes_graphiti(), [_edge(), _edge(), _edge()])

        assert seen == [['source-uuid', 'target-uuid']]

    @pytest.mark.asyncio
    async def test_no_edges_skips_the_lookup(self):
        assert await resolve_node_names(_no_nodes_graphiti(), []) == {}

    @pytest.mark.asyncio
    async def test_lookup_failure_degrades_to_empty(self):
        """Names are additive. A node-lookup failure must not fail the search."""
        assert await resolve_node_names(_no_nodes_graphiti(), [_edge()]) == {}


class TestFactsFromResults:
    @pytest.mark.asyncio
    async def test_scores_align_with_edges(self):
        edges = [_edge('fact one'), _edge('fact two'), _edge('fact three')]
        results = CoreSearchResults(edges=edges, edge_reranker_scores=[2.0, 0.5, 0.333])

        facts = await _facts_from_results(_no_nodes_graphiti(), results)

        assert [f.fact for f in facts] == ['fact one', 'fact two', 'fact three']
        assert [f.score for f in facts] == [2.0, 0.5, 0.333]

    @pytest.mark.asyncio
    async def test_empty_scores_does_not_drop_facts(self):
        """A zip() here would return nothing and silently empty the response."""
        edges = [_edge('fact one'), _edge('fact two')]
        results = CoreSearchResults(edges=edges, edge_reranker_scores=[])

        facts = await _facts_from_results(_no_nodes_graphiti(), results)

        assert len(facts) == 2
        assert [f.score for f in facts] == [0.0, 0.0]

    @pytest.mark.asyncio
    async def test_short_score_list_degrades_per_fact(self):
        edges = [_edge('fact one'), _edge('fact two'), _edge('fact three')]
        results = CoreSearchResults(edges=edges, edge_reranker_scores=[1.0])

        facts = await _facts_from_results(_no_nodes_graphiti(), results)

        assert len(facts) == 3
        assert [f.score for f in facts] == [1.0, 0.0, 0.0]

    @pytest.mark.asyncio
    async def test_names_are_joined_onto_every_fact(self, resolvable_nodes: None):
        edges = [_edge('fact one'), _edge('fact two')]
        results = CoreSearchResults(edges=edges, edge_reranker_scores=[1.0, 0.5])

        facts = await _facts_from_results(_no_nodes_graphiti(), results)

        assert all(f.source_node == 'compliance-platform' for f in facts)
        assert all(f.target_node == 'authorization-service' for f in facts)

    @pytest.mark.asyncio
    async def test_no_edges_yields_no_facts(self):
        assert await _facts_from_results(_no_nodes_graphiti(), CoreSearchResults()) == []


class TestRrfSearchConfig:
    def test_limit_is_applied(self):
        assert _rrf_search_config(25).limit == 25

    def test_recipe_singleton_is_not_mutated(self):
        """Regression guard: Graphiti.search() assigns limit on the shared
        module-level recipe, leaking one caller's limit into every other."""
        before = EDGE_HYBRID_SEARCH_RRF.limit

        _rrf_search_config(999)

        assert EDGE_HYBRID_SEARCH_RRF.limit == before

    def test_edge_config_is_deep_copied(self):
        config = _rrf_search_config(10)
        assert config.edge_config is not EDGE_HYBRID_SEARCH_RRF.edge_config
        assert config.edge_config is not None
        assert EDGE_HYBRID_SEARCH_RRF.edge_config is not None
        assert config.edge_config.reranker == EDGE_HYBRID_SEARCH_RRF.edge_config.reranker


@pytest.fixture
def fake_graphiti() -> _FakeGraphiti:
    return _FakeGraphiti(
        CoreSearchResults(
            edges=[_edge('fact one'), _edge('fact two')],
            edge_reranker_scores=[2.0, 0.5],
        )
    )


@pytest.fixture
def client(fake_graphiti: _FakeGraphiti) -> TestClient:
    app = FastAPI()
    app.include_router(retrieve_router.router)
    app.dependency_overrides[get_graphiti] = lambda: fake_graphiti
    return TestClient(app)


class TestSearchRoute:
    """End-to-end wiring. The bug this change fixes was a contract mismatch that
    no test covered, so the response shape is asserted over HTTP, not in-process."""

    def test_score_is_in_the_response_body(self, client: TestClient):
        response = client.post('/search', json={'query': 'who depends on b', 'max_facts': 2})

        assert response.status_code == 200
        facts = response.json()['facts']
        assert [f['fact'] for f in facts] == ['fact one', 'fact two']
        assert [f['score'] for f in facts] == [2.0, 0.5]

    def test_node_fields_are_in_the_response_body(self, client: TestClient, resolvable_nodes: None):
        response = client.post('/search', json={'query': 'q'})

        fact = response.json()['facts'][0]
        assert fact['source_node_uuid'] == 'source-uuid'
        assert fact['target_node_uuid'] == 'target-uuid'
        assert fact['source_node'] == 'compliance-platform'
        assert fact['target_node'] == 'authorization-service'

    def test_search_still_succeeds_when_node_lookup_fails(self, client: TestClient):
        """No resolvable_nodes fixture, so the lookup raises."""
        response = client.post('/search', json={'query': 'q'})

        assert response.status_code == 200
        fact = response.json()['facts'][0]
        assert fact['source_node'] is None
        assert fact['source_node_uuid'] == 'source-uuid'
        assert fact['score'] == 2.0

    def test_max_facts_becomes_config_limit(self, client: TestClient, fake_graphiti: _FakeGraphiti):
        client.post('/search', json={'query': 'q', 'max_facts': 7})

        assert fake_graphiti.calls[0]['config'].limit == 7

    def test_group_ids_are_forwarded(self, client: TestClient, fake_graphiti: _FakeGraphiti):
        client.post('/search', json={'query': 'q', 'group_ids': ['opr']})

        assert fake_graphiti.calls[0]['group_ids'] == ['opr']

    def test_rrf_reranker_is_used(self, client: TestClient, fake_graphiti: _FakeGraphiti):
        client.post('/search', json={'query': 'q'})

        edge_config = fake_graphiti.calls[0]['config'].edge_config
        assert edge_config is not None
        assert edge_config.reranker == EdgeReranker.rrf


class TestGetMemoryRoute:
    def _payload(self, **overrides) -> dict:
        payload = {
            'group_id': 'opr',
            'max_facts': 2,
            'center_node_uuid': None,
            'messages': [{'content': 'what depends on b', 'role_type': 'user', 'role': None}],
        }
        payload.update(overrides)
        return payload

    def test_score_and_nodes_in_the_response_body(self, client: TestClient, resolvable_nodes: None):
        response = client.post('/get-memory', json=self._payload())

        assert response.status_code == 200
        facts = response.json()['facts']
        assert [f['score'] for f in facts] == [2.0, 0.5]
        assert facts[0]['source_node'] == 'compliance-platform'

    def test_center_node_uuid_is_not_forwarded(
        self, client: TestClient, fake_graphiti: _FakeGraphiti
    ):
        """Pre-existing behavior: the field is accepted and ignored. Forwarding it
        would switch the reranker to node-distance and change ranking."""
        client.post('/get-memory', json=self._payload(center_node_uuid='some-uuid'))

        assert 'center_node_uuid' not in fake_graphiti.calls[0]
