"""Unit tests for reranker score plumbing on the retrieve routes.

graphiti_core returns reranker scores in SearchResults.edge_reranker_scores, a
list parallel to SearchResults.edges, rather than as a field on EntityEdge. These
tests cover the seam where the routes join the two, since getting it wrong either
drops the score (the bug this replaced) or drops facts.

No database required: the helpers are pure, and the route tests override the
graphiti dependency with a fake that records what the routes send.
"""

from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from graphiti_core.edges import EntityEdge
from graphiti_core.search.search_config import EdgeReranker
from graphiti_core.search.search_config import SearchResults as CoreSearchResults
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

from graph_service.routers import retrieve as retrieve_router
from graph_service.routers.retrieve import _facts_with_scores, _rrf_search_config
from graph_service.zep_graphiti import get_fact_result_from_edge, get_graphiti


def _edge(fact: str = 'a depends on b') -> EntityEdge:
    return EntityEdge(
        group_id='test-group',
        source_node_uuid='source-uuid',
        target_node_uuid='target-uuid',
        created_at=datetime.now(timezone.utc),
        name='DEPENDS_ON',
        fact=fact,
    )


class TestGetFactResultFromEdge:
    def test_score_defaults_to_zero(self):
        """Callers that did not rank (fetch-by-uuid) get 0.0, not an error."""
        result = get_fact_result_from_edge(_edge())
        assert result.score == 0.0

    def test_score_is_passed_through(self):
        result = get_fact_result_from_edge(_edge(), 1.5)
        assert result.score == 1.5

    def test_other_fields_still_serialized(self):
        edge = _edge(fact='x publishes to y')
        result = get_fact_result_from_edge(edge, 0.5)
        assert result.uuid == edge.uuid
        assert result.name == 'DEPENDS_ON'
        assert result.fact == 'x publishes to y'
        assert result.expired_at is None


class TestFactsWithScores:
    def test_scores_align_with_edges(self):
        edges = [_edge('fact one'), _edge('fact two'), _edge('fact three')]
        results = CoreSearchResults(edges=edges, edge_reranker_scores=[2.0, 0.5, 0.333])

        facts = _facts_with_scores(results)

        assert [f.fact for f in facts] == ['fact one', 'fact two', 'fact three']
        assert [f.score for f in facts] == [2.0, 0.5, 0.333]

    def test_empty_scores_does_not_drop_facts(self):
        """A zip() here would return nothing and silently empty the response."""
        edges = [_edge('fact one'), _edge('fact two')]
        results = CoreSearchResults(edges=edges, edge_reranker_scores=[])

        facts = _facts_with_scores(results)

        assert len(facts) == 2
        assert [f.score for f in facts] == [0.0, 0.0]

    def test_short_score_list_degrades_per_fact(self):
        edges = [_edge('fact one'), _edge('fact two'), _edge('fact three')]
        results = CoreSearchResults(edges=edges, edge_reranker_scores=[1.0])

        facts = _facts_with_scores(results)

        assert len(facts) == 3
        assert [f.score for f in facts] == [1.0, 0.0, 0.0]

    def test_no_edges_yields_no_facts(self):
        assert _facts_with_scores(CoreSearchResults()) == []


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


class _FakeGraphiti:
    """Records search_() kwargs and returns canned results.

    Deliberately does not subclass ZepGraphiti: the point is to assert what the
    routes send and serialize, without a graph database.
    """

    def __init__(self, results: CoreSearchResults):
        self._results = results
        self.calls: list[dict] = []

    async def search_(self, **kwargs) -> CoreSearchResults:
        self.calls.append(kwargs)
        return self._results


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

    def test_score_is_in_the_response_body(self, client: TestClient):
        response = client.post('/get-memory', json=self._payload())

        assert response.status_code == 200
        assert [f['score'] for f in response.json()['facts']] == [2.0, 0.5]

    def test_center_node_uuid_is_not_forwarded(
        self, client: TestClient, fake_graphiti: _FakeGraphiti
    ):
        """Pre-existing behavior: the field is accepted and ignored. Forwarding it
        would switch the reranker to node-distance and change ranking."""
        client.post('/get-memory', json=self._payload(center_node_uuid='some-uuid'))

        assert 'center_node_uuid' not in fake_graphiti.calls[0]
