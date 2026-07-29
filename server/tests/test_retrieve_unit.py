"""Unit tests for the retrieve router's episode provenance and status.

No database required: the ZepGraphiti dependency is overridden and
EpisodicNode lookups are patched.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from graphiti_core.errors import NodeNotFoundError

from graph_service.routers import retrieve
from graph_service.zep_graphiti import get_graphiti

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)


def _edge(uuid: str, fact: str, episodes: list[str]):
    edge = MagicMock()
    edge.uuid = uuid
    edge.name = f"fact-{uuid}"
    edge.fact = fact
    edge.valid_at = NOW
    edge.invalid_at = None
    edge.created_at = NOW
    edge.expired_at = None
    edge.episodes = episodes
    return edge


def _episode_node(uuid: str, name: str):
    node = MagicMock()
    node.uuid = uuid
    node.name = name
    return node


def _client(graphiti: MagicMock) -> TestClient:
    app = FastAPI()
    app.include_router(retrieve.router)
    app.dependency_overrides[get_graphiti] = lambda: graphiti
    return TestClient(app)


class TestSearchProvenance:
    def test_search_returns_episode_uuids_and_names(self):
        graphiti = MagicMock()
        graphiti.search = AsyncMock(
            return_value=[_edge("e1", "A depends on B", ["ep-1", "ep-2"])]
        )
        with patch(
            "graph_service.zep_graphiti.EpisodicNode.get_by_uuids",
            new=AsyncMock(
                return_value=[
                    _episode_node("ep-1", "curated:knowledge/a.md"),
                    _episode_node("ep-2", "curated:knowledge/b.md"),
                ]
            ),
        ):
            resp = _client(graphiti).post(
                "/search", json={"query": "a", "max_facts": 5}
            )

        assert resp.status_code == 200
        fact = resp.json()["facts"][0]
        assert fact["episodes"] == ["ep-1", "ep-2"]
        assert fact["episode_names"] == {
            "ep-1": "curated:knowledge/a.md",
            "ep-2": "curated:knowledge/b.md",
        }

    def test_deleted_episode_absent_from_names_but_kept_in_uuids(self):
        graphiti = MagicMock()
        graphiti.search = AsyncMock(
            return_value=[_edge("e1", "A depends on B", ["ep-1", "ep-gone"])]
        )
        with patch(
            "graph_service.zep_graphiti.EpisodicNode.get_by_uuids",
            new=AsyncMock(
                return_value=[_episode_node("ep-1", "curated:knowledge/a.md")]
            ),
        ):
            resp = _client(graphiti).post(
                "/search", json={"query": "a", "max_facts": 5}
            )

        fact = resp.json()["facts"][0]
        assert fact["episodes"] == ["ep-1", "ep-gone"]
        assert fact["episode_names"] == {"ep-1": "curated:knowledge/a.md"}

    def test_search_with_no_episode_refs_skips_lookup(self):
        graphiti = MagicMock()
        graphiti.search = AsyncMock(return_value=[_edge("e1", "fact", [])])
        with patch(
            "graph_service.zep_graphiti.EpisodicNode.get_by_uuids",
            new=AsyncMock(),
        ) as lookup:
            resp = _client(graphiti).post(
                "/search", json={"query": "a", "max_facts": 5}
            )

        assert resp.status_code == 200
        lookup.assert_not_awaited()
        fact = resp.json()["facts"][0]
        assert fact["episodes"] == []
        assert fact["episode_names"] == {}


class TestEpisodeStatus:
    def test_existing_episode(self):
        graphiti = MagicMock()
        with patch(
            "graph_service.routers.retrieve.EpisodicNode.get_by_uuid",
            new=AsyncMock(
                return_value=_episode_node("ep-1", "curated:knowledge/a.md")
            ),
        ):
            resp = _client(graphiti).get("/episodes/status/ep-1")

        assert resp.status_code == 200
        assert resp.json() == {
            "uuid": "ep-1",
            "exists": True,
            "name": "curated:knowledge/a.md",
        }

    def test_missing_episode(self):
        graphiti = MagicMock()
        with patch(
            "graph_service.routers.retrieve.EpisodicNode.get_by_uuid",
            new=AsyncMock(side_effect=NodeNotFoundError("ep-x")),
        ):
            resp = _client(graphiti).get("/episodes/status/ep-x")

        assert resp.status_code == 200
        assert resp.json() == {"uuid": "ep-x", "exists": False, "name": None}
