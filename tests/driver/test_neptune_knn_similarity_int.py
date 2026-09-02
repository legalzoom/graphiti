"""k-NN similarity search against a real OpenSearch instance.

These tests stand up the node_name_embedding and edge_fact_embedding vector
indexes on a real OpenSearch instance (the k-NN plugin ships with the
default opensearchproject/opensearch image) and drive
graphiti_core.search.search_utils.node_similarity_search /
edge_similarity_search end to end through a NeptuneDriver whose aoss_client
points at that instance. The Neptune side is a fake: driver.execute_query
returns canned records for the uuids the fetch step requests, since real
Neptune is not reachable from a test environment.

Run OpenSearch locally before running these tests:

    docker run -d --name graphiti-test-opensearch \\
        -p 9201:9200 -p 9601:9600 \\
        -e discovery.type=single-node \\
        -e DISABLE_SECURITY_PLUGIN=true \\
        opensearchproject/opensearch:2.19.1

Tests skip themselves if OpenSearch is not reachable on OPENSEARCH_TEST_PORT
(default 9201).
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pytest

import graphiti_core.driver.neptune_driver as neptune_driver_module
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.neptune_driver import NeptuneDriver, cosine_similarity_from_knn_score
from graphiti_core.search import search_utils
from graphiti_core.search.search_filters import SearchFilters

pytestmark = pytest.mark.integration

OPENSEARCH_TEST_PORT = int(os.getenv('OPENSEARCH_TEST_PORT', '9201'))
DIMENSION = 4


@pytest.fixture(scope='module')
def opensearch_client():
    from opensearchpy import OpenSearch

    client = OpenSearch(
        hosts=[{'host': 'localhost', 'port': OPENSEARCH_TEST_PORT}],
        use_ssl=False,
        verify_certs=False,
        timeout=2,
    )
    try:
        reachable = client.ping()
    except Exception:
        reachable = False
    if not reachable:
        pytest.skip(f'OpenSearch is not reachable on localhost:{OPENSEARCH_TEST_PORT}')
    yield client


@pytest.fixture
def driver(opensearch_client):
    """A NeptuneDriver whose aoss_client is real OpenSearch. Neptune query
    execution (execute_query) is overridden per-test to fake the Neptune side."""
    d = object.__new__(NeptuneDriver)
    d._aoss_host = f'localhost:{OPENSEARCH_TEST_PORT}'
    d.aoss_client = opensearch_client
    d.provider = GraphProvider.NEPTUNE
    d.search_interface = None
    return d


def _fresh_index_name(prefix: str) -> str:
    return f'{prefix}_{uuid.uuid4().hex[:8]}'


@pytest.fixture
def node_index(driver):
    name = _fresh_index_name('test_node_name_embedding')
    driver.aoss_client.indices.create(
        index=name, body=neptune_driver_module._vector_index_body(DIMENSION)
    )
    yield name
    driver.aoss_client.indices.delete(index=name, ignore=[404])


@pytest.fixture
async def real_vector_indices(driver, monkeypatch):
    """Create the actual node_name_embedding/edge_fact_embedding indices used
    by search_utils, sized to the small test dimension, then delete them."""
    small_dimension_indices = [
        {
            'index_name': 'node_name_embedding',
            'body': neptune_driver_module._vector_index_body(DIMENSION),
        },
        {
            'index_name': 'edge_fact_embedding',
            'body': neptune_driver_module._vector_index_body(DIMENSION),
        },
    ]
    monkeypatch.setattr(neptune_driver_module, 'vector_aoss_indices', small_dimension_indices)
    await driver.create_vector_aoss_indices()
    yield
    driver.aoss_client.indices.delete(index='node_name_embedding', ignore=[404])
    driver.aoss_client.indices.delete(index='edge_fact_embedding', ignore=[404])


def _index_and_refresh(driver, index_name: str, docs: list[dict[str, Any]]) -> None:
    driver.save_to_aoss(index_name, docs)
    driver.aoss_client.indices.refresh(index=index_name)


def _numpy_cosine(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


class TestRunAossKnnQuery:
    @pytest.mark.asyncio
    async def test_orders_by_descending_cosine_similarity(self, driver, node_index):
        docs = [
            {'uuid': 'close', 'group_id': 'g1', 'embedding': [1.0, 0.0, 0.0, 0.0]},
            {'uuid': 'mid', 'group_id': 'g1', 'embedding': [0.7, 0.7, 0.0, 0.0]},
            {'uuid': 'far', 'group_id': 'g1', 'embedding': [0.0, 1.0, 0.0, 0.0]},
        ]
        _index_and_refresh(driver, node_index, docs)

        matches = await driver.run_aoss_knn_query(node_index, [1.0, 0.0, 0.0, 0.0], 10, -1.0, None)

        assert [m['id'] for m in matches] == ['close', 'mid', 'far']
        assert matches == sorted(matches, key=lambda m: m['score'], reverse=True)
        for match in matches:
            expected = _numpy_cosine(
                [1.0, 0.0, 0.0, 0.0],
                next(d['embedding'] for d in docs if d['uuid'] == match['id']),
            )
            assert match['score'] == pytest.approx(expected, abs=1e-3)

    @pytest.mark.asyncio
    async def test_applies_min_score_cutoff(self, driver, node_index):
        docs = [
            {'uuid': 'same', 'group_id': 'g1', 'embedding': [1.0, 0.0, 0.0, 0.0]},
            {'uuid': 'orthogonal', 'group_id': 'g1', 'embedding': [0.0, 1.0, 0.0, 0.0]},
        ]
        _index_and_refresh(driver, node_index, docs)

        matches = await driver.run_aoss_knn_query(node_index, [1.0, 0.0, 0.0, 0.0], 10, 0.5, None)

        assert [m['id'] for m in matches] == ['same']

    @pytest.mark.asyncio
    async def test_filters_by_group_id(self, driver, node_index):
        docs = [
            {'uuid': 'in-group', 'group_id': 'g1', 'embedding': [1.0, 0.0, 0.0, 0.0]},
            {'uuid': 'other-group', 'group_id': 'g2', 'embedding': [1.0, 0.0, 0.0, 0.0]},
        ]
        _index_and_refresh(driver, node_index, docs)

        matches = await driver.run_aoss_knn_query(
            node_index, [1.0, 0.0, 0.0, 0.0], 10, -1.0, ['g1']
        )

        assert [m['id'] for m in matches] == ['in-group']

    @pytest.mark.asyncio
    async def test_missing_index_raises_clear_error(self, driver):
        missing_name = _fresh_index_name('missing')

        with pytest.raises(RuntimeError) as exc_info:
            await driver.run_aoss_knn_query(missing_name, [1.0, 0.0, 0.0, 0.0], 10, -1.0, None)

        message = str(exc_info.value)
        assert missing_name in message
        assert 'backfill_embeddings' in message

    @pytest.mark.asyncio
    async def test_empty_index_returns_no_results(self, driver, node_index):
        matches = await driver.run_aoss_knn_query(node_index, [1.0, 0.0, 0.0, 0.0], 10, -1.0, None)

        assert matches == []


class TestNodeAndEdgeSimilaritySearchEndToEnd:
    """Drives the real search_utils.node_similarity_search / edge_similarity_search
    NEPTUNE branches against the real node_name_embedding/edge_fact_embedding
    indices, with a fake Neptune fetch step."""

    @pytest.mark.asyncio
    async def test_node_similarity_search_fetches_by_uuid_in_score_order(
        self, driver, real_vector_indices
    ):
        docs = [
            {'uuid': 'n-close', 'group_id': 'g1', 'embedding': [1.0, 0.0, 0.0, 0.0]},
            {'uuid': 'n-far', 'group_id': 'g1', 'embedding': [0.6, 0.8, 0.0, 0.0]},
        ]
        _index_and_refresh(driver, 'node_name_embedding', docs)

        records_by_uuid = {
            'n-close': {
                'uuid': 'n-close',
                'name': 'close node',
                'group_id': 'g1',
                'created_at': datetime(2026, 1, 1, tzinfo=timezone.utc),
                'summary': '',
                'labels': ['Entity'],
                'attributes': {},
            },
            'n-far': {
                'uuid': 'n-far',
                'name': 'far node',
                'group_id': 'g1',
                'created_at': datetime(2026, 1, 1, tzinfo=timezone.utc),
                'summary': '',
                'labels': ['Entity'],
                'attributes': {},
            },
        }

        async def fake_execute_query(query, **kwargs):
            requested_ids = [i['id'] for i in kwargs['ids']]
            return [records_by_uuid[uid] for uid in requested_ids], None, None

        driver.execute_query = fake_execute_query

        results = await search_utils.node_similarity_search(
            driver,
            [1.0, 0.0, 0.0, 0.0],
            SearchFilters(),
            group_ids=['g1'],
            limit=10,
            min_score=-1.0,
        )

        assert [n.uuid for n in results] == ['n-close', 'n-far']

    @pytest.mark.asyncio
    async def test_edge_similarity_search_fetches_by_uuid_in_score_order(
        self, driver, real_vector_indices
    ):
        docs = [
            {'uuid': 'e-close', 'group_id': 'g1', 'embedding': [1.0, 0.0, 0.0, 0.0]},
            {'uuid': 'e-far', 'group_id': 'g1', 'embedding': [0.6, 0.8, 0.0, 0.0]},
        ]
        _index_and_refresh(driver, 'edge_fact_embedding', docs)

        records_by_uuid = {
            'e-close': {
                'uuid': 'e-close',
                'source_node_uuid': 'n1',
                'target_node_uuid': 'n2',
                'group_id': 'g1',
                'name': 'REL',
                'fact': 'close fact',
                'episodes': [],
                'created_at': datetime(2026, 1, 1, tzinfo=timezone.utc),
                'expired_at': None,
                'valid_at': None,
                'invalid_at': None,
                'reference_time': None,
                'attributes': {},
            },
            'e-far': {
                'uuid': 'e-far',
                'source_node_uuid': 'n1',
                'target_node_uuid': 'n3',
                'group_id': 'g1',
                'name': 'REL',
                'fact': 'far fact',
                'episodes': [],
                'created_at': datetime(2026, 1, 1, tzinfo=timezone.utc),
                'expired_at': None,
                'valid_at': None,
                'invalid_at': None,
                'reference_time': None,
                'attributes': {},
            },
        }

        async def fake_execute_query(query, **kwargs):
            requested_ids = [i['id'] for i in kwargs['ids']]
            return [records_by_uuid[uid] for uid in requested_ids], None, None

        driver.execute_query = fake_execute_query

        results = await search_utils.edge_similarity_search(
            driver,
            [1.0, 0.0, 0.0, 0.0],
            None,
            None,
            SearchFilters(),
            group_ids=['g1'],
            limit=10,
            min_score=-1.0,
        )

        assert [e.uuid for e in results] == ['e-close', 'e-far']


def test_cosine_conversion_matches_faiss_cosinesimil_formula():
    # Sanity check the conversion constant against the documented OpenSearch formula
    # for the faiss/nmslib cosinesimil space: score = 1 / (1 + d), d = 1 - cosine.
    for cosine in (-1.0, -0.3, 0.0, 0.42, 1.0):
        d = 1 - cosine
        score = 1 / (1 + d)
        assert cosine_similarity_from_knn_score(score) == pytest.approx(cosine, abs=1e-9)
