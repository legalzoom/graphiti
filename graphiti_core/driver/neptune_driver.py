"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import datetime
import json
import logging
from collections.abc import Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from functools import partial
from typing import Any

import boto3
from botocore.config import Config
from opensearchpy import OpenSearch, Urllib3AWSV4SignerAuth, Urllib3HttpConnection, helpers
from opensearchpy.exceptions import NotFoundError

from graphiti_core.async_limiter import (
    AsyncCapacityLease,
    AsyncCapacityLimiter,
    current_capacity_lease,
)
from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
from graphiti_core.driver.neptune.operations.community_edge_ops import (
    NeptuneCommunityEdgeOperations,
)
from graphiti_core.driver.neptune.operations.community_node_ops import (
    NeptuneCommunityNodeOperations,
)
from graphiti_core.driver.neptune.operations.entity_edge_ops import NeptuneEntityEdgeOperations
from graphiti_core.driver.neptune.operations.entity_node_ops import NeptuneEntityNodeOperations
from graphiti_core.driver.neptune.operations.episode_node_ops import NeptuneEpisodeNodeOperations
from graphiti_core.driver.neptune.operations.episodic_edge_ops import NeptuneEpisodicEdgeOperations
from graphiti_core.driver.neptune.operations.graph_ops import NeptuneGraphMaintenanceOperations
from graphiti_core.driver.neptune.operations.has_episode_edge_ops import (
    NeptuneHasEpisodeEdgeOperations,
)
from graphiti_core.driver.neptune.operations.next_episode_edge_ops import (
    NeptuneNextEpisodeEdgeOperations,
)
from graphiti_core.driver.neptune.operations.saga_node_ops import NeptuneSagaNodeOperations
from graphiti_core.driver.neptune.operations.search_ops import NeptuneSearchOperations
from graphiti_core.driver.operations.community_edge_ops import CommunityEdgeOperations
from graphiti_core.driver.operations.community_node_ops import CommunityNodeOperations
from graphiti_core.driver.operations.entity_edge_ops import EntityEdgeOperations
from graphiti_core.driver.operations.entity_node_ops import EntityNodeOperations
from graphiti_core.driver.operations.episode_node_ops import EpisodeNodeOperations
from graphiti_core.driver.operations.episodic_edge_ops import EpisodicEdgeOperations
from graphiti_core.driver.operations.graph_ops import GraphMaintenanceOperations
from graphiti_core.driver.operations.has_episode_edge_ops import HasEpisodeEdgeOperations
from graphiti_core.driver.operations.next_episode_edge_ops import NextEpisodeEdgeOperations
from graphiti_core.driver.operations.saga_node_ops import SagaNodeOperations
from graphiti_core.driver.operations.search_ops import SearchOperations
from graphiti_core.embedder.client import EMBEDDING_DIM

logger = logging.getLogger(__name__)
DEFAULT_SIZE = 10
AOSS_QUERY_CONCURRENCY = 2
AOSS_QUERY_MAX_WAITERS = AOSS_QUERY_CONCURRENCY * 16
MAX_AOSS_QUERY_SIZE = 1000

# Keep AOSS reads off asyncio's default executor, which Neptune graph I/O also uses. Async
# admission ensures bursts and cancelled waiters cannot create an unbounded executor backlog.
_AOSS_QUERY_EXECUTOR = ThreadPoolExecutor(
    max_workers=AOSS_QUERY_CONCURRENCY,
    thread_name_prefix='graphiti-aoss-query',
)
_AOSS_QUERY_CAPACITY = AsyncCapacityLimiter(
    AOSS_QUERY_CONCURRENCY,
    max_waiters=AOSS_QUERY_MAX_WAITERS,
)

# Only similarity pipelines with a bound capacity lease use this executor. Direct submission gives
# us the concurrent.futures.Future needed to retain that lease independently of an event loop that
# may close while the synchronous Neptune client is still running. Omitting max_workers is
# intentional: the similarity admission limit (currently two pipelines) bounds submissions, and a
# second independently configured concurrency limit could drift or deadlock admitted work.
_NEPTUNE_SIMILARITY_QUERY_EXECUTOR = ThreadPoolExecutor(
    thread_name_prefix='graphiti-neptune-similarity-query'
)


def _finish_detached_aoss_query(future: Future[Any], lease: AsyncCapacityLease) -> None:
    try:
        if future.cancelled():
            return
        exception = future.exception()
        if exception is not None:
            logger.error('Detached AOSS query failed after caller cancellation', exc_info=exception)
    finally:
        lease.release()


def _consume_aoss_wrapper_result(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


def _consume_detached_neptune_query(future: Future[Any]) -> None:
    if future.cancelled():
        return
    exception = future.exception()
    if exception is not None:
        logger.error(
            'Detached Neptune similarity query failed after caller cancellation',
            exc_info=exception,
        )


# read_timeout is kept slightly above the default Neptune cluster
# `neptune_query_timeout` (120s) so a slow-but-alive query gets a clean
# botocore timeout from Neptune instead of the socket being torn down first.
# `standard` retry mode retries transient connection errors (e.g. the
# connection getting reset mid-response) instead of surfacing them directly.
NEPTUNE_BOTO_CONFIG = Config(
    connect_timeout=10,
    read_timeout=130,
    retries={'max_attempts': 3, 'mode': 'standard'},
)

aoss_indices = [
    {
        'index_name': 'node_name_and_summary',
        'body': {
            'mappings': {
                'properties': {
                    'uuid': {'type': 'keyword'},
                    'name': {'type': 'text'},
                    'summary': {'type': 'text'},
                    'group_id': {'type': 'text'},
                }
            }
        },
        'query': {
            'query': {'multi_match': {'query': '', 'fields': ['name', 'summary', 'group_id']}},
            'size': DEFAULT_SIZE,
        },
    },
    {
        'index_name': 'community_name',
        'body': {
            'mappings': {
                'properties': {
                    'uuid': {'type': 'keyword'},
                    'name': {'type': 'text'},
                    'group_id': {'type': 'text'},
                }
            }
        },
        'query': {
            'query': {'multi_match': {'query': '', 'fields': ['name', 'group_id']}},
            'size': DEFAULT_SIZE,
        },
    },
    {
        'index_name': 'episode_content',
        'body': {
            'mappings': {
                'properties': {
                    'uuid': {'type': 'keyword'},
                    'content': {'type': 'text'},
                    'source': {'type': 'text'},
                    'source_description': {'type': 'text'},
                    'group_id': {'type': 'text'},
                }
            }
        },
        'query': {
            'query': {
                'multi_match': {
                    'query': '',
                    'fields': ['content', 'source', 'source_description', 'group_id'],
                }
            },
            'size': DEFAULT_SIZE,
        },
    },
    {
        'index_name': 'edge_name_and_fact',
        'body': {
            'mappings': {
                'properties': {
                    'uuid': {'type': 'keyword'},
                    'name': {'type': 'text'},
                    'fact': {'type': 'text'},
                    'group_id': {'type': 'text'},
                }
            }
        },
        'query': {
            'query': {'multi_match': {'query': '', 'fields': ['name', 'fact', 'group_id']}},
            'size': DEFAULT_SIZE,
        },
    },
]

# The OpenSearch Serverless collection backing these indexes must be of type
# VECTORSEARCH. A SEARCH type collection rejects the knn_vector field type.
VECTOR_EMBEDDING_FIELD = 'embedding'
VECTOR_INDEX_SPACE_TYPE = 'cosinesimil'
VECTOR_INDEX_ENGINE = 'faiss'
VECTOR_INDEX_METHOD = 'hnsw'


class VectorIndexUnsupportedError(RuntimeError):
    """Raised when the AOSS collection rejects a knn_vector index mapping.

    An OpenSearch Serverless collection of type SEARCH rejects knn_vector
    fields; only a VECTORSEARCH collection accepts them. There is no scan
    fallback: a rejected mapping is a deployment configuration error, not a
    degraded mode to run in.
    """


def _mapping_has_knn_vector(body: dict[str, Any]) -> bool:
    properties = body.get('mappings', {}).get('properties', {})
    return any(prop.get('type') == 'knn_vector' for prop in properties.values())


def _vector_index_body(dimension: int) -> dict[str, Any]:
    return {
        'settings': {'index': {'knn': True}},
        'mappings': {
            'properties': {
                'uuid': {'type': 'keyword'},
                'group_id': {'type': 'keyword'},
                VECTOR_EMBEDDING_FIELD: {
                    'type': 'knn_vector',
                    'dimension': dimension,
                    'space_type': VECTOR_INDEX_SPACE_TYPE,
                    'method': {
                        'name': VECTOR_INDEX_METHOD,
                        'engine': VECTOR_INDEX_ENGINE,
                    },
                },
            }
        },
    }


# knn_vector indexes for Neptune similarity search. OpenSearch cannot enable
# `index.knn` on an existing index, so these are separate indexes from the
# four text indexes above rather than added fields on them.
vector_aoss_indices = [
    {
        'index_name': 'node_name_embedding',
        'body': _vector_index_body(EMBEDDING_DIM),
    },
    {
        'index_name': 'edge_fact_embedding',
        'body': _vector_index_body(EMBEDDING_DIM),
    },
]

aoss_indices = aoss_indices + vector_aoss_indices


def cosine_similarity_from_knn_score(score: float) -> float:
    """Convert an OpenSearch k-NN score back to cosine similarity.

    The vector indexes use the cosinesimil space with the faiss engine, where
    OpenSearch scores as ``1 / (1 + d)`` and ``d = 1 - cosine_similarity``.
    Solving for cosine similarity gives ``2 - (1 / score)``.
    """
    if score <= 0:
        return -1.0
    return 2.0 - (1.0 / score)


class NeptuneDatabaseClient:
    """Lightweight Neptune Database query client using boto3 directly.

    This replaces langchain_aws's NeptuneGraph, which unconditionally calls the
    Summary/Statistics API on construction—requiring engine >=1.2.1.0 with
    statistics enabled.  Graphiti only needs openCypher query execution, not
    langchain's schema introspection, so we skip that overhead entirely.
    """

    def __init__(self, host: str, port: int = 8182):
        session = boto3.Session()
        self.client = session.client(
            'neptunedata',
            endpoint_url=f'https://{host}:{port}',
            config=NEPTUNE_BOTO_CONFIG,
        )

    def query(self, query: str, params: dict | None = None) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {'openCypherQuery': query}
        if params:
            kwargs['parameters'] = json.dumps(params)
        return self.client.execute_open_cypher_query(**kwargs)['results']


class NeptuneAnalyticsClient:
    """Lightweight Neptune Analytics query client using boto3 directly.

    Same rationale as NeptuneDatabaseClient—avoids the langchain_aws
    NeptuneAnalyticsGraph class and its mandatory schema introspection.
    """

    def __init__(self, graph_identifier: str):
        session = boto3.Session()
        self.client = session.client('neptune-graph', config=NEPTUNE_BOTO_CONFIG)
        self.graph_identifier = graph_identifier

    def query(self, query: str, params: dict | None = None) -> list[dict[str, Any]]:
        resp = self.client.execute_query(
            graphIdentifier=self.graph_identifier,
            queryString=query,
            parameters=params or {},
            language='OPEN_CYPHER',
        )
        return json.loads(resp['payload'].read().decode('UTF-8'))['results']


class NeptuneDriver(GraphDriver):
    provider: GraphProvider = GraphProvider.NEPTUNE

    def __init__(self, host: str, aoss_host: str, port: int = 8182, aoss_port: int = 443):
        """This initializes a NeptuneDriver for use with Neptune as a backend

        Args:
            host (str): The Neptune Database or Neptune Analytics host
            aoss_host (str): The OpenSearch host value
            port (int, optional): The Neptune Database port, ignored for Neptune Analytics. Defaults to 8182.
            aoss_port (int, optional): The OpenSearch port. Defaults to 443.
        """
        self._database = 'default'

        if not host:
            raise ValueError('You must provide an endpoint to create a NeptuneDriver')

        if host.startswith('neptune-db://'):
            # This is a Neptune Database Cluster
            endpoint = host.replace('neptune-db://', '')
            self.client = NeptuneDatabaseClient(endpoint, port)
            logger.debug('Creating Neptune Database session for %s', host)
        elif host.startswith('neptune-graph://'):
            # This is a Neptune Analytics Graph
            graphId = host.replace('neptune-graph://', '')
            self.client = NeptuneAnalyticsClient(graphId)
            logger.debug('Creating Neptune Graph session for %s', host)
        else:
            raise ValueError(
                'You must provide an endpoint to create a NeptuneDriver as either neptune-db://<endpoint> or neptune-graph://<graphid>'
            )

        if not aoss_host:
            raise ValueError('You must provide an AOSS endpoint to create an OpenSearch driver.')

        self._aoss_host = aoss_host

        session = boto3.Session()
        self.aoss_client = OpenSearch(
            hosts=[{'host': aoss_host, 'port': aoss_port}],
            http_auth=Urllib3AWSV4SignerAuth(
                session.get_credentials(), session.region_name, 'aoss'
            ),
            use_ssl=True,
            verify_certs=True,
            connection_class=Urllib3HttpConnection,
            pool_maxsize=20,
        )

        # Instantiate Neptune operations
        self._entity_node_ops = NeptuneEntityNodeOperations()
        self._episode_node_ops = NeptuneEpisodeNodeOperations()
        self._community_node_ops = NeptuneCommunityNodeOperations(driver=self)
        self._saga_node_ops = NeptuneSagaNodeOperations()
        self._entity_edge_ops = NeptuneEntityEdgeOperations()
        self._episodic_edge_ops = NeptuneEpisodicEdgeOperations()
        self._community_edge_ops = NeptuneCommunityEdgeOperations()
        self._has_episode_edge_ops = NeptuneHasEpisodeEdgeOperations()
        self._next_episode_edge_ops = NeptuneNextEpisodeEdgeOperations()
        self._search_ops = NeptuneSearchOperations(driver=self)
        self._graph_ops = NeptuneGraphMaintenanceOperations(driver=self)

    # --- Operations properties ---

    @property
    def entity_node_ops(self) -> EntityNodeOperations:
        return self._entity_node_ops

    @property
    def episode_node_ops(self) -> EpisodeNodeOperations:
        return self._episode_node_ops

    @property
    def community_node_ops(self) -> CommunityNodeOperations:
        return self._community_node_ops

    @property
    def saga_node_ops(self) -> SagaNodeOperations:
        return self._saga_node_ops

    @property
    def entity_edge_ops(self) -> EntityEdgeOperations:
        return self._entity_edge_ops

    @property
    def episodic_edge_ops(self) -> EpisodicEdgeOperations:
        return self._episodic_edge_ops

    @property
    def community_edge_ops(self) -> CommunityEdgeOperations:
        return self._community_edge_ops

    @property
    def has_episode_edge_ops(self) -> HasEpisodeEdgeOperations:
        return self._has_episode_edge_ops

    @property
    def next_episode_edge_ops(self) -> NextEpisodeEdgeOperations:
        return self._next_episode_edge_ops

    @property
    def search_ops(self) -> SearchOperations:
        return self._search_ops

    @property
    def graph_ops(self) -> GraphMaintenanceOperations:
        return self._graph_ops

    def _sanitize_parameters(self, query, params: dict):
        if isinstance(query, list):
            queries = []
            for q in query:
                queries.append(self._sanitize_parameters(q, params))
            return queries
        else:
            for k, v in params.items():
                if isinstance(v, datetime.datetime):
                    params[k] = v.isoformat()
                elif isinstance(v, list):
                    # Handle lists that might contain datetime objects
                    for i, item in enumerate(v):
                        if isinstance(item, datetime.datetime):
                            v[i] = item.isoformat()
                            query = str(query).replace(f'${k}', f'datetime(${k})')
                        if isinstance(item, dict):
                            query = self._sanitize_parameters(query, v[i])

                    # If the list contains datetime objects, we need to wrap each element with datetime()
                    if any(isinstance(item, str) and 'T' in item for item in v):
                        # Create a new list expression with datetime() wrapped around each element
                        datetime_list = (
                            '['
                            + ', '.join(
                                f'datetime("{item}")'
                                if isinstance(item, str) and 'T' in item
                                else repr(item)
                                for item in v
                            )
                            + ']'
                        )
                        query = str(query).replace(f'${k}', datetime_list)
                elif isinstance(v, dict):
                    query = self._sanitize_parameters(query, v)
            return query

    async def execute_query(
        self, cypher_query_, **kwargs: Any
    ) -> tuple[list[dict[str, Any]], None, None]:
        params = dict(kwargs)
        # Unwrap nested 'params' dict (legacy search_utils compatibility)
        if 'params' in params and isinstance(params['params'], dict):
            nested = params.pop('params')
            params.update(nested)
        # Remove kwargs that are not openCypher parameters
        for key in ('routing_', 'database_', 'search_vector', 'min_score'):
            params.pop(key, None)
        if isinstance(cypher_query_, list):
            result: list[dict[str, Any]] = []
            for q in cypher_query_:
                result, _, _ = await self._execute_query_in_thread(q[0], q[1])
            return result, None, None
        else:
            return await self._execute_query_in_thread(cypher_query_, params)

    async def _execute_query_in_thread(
        self, cypher_query_: str, params: Any
    ) -> tuple[list[dict[str, Any]], None, None]:
        lease = current_capacity_lease()
        if lease is None:
            return await asyncio.to_thread(self._run_query, cypher_query_, params)

        loop = asyncio.get_running_loop()
        thread_future = _NEPTUNE_SIMILARITY_QUERY_EXECUTOR.submit(
            self._run_query, cypher_query_, params
        )
        lease.hold_until_complete(thread_future)
        query_future = asyncio.wrap_future(thread_future, loop=loop)
        try:
            return await query_future
        except asyncio.CancelledError:
            thread_future.add_done_callback(_consume_detached_neptune_query)
            raise

    def _run_query(self, cypher_query_, params):
        cypher_query_ = str(self._sanitize_parameters(cypher_query_, params))
        try:
            result = self.client.query(cypher_query_, params=params)
        except Exception as e:
            logger.error('Query: %s', cypher_query_)
            logger.error('Parameters: %s', params)
            logger.error('Error executing query: %s', e)
            raise e

        return result, None, None

    def session(self, database: str | None = None) -> GraphDriverSession:
        return NeptuneDriverSession(driver=self)

    async def close(self) -> None:
        return self.client.client.close()

    async def _delete_all_data(self) -> Any:
        return await self.execute_query(
            'MATCH (n) WITH n ORDER BY n.uuid '
            'SET n._opr_conditional_delete_lock = true '
            'REMOVE n._opr_conditional_delete_lock '
            'WITH n WHERE coalesce(n.opr_deleted, false) = false DETACH DELETE n'
        )

    def delete_all_indexes(self) -> Coroutine[Any, Any, Any]:
        return self.delete_all_indexes_impl()

    async def delete_all_indexes_impl(self) -> Coroutine[Any, Any, Any]:
        # No matter what happens above, always return True
        return self.delete_aoss_indices()

    def _create_aoss_index(self, index: dict[str, Any]) -> None:
        index_name = index['index_name']
        client = self.aoss_client
        if client.indices.exists(index=index_name):
            return
        try:
            client.indices.create(index=index_name, body=index['body'])
        except Exception as e:
            if _mapping_has_knn_vector(index['body']):
                raise VectorIndexUnsupportedError(
                    f"OpenSearch host '{self._aoss_host}' rejected creating vector index "
                    f"'{index_name}': {e}. If this is an illegal_argument or "
                    'mapper_parsing error for the knn_vector field, the AOSS collection '
                    'must be of type VECTORSEARCH; a SEARCH type collection rejects '
                    'knn_vector mappings.'
                ) from e
            raise

    async def create_aoss_indices(self):
        for index in aoss_indices:
            self._create_aoss_index(index)
        # Sleep for 1 minute to let the index creation complete
        await asyncio.sleep(60)

    async def create_vector_aoss_indices(self) -> None:
        """Create (or confirm existing) node_name_embedding and edge_fact_embedding only.

        Used to check whether the AOSS collection accepts knn_vector fields without
        creating the four text indexes or waiting for index propagation.
        """
        for index in vector_aoss_indices:
            self._create_aoss_index(index)

    async def delete_aoss_indices(self):
        for index in aoss_indices:
            index_name = index['index_name']
            client = self.aoss_client
            if client.indices.exists(index=index_name):
                client.indices.delete(index=index_name)

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        # Neptune uses OpenSearch (AOSS) for indexing
        if delete_existing:
            await self.delete_aoss_indices()
        await self.create_aoss_indices()

    async def _execute_aoss_search(self, index_name: str, body: dict[str, Any]) -> dict[str, Any]:
        """Run a search body on the bounded AOSS executor and return the raw response.

        Shared by full-text (multi_match) and k-NN queries so both stay behind the
        same capacity limiter and executor.
        """
        lease = await _AOSS_QUERY_CAPACITY.acquire()
        loop = asyncio.get_running_loop()
        try:
            thread_future = _AOSS_QUERY_EXECUTOR.submit(
                partial(
                    self.aoss_client.search,
                    body=body,
                    index=index_name,
                ),
            )
        except BaseException:
            lease.release()
            raise
        search_future = asyncio.wrap_future(thread_future, loop=loop)

        # Cancelling the waiter cannot stop a running thread. The request is read-only and
        # its body is per-call. Keep the slot until abandoned work actually finishes.
        try:
            response = await asyncio.shield(search_future)
        except asyncio.CancelledError:
            search_future.add_done_callback(_consume_aoss_wrapper_result)
            thread_future.add_done_callback(partial(_finish_detached_aoss_query, lease=lease))
            raise
        except BaseException:
            lease.release()
            raise
        else:
            lease.release()
            return response

    async def run_aoss_query(self, name: str, query_text: str, limit: int = 10) -> dict[str, Any]:
        for index in aoss_indices:
            if name.lower() == index['index_name']:
                query = deepcopy(index['query'])
                query['query']['multi_match']['query'] = query_text
                query['size'] = max(0, min(limit, MAX_AOSS_QUERY_SIZE))
                return await self._execute_aoss_search(index['index_name'], query)
        return {}

    async def run_aoss_knn_query(
        self,
        name: str,
        vector: list[float],
        limit: int,
        min_score: float,
        group_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a k-NN similarity query against a vector index.

        Returns candidates as ``{'id': uuid, 'score': cosine_similarity}``, filtered to
        scores above ``min_score`` and ordered by descending score. Raises RuntimeError
        naming the index and the backfill command when the index does not exist. An
        existing but empty index returns no results.
        """
        size = max(0, min(limit, MAX_AOSS_QUERY_SIZE))
        knn_clause: dict[str, Any] = {'vector': vector, 'k': size}
        if group_ids is not None:
            knn_clause['filter'] = {'terms': {'group_id': group_ids}}
        body = {
            'size': size,
            '_source': ['uuid'],
            'query': {'knn': {VECTOR_EMBEDDING_FIELD: knn_clause}},
        }

        try:
            response = await self._execute_aoss_search(name, body)
        except NotFoundError as e:
            raise RuntimeError(
                f"OpenSearch vector index '{name}' does not exist. Create it by calling "
                'NeptuneDriver.build_indices_and_constraints, then backfill existing '
                'embeddings with `python -m graph_service.backfill_embeddings '
                f'--group-id <group_id>`. ({e})'
            ) from e

        scored: list[dict[str, Any]] = []
        for hit in response['hits']['hits']:
            cosine = cosine_similarity_from_knn_score(hit['_score'])
            if cosine > min_score:
                scored.append({'id': hit['_source']['uuid'], 'score': cosine})
        scored.sort(key=lambda item: item['score'], reverse=True)
        return scored

    def save_to_aoss(self, name: str, data: list[dict]) -> int:
        for index in aoss_indices:
            if name.lower() == index['index_name']:
                to_index = []
                for d in data:
                    item = {'_index': name, '_id': d['uuid']}
                    if '_version' in d:
                        item['_version'] = d['_version']
                        item['_version_type'] = 'external_gte'
                    for p in index['body']['mappings']['properties']:
                        if p in d:
                            item[p] = d[p]
                    to_index.append(item)
                try:
                    success, failed = helpers.bulk(self.aoss_client, to_index, stats_only=True)
                    return success
                except Exception as e:
                    logger.error('save_to_aoss failed for index %s: %s', name, e)
                    return 0

        return 0


class NeptuneDriverSession(GraphDriverSession):
    provider = GraphProvider.NEPTUNE

    def __init__(self, driver: NeptuneDriver):  # type: ignore[reportUnknownArgumentType]
        self.driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # No cleanup needed for Neptune, but method must exist
        pass

    async def close(self):
        # No explicit close needed for Neptune, but method must exist
        pass

    async def execute_write(self, func, *args, **kwargs):
        # Directly await the provided async function with `self` as the transaction/session
        return await func(self, *args, **kwargs)

    async def run(self, query: str | list, **kwargs: Any) -> Any:
        if isinstance(query, list):
            res = None
            for q in query:
                res = await self.driver.execute_query(q, **kwargs)
            return res
        else:
            return await self.driver.execute_query(str(query), **kwargs)
