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
from collections.abc import Callable, Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from functools import partial
from typing import Any, TypeVar

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
AOSS_MUTATION_CONCURRENCY = 2
AOSS_MUTATION_MAX_WAITERS = AOSS_MUTATION_CONCURRENCY * 16
MAX_AOSS_QUERY_SIZE = 1000

_AOSS_MUTATION_RESULT = TypeVar('_AOSS_MUTATION_RESULT')

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

# Projection writes and lifecycle cleanup use the synchronous OpenSearch client. Keep that work
# off the event loop and behind a bounded queue. A separate pool prevents slow mutation traffic
# from consuming the read capacity used by similarity and full-text queries.
_AOSS_MUTATION_EXECUTOR = ThreadPoolExecutor(
    max_workers=AOSS_MUTATION_CONCURRENCY,
    thread_name_prefix='graphiti-aoss-mutation',
)
_AOSS_MUTATION_CAPACITY = AsyncCapacityLimiter(
    AOSS_MUTATION_CONCURRENCY,
    max_waiters=AOSS_MUTATION_MAX_WAITERS,
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

text_aoss_indices = [
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
VECTOR_AOSS_TOMBSTONE_FIELD = 'projection_deleted'
# Vector documents use a graph-committed generation above this epoch. Saves, clears, and deletion
# tombstones all carry the generation reserved in Neptune's durable projection-version ledger, so
# late work cannot overwrite a newer graph state and UUID reuse continues at a higher generation.
VECTOR_AOSS_WRITE_VERSION = 2_000_000_000_000_000
VECTOR_AOSS_VERSION_CEILING = 3_000_000_000_000_000
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


class VectorIndexConfigurationError(VectorIndexUnsupportedError):
    """Raised when an existing vector index does not match the required contract."""


class AossProjectionError(RuntimeError):
    """Raised when a required AOSS materialized-view update is not durable."""


def _mapping_has_knn_vector(body: dict[str, Any]) -> bool:
    properties = body.get('mappings', {}).get('properties', {})
    return any(prop.get('type') == 'knn_vector' for prop in properties.values())


def _vector_index_body(dimension: int) -> dict[str, Any]:
    if dimension <= 0:
        raise ValueError('Vector index dimension must be greater than zero')
    return {
        'settings': {'index': {'knn': True}},
        'mappings': {
            'properties': {
                'uuid': {'type': 'keyword'},
                'group_id': {'type': 'keyword'},
                VECTOR_AOSS_TOMBSTONE_FIELD: {'type': 'boolean'},
                VECTOR_EMBEDDING_FIELD: {
                    'type': 'knn_vector',
                    'dimension': dimension,
                    'method': {
                        'name': VECTOR_INDEX_METHOD,
                        'engine': VECTOR_INDEX_ENGINE,
                        'space_type': VECTOR_INDEX_SPACE_TYPE,
                    },
                },
            }
        },
    }


# knn_vector indexes for Neptune similarity search. OpenSearch cannot enable
# `index.knn` on an existing index, so these are separate indexes from the
# four text indexes above rather than added fields on them.
def _vector_aoss_indices(dimension: int) -> list[dict[str, Any]]:
    return [
        {
            'index_name': 'node_name_embedding',
            'body': _vector_index_body(dimension),
        },
        {
            'index_name': 'edge_fact_embedding',
            'body': _vector_index_body(dimension),
        },
    ]


vector_aoss_indices = _vector_aoss_indices(EMBEDDING_DIM)

aoss_indices = text_aoss_indices + vector_aoss_indices


def cosine_similarity_from_knn_score(score: float) -> float:
    """Convert an OpenSearch k-NN score back to cosine similarity.

    The vector indexes use the cosinesimil space with the faiss engine. OpenSearch
    defines ``score = (2 - d) / 2`` and ``d = 1 - cosine_similarity``, so the
    inverse is ``cosine_similarity = 2 * score - 1``.
    """
    if score <= 0:
        return -1.0
    return 2.0 * score - 1.0


def vector_aoss_external_version(generation: int) -> int:
    """Map a Neptune projection generation into OpenSearch's external-version range."""
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
        raise ValueError('Vector projection generation must be a non-negative integer')
    external_version = VECTOR_AOSS_WRITE_VERSION + generation
    if external_version >= VECTOR_AOSS_VERSION_CEILING:
        raise ValueError(f'Vector projection generation {generation} exceeds the supported range')
    return external_version


def _is_bulk_version_conflict(
    error: object,
    *,
    versioned_ids: set[str] | None = None,
) -> bool:
    """Return true only for an external-version conflict on a versioned bulk action."""
    if not isinstance(error, dict):
        return False
    detail: object = error
    for operation in ('index', 'create', 'update', 'delete'):
        candidate = error.get(operation)
        if isinstance(candidate, dict):
            detail = candidate
            break
    if not isinstance(detail, dict) or detail.get('status') != 409:
        return False
    if versioned_ids is not None and str(detail.get('_id')) not in versioned_ids:
        return False

    cause = detail.get('error')
    if isinstance(cause, dict):
        if cause.get('type') == 'version_conflict_engine_exception':
            return True
        caused_by = cause.get('caused_by')
        return isinstance(caused_by, dict) and (
            caused_by.get('type') == 'version_conflict_engine_exception'
        )
    return isinstance(cause, str) and 'version_conflict_engine_exception' in cause


def _sanitize_bulk_failures(errors: list[object]) -> list[dict[str, object]]:
    """Keep bulk diagnostics useful without logging indexed document bodies or error reasons."""
    sanitized: list[dict[str, object]] = []
    for error in errors:
        if not isinstance(error, dict):
            sanitized.append({'error_type': type(error).__name__})
            continue

        operation = next(
            (
                candidate
                for candidate in ('index', 'create', 'update', 'delete')
                if isinstance(error.get(candidate), dict)
            ),
            'unknown',
        )
        detail = error.get(operation)
        if not isinstance(detail, dict):
            sanitized.append({'operation': operation, 'error_type': 'malformed_bulk_error'})
            continue

        item: dict[str, object] = {'operation': operation}
        if '_id' in detail:
            item['document_id'] = str(detail['_id'])
        if isinstance(detail.get('status'), int):
            item['status'] = detail['status']
        cause = detail.get('error')
        if isinstance(cause, dict) and isinstance(cause.get('type'), str):
            item['error_type'] = cause['type']
        elif cause is not None:
            item['error_type'] = type(cause).__name__
        sanitized.append(item)
    return sanitized


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

    def __init__(
        self,
        host: str,
        aoss_host: str,
        port: int = 8182,
        aoss_port: int = 443,
        embedding_dim: int = EMBEDDING_DIM,
        vector_search_enabled: bool = False,
        vector_projection_enabled: bool | None = None,
        vector_aoss_host: str | None = None,
        vector_aoss_port: int | None = None,
    ):
        """This initializes a NeptuneDriver for use with Neptune as a backend

        Args:
            host (str): The Neptune Database or Neptune Analytics host
            aoss_host (str): The OpenSearch host value
            port (int, optional): The Neptune Database port, ignored for Neptune Analytics. Defaults to 8182.
            aoss_port (int, optional): The OpenSearch port. Defaults to 443.
            embedding_dim (int, optional): Dimension emitted by the configured embedder. The
                vector-index mapping must use the same value. Defaults to EMBEDDING_DIM.
            vector_search_enabled (bool, optional): Read similarity results from vector indexes.
            vector_projection_enabled (bool | None, optional): Create and maintain vector
                projection indexes. Defaults to vector_search_enabled for backwards compatibility.
                Staged rollouts should enable projections before backfill, then enable search.
            vector_aoss_host (str | None, optional): A separate VECTORSEARCH collection host.
                Defaults to aoss_host when the text collection itself supports vectors.
            vector_aoss_port (int | None, optional): Port for vector_aoss_host. Defaults to
                aoss_port.
        """
        self._database = 'default'

        if embedding_dim <= 0:
            raise ValueError('embedding_dim must be greater than zero')
        self.embedding_dim = embedding_dim
        self.vector_search_enabled = vector_search_enabled
        self.vector_projection_enabled = (
            vector_search_enabled
            if vector_projection_enabled is None
            else vector_projection_enabled
        )
        if self.vector_search_enabled and not self.vector_projection_enabled:
            raise ValueError('vector_search_enabled requires vector_projection_enabled')
        self._vector_aoss_indices = _vector_aoss_indices(embedding_dim)
        self._aoss_indices = text_aoss_indices + self._vector_aoss_indices

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
        self._vector_aoss_host = vector_aoss_host or aoss_host
        resolved_vector_aoss_port = vector_aoss_port or aoss_port

        session = boto3.Session()
        signer = Urllib3AWSV4SignerAuth(session.get_credentials(), session.region_name, 'aoss')

        def build_aoss_client(endpoint_host: str, endpoint_port: int) -> OpenSearch:
            return OpenSearch(
                hosts=[{'host': endpoint_host, 'port': endpoint_port}],
                http_auth=signer,
                use_ssl=True,
                verify_certs=True,
                connection_class=Urllib3HttpConnection,
                pool_maxsize=20,
            )

        self.aoss_client = build_aoss_client(aoss_host, aoss_port)
        self.vector_aoss_client = (
            self.aoss_client
            if self._vector_aoss_host == aoss_host and resolved_vector_aoss_port == aoss_port
            else build_aoss_client(self._vector_aoss_host, resolved_vector_aoss_port)
        )

        # Instantiate Neptune operations
        self._entity_node_ops = NeptuneEntityNodeOperations(driver=self)
        self._episode_node_ops = NeptuneEpisodeNodeOperations()
        self._community_node_ops = NeptuneCommunityNodeOperations(driver=self)
        self._saga_node_ops = NeptuneSagaNodeOperations()
        self._entity_edge_ops = NeptuneEntityEdgeOperations(driver=self)
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

    def _configured_aoss_indices(self) -> list[dict[str, Any]]:
        # Some focused tests construct a driver with object.__new__. Keep those tests and
        # downstream subclasses compatible while production instances use their configured
        # embedding dimension.
        return getattr(self, '_aoss_indices', aoss_indices)

    def _configured_vector_aoss_indices(self) -> list[dict[str, Any]]:
        return getattr(self, '_vector_aoss_indices', vector_aoss_indices)

    def _vector_client(self) -> OpenSearch:
        return getattr(self, 'vector_aoss_client', self.aoss_client)

    def _vector_host(self) -> str:
        return getattr(self, '_vector_aoss_host', self._aoss_host)

    def _client_for_index(self, index: dict[str, Any]) -> OpenSearch:
        return self._vector_client() if _mapping_has_knn_vector(index['body']) else self.aoss_client

    @staticmethod
    def _is_vector_mapping_rejection(error: Exception) -> bool:
        message = str(error).lower()
        mentions_vector_mapping = 'knn_vector' in message or 'index.knn' in message
        is_mapping_error = any(
            marker in message
            for marker in (
                'illegal_argument',
                'mapper_parsing',
                'mapping',
                'not supported',
                'unknown field',
                'validation failed',
            )
        )
        return mentions_vector_mapping and is_mapping_error

    @staticmethod
    def _response_index_config(
        response: dict[str, Any], index_name: str, response_name: str
    ) -> dict[str, Any]:
        if index_name in response:
            return response[index_name]
        if len(response) == 1:
            return next(iter(response.values()))
        raise VectorIndexConfigurationError(
            f"OpenSearch returned no {response_name} for vector index '{index_name}'"
        )

    def _validate_vector_aoss_index(self, index: dict[str, Any]) -> None:
        """Validate an existing vector index instead of trusting its name."""
        index_name = index['index_name']
        expected_properties = index['body']['mappings']['properties']
        expected_embedding = expected_properties[VECTOR_EMBEDDING_FIELD]

        client = self._vector_client()
        mapping_response = client.indices.get_mapping(index=index_name)
        mapping_config = self._response_index_config(mapping_response, index_name, 'mapping')
        actual_properties = mapping_config.get('mappings', {}).get('properties', {})
        actual_embedding = actual_properties.get(VECTOR_EMBEDDING_FIELD, {})
        actual_method = actual_embedding.get('method', {})
        actual_space = actual_method.get('space_type', actual_embedding.get('space_type'))

        settings_response = client.indices.get_settings(index=index_name)
        settings_config = self._response_index_config(settings_response, index_name, 'settings')
        knn_enabled = settings_config.get('settings', {}).get('index', {}).get('knn')

        mismatches: list[str] = []
        for field_name in ('uuid', 'group_id'):
            expected_type = expected_properties[field_name]['type']
            actual_type = actual_properties.get(field_name, {}).get('type')
            if actual_type != expected_type:
                mismatches.append(f'{field_name}.type={actual_type!r} (expected {expected_type!r})')
        if actual_embedding.get('type') != 'knn_vector':
            mismatches.append(
                f'{VECTOR_EMBEDDING_FIELD}.type={actual_embedding.get("type")!r} '
                "(expected 'knn_vector')"
            )
        if actual_embedding.get('dimension') != expected_embedding['dimension']:
            mismatches.append(
                f'{VECTOR_EMBEDDING_FIELD}.dimension='
                f'{actual_embedding.get("dimension")!r} '
                f'(expected {expected_embedding["dimension"]!r})'
            )
        if actual_method.get('name') != VECTOR_INDEX_METHOD:
            mismatches.append(
                f'{VECTOR_EMBEDDING_FIELD}.method.name={actual_method.get("name")!r} '
                f'(expected {VECTOR_INDEX_METHOD!r})'
            )
        if actual_method.get('engine') != VECTOR_INDEX_ENGINE:
            mismatches.append(
                f'{VECTOR_EMBEDDING_FIELD}.method.engine={actual_method.get("engine")!r} '
                f'(expected {VECTOR_INDEX_ENGINE!r})'
            )
        if actual_space != VECTOR_INDEX_SPACE_TYPE:
            mismatches.append(
                f'{VECTOR_EMBEDDING_FIELD}.space_type={actual_space!r} '
                f'(expected {VECTOR_INDEX_SPACE_TYPE!r})'
            )
        if str(knn_enabled).lower() != 'true':
            mismatches.append(f'index.knn={knn_enabled!r} (expected true)')

        if mismatches:
            raise VectorIndexConfigurationError(
                f"OpenSearch vector index '{index_name}' on host '{self._vector_host()}' is "
                f'incompatible: {"; ".join(mismatches)}. Recreate the vector index with '
                f'dimension {expected_embedding["dimension"]} before backfilling.'
            )

    def _create_aoss_index(self, index: dict[str, Any]) -> bool:
        index_name = index['index_name']
        is_vector_index = _mapping_has_knn_vector(index['body'])
        client = self._client_for_index(index)
        if client.indices.exists(index=index_name):
            if is_vector_index:
                self._validate_vector_aoss_index(index)
            return False
        try:
            client.indices.create(index=index_name, body=index['body'])
        except Exception as e:
            message = str(e).lower()
            if is_vector_index and 'resource_already_exists' in message:
                self._validate_vector_aoss_index(index)
                return False
            if is_vector_index and self._is_vector_mapping_rejection(e):
                raise VectorIndexUnsupportedError(
                    f"OpenSearch host '{self._vector_host()}' rejected creating vector index "
                    f"'{index_name}' (error_type={type(e).__name__}). The AOSS collection must "
                    'be of type VECTORSEARCH; a SEARCH type collection rejects knn_vector '
                    'mappings.'
                ) from None
            raise
        return True

    async def create_aoss_indices(self):
        indices = (
            self._configured_aoss_indices() if self.vector_projection_enabled else text_aoss_indices
        )
        for index in indices:
            self._create_aoss_index(index)
        # Sleep for 1 minute to let the index creation complete
        await asyncio.sleep(60)

    async def create_vector_aoss_indices(self, wait_for_propagation: bool = False) -> None:
        """Create (or confirm existing) node_name_embedding and edge_fact_embedding only.

        Used to check whether the AOSS collection accepts knn_vector fields without creating the
        four text indexes. A reset/backfill caller must request the propagation wait before writes.
        """
        created = False
        for index in self._configured_vector_aoss_indices():
            created = self._create_aoss_index(index) or created
        if created and wait_for_propagation:
            await asyncio.sleep(60)

    async def _delete_configured_aoss_indices(self, indices: list[dict[str, Any]]) -> None:
        for index in indices:
            index_name = index['index_name']
            client = self._client_for_index(index)
            if client.indices.exists(index=index_name):
                client.indices.delete(index=index_name)

    async def delete_aoss_indices(self) -> None:
        """Delete indexes managed by the active rollout configuration.

        A process with vector projections disabled must not delete indexes that are actively
        maintained by a projection-enabled deployment during a staged rollout. Explicit vector
        rollback/cleanup uses :meth:`delete_vector_aoss_indices` instead.
        """
        indices = (
            self._configured_aoss_indices()
            if getattr(self, 'vector_projection_enabled', True)
            else text_aoss_indices
        )
        await self._delete_configured_aoss_indices(indices)

    async def delete_text_aoss_indices(self) -> None:
        """Delete only full-text indexes, preserving versioned vector projections."""
        await self._delete_configured_aoss_indices(text_aoss_indices)

    async def delete_vector_aoss_indices(self) -> None:
        """Explicitly delete the two vector indexes during rollback or re-provisioning."""
        await self._delete_configured_aoss_indices(self._configured_vector_aoss_indices())

    def purge_vector_aoss_group_documents(self, group_id: str) -> int:
        """Hard-delete one group's vectors during an explicitly quiesced repair."""
        if not group_id:
            raise ValueError('group_id must be non-empty')
        deleted = 0
        query = {'term': {'group_id': group_id}}
        for index in self._configured_vector_aoss_indices():
            deleted += self._purge_aoss_query(index['index_name'], query)
        return deleted

    async def purge_vector_aoss_group_documents_async(self, group_id: str) -> int:
        """Run the explicitly quiesced group purge without blocking the event loop."""
        return await self._execute_aoss_mutation(
            partial(self.purge_vector_aoss_group_documents, group_id)
        )

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        # Neptune uses OpenSearch (AOSS) for indexing
        if delete_existing:
            # Generic schema rebuilds cannot safely empty live vector indexes: rebuilding their
            # mappings also requires a quiesced, all-groups backfill. Preserve them here and leave
            # explicit vector resets to the guarded backfill command.
            await self.delete_text_aoss_indices()
        await self.create_aoss_indices()

    async def _execute_aoss_mutation(
        self,
        operation: Callable[[], _AOSS_MUTATION_RESULT],
    ) -> _AOSS_MUTATION_RESULT:
        """Execute one synchronous AOSS mutation behind bounded, cancellation-safe capacity."""
        lease = await _AOSS_MUTATION_CAPACITY.acquire()
        try:
            thread_future = _AOSS_MUTATION_EXECUTOR.submit(operation)
            lease.hold_until_complete(thread_future)
        except BaseException:
            lease.release()
            raise

        # Request release now; the hold above delays it until the non-cancellable worker actually
        # finishes, including when the awaiting asyncio task is cancelled or its loop is closed.
        lease.release()
        loop = asyncio.get_running_loop()
        mutation_future = asyncio.wrap_future(thread_future, loop=loop)
        try:
            return await asyncio.shield(mutation_future)
        except asyncio.CancelledError:
            # The worker cannot be stopped, and the capacity lease remains held by
            # ``hold_until_complete``. Direct callers receive cancellation immediately; graph
            # lifecycle operations wrap their entire Neptune+AOSS boundary and defer it until
            # both sides are consistent.
            mutation_future.add_done_callback(_consume_aoss_wrapper_result)
            raise

    async def _execute_aoss_search(
        self,
        index_name: str,
        body: dict[str, Any],
        *,
        vector: bool = False,
    ) -> dict[str, Any]:
        """Run a search body on the bounded AOSS executor and return the raw response.

        Shared by full-text (multi_match) and k-NN queries so both stay behind the
        same capacity limiter and executor.
        """
        lease = await _AOSS_QUERY_CAPACITY.acquire()
        loop = asyncio.get_running_loop()
        client = self._vector_client() if vector else self.aoss_client
        try:
            thread_future = _AOSS_QUERY_EXECUTOR.submit(
                partial(
                    client.search,
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
        for index in self._configured_aoss_indices():
            if name.lower() == index['index_name']:
                query_template = index.get('query')
                if query_template is None:
                    return {}
                query = deepcopy(query_template)
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
        uuids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a k-NN similarity query against a vector index.

        Returns candidates as ``{'id': uuid, 'score': cosine_similarity}``, filtered to
        scores above ``min_score`` and ordered by descending score. Raises RuntimeError
        naming the index and the backfill command when the index does not exist. An
        existing but empty index returns no results.
        """
        if getattr(self, 'vector_search_enabled', True) is False:
            raise RuntimeError('Neptune vector search is disabled for this driver')
        size = max(0, min(limit, MAX_AOSS_QUERY_SIZE))
        if size == 0 or group_ids == [] or uuids == []:
            return []

        expected_dimension = getattr(self, 'embedding_dim', None)
        if expected_dimension is not None and len(vector) != expected_dimension:
            raise ValueError(
                f'Query vector has dimension {len(vector)}; expected {expected_dimension}'
            )

        knn_clause: dict[str, Any] = {'vector': vector, 'k': size}
        filters: list[dict[str, Any]] = []
        if group_ids is not None:
            filters.append({'terms': {'group_id': group_ids}})
        if uuids is not None:
            filters.append({'terms': {'uuid': uuids}})
        if len(filters) == 1:
            knn_clause['filter'] = filters[0]
        elif filters:
            knn_clause['filter'] = {'bool': {'filter': filters}}
        body = {
            'size': size,
            '_source': ['uuid'],
            'query': {'knn': {VECTOR_EMBEDDING_FIELD: knn_clause}},
        }

        try:
            response = await self._execute_aoss_search(name, body, vector=True)
        except NotFoundError as e:
            raise RuntimeError(
                f"OpenSearch vector index '{name}' does not exist. Create it by calling "
                'NeptuneDriver.build_indices_and_constraints, then backfill existing '
                'embeddings with `python -m graph_service.backfill_embeddings '
                '--all-groups --reset-vector-indices '
                f'--acknowledge-ingestion-and-deletion-quiesced`. ({e})'
            ) from e

        scored: list[dict[str, Any]] = []
        for hit in response['hits']['hits']:
            cosine = cosine_similarity_from_knn_score(hit['_score'])
            if cosine > min_score:
                scored.append({'id': hit['_source']['uuid'], 'score': cosine})
        scored.sort(key=lambda item: item['score'], reverse=True)
        return scored

    def save_to_aoss(self, name: str, data: list[dict]) -> int:
        for index in self._configured_aoss_indices():
            if name.lower() == index['index_name']:
                to_index = []
                expected_dimension = (
                    index['body']['mappings']['properties']
                    .get(VECTOR_EMBEDDING_FIELD, {})
                    .get('dimension')
                )
                for d in data:
                    embedding = d.get(VECTOR_EMBEDDING_FIELD)
                    is_projection_tombstone = d.get(VECTOR_AOSS_TOMBSTONE_FIELD) is True
                    if (
                        expected_dimension is not None
                        and not is_projection_tombstone
                        and (
                            not isinstance(embedding, list) or len(embedding) != expected_dimension
                        )
                    ):
                        logger.error(
                            'save_to_aoss rejected %s document %s with embedding dimension %s; '
                            'expected %s',
                            name,
                            d.get('uuid'),
                            len(embedding) if isinstance(embedding, list) else None,
                            expected_dimension,
                        )
                        return 0
                    item = {'_index': name, '_id': d['uuid']}
                    if '_version' in d:
                        item['_version'] = d['_version']
                        item['_version_type'] = d.get('_version_type', 'external_gte')
                    for p in index['body']['mappings']['properties']:
                        if p in d:
                            item[p] = d[p]
                    to_index.append(item)
                try:
                    client = self._client_for_index(index)
                    success, errors = helpers.bulk(
                        client,
                        to_index,
                        stats_only=False,
                        raise_on_error=False,
                        raise_on_exception=False,
                    )
                    if not isinstance(errors, list):
                        return int(success)

                    versioned_ids = {str(item['_id']) for item in to_index if '_version' in item}
                    superseded = 0
                    failures = []
                    for error in errors:
                        if _is_bulk_version_conflict(error, versioned_ids=versioned_ids):
                            superseded += 1
                        else:
                            failures.append(error)
                    if failures:
                        logger.error(
                            'save_to_aoss failed for index %s: %s',
                            name,
                            _sanitize_bulk_failures(failures),
                        )
                    return int(success) + superseded
                except Exception as e:
                    logger.error(
                        'save_to_aoss failed for index %s (error_type=%s)',
                        name,
                        type(e).__name__,
                    )
                    return 0

        return 0

    def save_vector_to_aoss(self, name: str, data: list[dict[str, Any]]) -> int:
        """Write a required vector projection or raise instead of hiding drift."""
        if not data:
            return 0
        if getattr(self, 'vector_projection_enabled', True) is False:
            return len(data)
        configured_names = {index['index_name'] for index in self._configured_vector_aoss_indices()}
        if name.lower() not in configured_names:
            raise ValueError(f"Unknown vector AOSS index '{name}'")

        versioned_documents: list[dict[str, Any]] = []
        for document in data:
            generation = document.get('_version')
            if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
                raise ValueError(
                    f"Vector projection '{name}' document {document.get('uuid')!r} requires "
                    'a non-negative graph-committed _version'
                )
            external_version = vector_aoss_external_version(generation)
            versioned_document = dict(document)
            versioned_document['_version'] = external_version
            # Strict external ordering prevents an equal-generation backfill replay from
            # replacing a deletion tombstone after graph finalization failed.
            versioned_document['_version_type'] = 'external'
            if versioned_document.get(VECTOR_EMBEDDING_FIELD) is None:
                versioned_document[VECTOR_AOSS_TOMBSTONE_FIELD] = True
            else:
                versioned_document.pop(VECTOR_AOSS_TOMBSTONE_FIELD, None)
            versioned_documents.append(versioned_document)

        success = self.save_to_aoss(name, versioned_documents)
        if success != len(data):
            raise AossProjectionError(
                f"Vector projection '{name}' indexed {success}/{len(data)} documents"
            )
        return success

    async def save_vector_to_aoss_async(self, name: str, data: list[dict[str, Any]]) -> int:
        """Asynchronously write a vector projection without blocking the event loop."""
        return await self._execute_aoss_mutation(partial(self.save_vector_to_aoss, name, data))

    def delete_from_aoss(
        self,
        name: str,
        *,
        uuids: list[str] | None = None,
        group_ids: list[str] | None = None,
        versions: dict[str, int] | None = None,
    ) -> int:
        """Delete vector projection documents by UUID and/or tenant group.

        Deletion is idempotent: a missing index or document counts as already deleted. Transport
        errors and partial task failures raise because retaining a derived embedding is both a
        search-correctness and data-lifecycle failure.
        """
        configured_names = {index['index_name'] for index in self._configured_vector_aoss_indices()}
        if name.lower() not in configured_names:
            raise ValueError(f"Unknown vector AOSS index '{name}'")
        if uuids is None and group_ids is None:
            raise ValueError('At least one of uuids or group_ids is required')
        if uuids == [] or group_ids == []:
            return 0
        if versions is None:
            raise ValueError('Graph-committed versions are required for vector projection deletion')

        filters: list[dict[str, Any]] = []
        if uuids is not None:
            filters.append({'terms': {'uuid': uuids}})
        if group_ids is not None:
            filters.append({'terms': {'group_id': group_ids}})
        query = filters[0] if len(filters) == 1 else {'bool': {'filter': filters}}

        if group_ids is None and uuids is not None:
            return self._delete_aoss_document_ids(name, uuids, versions)
        return self._delete_aoss_query(name, query, versions)

    async def delete_from_aoss_async(
        self,
        name: str,
        *,
        uuids: list[str] | None = None,
        group_ids: list[str] | None = None,
        versions: dict[str, int] | None = None,
    ) -> int:
        """Asynchronously delete vector projections without blocking the event loop."""
        return await self._execute_aoss_mutation(
            partial(
                self.delete_from_aoss,
                name,
                uuids=uuids,
                group_ids=group_ids,
                versions=versions,
            )
        )

    def delete_all_from_aoss(self, name: str) -> int:
        """Delete every document from one configured index while retaining its mapping."""
        if getattr(self, 'vector_projection_enabled', True) is False:
            return 0
        configured_names = {index['index_name'] for index in self._configured_vector_aoss_indices()}
        if name.lower() not in configured_names:
            raise ValueError(f"Unknown vector AOSS index '{name}'")
        raise ValueError(
            'delete_all_from_aoss requires an explicit vector-index reset so projection '
            'generations and documents remain aligned'
        )

    async def delete_all_from_aoss_async(self, name: str) -> int:
        """Asynchronously clear one vector projection without blocking the event loop."""
        return await self._execute_aoss_mutation(partial(self.delete_all_from_aoss, name))

    def _delete_aoss_query(
        self,
        name: str,
        query: dict[str, Any],
        versions: dict[str, int],
    ) -> int:
        """Search and delete matching AOSS documents in bounded, Serverless-safe pages."""
        client = self._vector_client()
        deleted = 0
        search_after: list[Any] | None = None
        while True:
            body: dict[str, Any] = {
                'size': MAX_AOSS_QUERY_SIZE,
                '_source': False,
                'query': query,
                'sort': [{'uuid': 'asc'}],
            }
            if search_after is not None:
                body['search_after'] = search_after
            try:
                response = client.search(index=name, body=body)
            except NotFoundError:
                return deleted
            except Exception as e:
                raise AossProjectionError(
                    f"Failed listing documents from vector projection '{name}' "
                    f'(error_type={type(e).__name__})'
                ) from None

            hits = response.get('hits', {}).get('hits', [])
            if not hits:
                break
            deleted += self._delete_aoss_document_ids(
                name,
                [str(hit['_id']) for hit in hits],
                versions,
            )
            if len(hits) < MAX_AOSS_QUERY_SIZE:
                break
            last_sort = hits[-1].get('sort')
            if not isinstance(last_sort, list) or not last_sort:
                raise AossProjectionError(
                    f"Vector projection '{name}' did not return search_after sort values"
                )
            search_after = last_sort
        return deleted

    def _purge_aoss_query(self, name: str, query: dict[str, Any]) -> int:
        """Hard-delete a quiesced repair scope, including stale AOSS-only documents."""
        client = self._vector_client()
        deleted = 0
        search_after: list[Any] | None = None
        while True:
            body: dict[str, Any] = {
                'size': MAX_AOSS_QUERY_SIZE,
                '_source': False,
                'query': query,
                'sort': [{'uuid': 'asc'}],
            }
            if search_after is not None:
                body['search_after'] = search_after
            try:
                response = client.search(index=name, body=body)
            except NotFoundError:
                return deleted
            except Exception as e:
                raise AossProjectionError(
                    f"Failed listing quiesced repair documents from vector projection '{name}' "
                    f'(error_type={type(e).__name__})'
                ) from None

            hits = response.get('hits', {}).get('hits', [])
            if not hits:
                return deleted
            actions = [
                {'_op_type': 'delete', '_index': name, '_id': str(hit['_id'])} for hit in hits
            ]
            try:
                success, errors = helpers.bulk(
                    client,
                    actions,
                    stats_only=False,
                    raise_on_error=False,
                    raise_on_exception=False,
                )
            except Exception as e:
                raise AossProjectionError(
                    f"Failed purging quiesced repair documents from vector projection '{name}' "
                    f'(error_type={type(e).__name__})'
                ) from None
            if errors:
                raise AossProjectionError(
                    f"Vector projection '{name}' quiesced repair purge reported failures: "
                    f'{_sanitize_bulk_failures(errors)}'
                )
            deleted += int(success)
            if len(hits) < MAX_AOSS_QUERY_SIZE:
                return deleted
            last_sort = hits[-1].get('sort')
            if not isinstance(last_sort, list) or not last_sort:
                raise AossProjectionError(
                    f"Vector projection '{name}' did not return repair search_after sort values"
                )
            search_after = last_sort

    def _delete_aoss_document_ids(
        self,
        name: str,
        document_ids: list[str],
        versions: dict[str, int],
    ) -> int:
        if not document_ids:
            return 0
        client = self._vector_client()
        try:
            if not client.indices.exists(index=name):
                return len(set(document_ids))
        except Exception as e:
            raise AossProjectionError(
                f"Failed checking vector projection '{name}' before deletion "
                f'(error_type={type(e).__name__})'
            ) from None
        deleted = 0
        for start in range(0, len(document_ids), MAX_AOSS_QUERY_SIZE):
            chunk = list(dict.fromkeys(document_ids[start : start + MAX_AOSS_QUERY_SIZE]))
            actions = []
            for document_id in chunk:
                if document_id not in versions:
                    raise AossProjectionError(
                        f"Vector projection '{name}' deletion is missing the graph generation "
                        f'for {document_id!r}'
                    )
                actions.append(
                    {
                        '_op_type': 'index',
                        '_index': name,
                        '_id': document_id,
                        '_version': vector_aoss_external_version(versions[document_id]),
                        '_version_type': 'external',
                        'uuid': document_id,
                        VECTOR_AOSS_TOMBSTONE_FIELD: True,
                    }
                )
            try:
                success, errors = helpers.bulk(
                    client,
                    actions,
                    stats_only=False,
                    raise_on_error=False,
                    raise_on_exception=False,
                )
            except Exception as e:
                raise AossProjectionError(
                    f"Failed deleting documents from vector projection '{name}' "
                    f'(error_type={type(e).__name__})'
                ) from None

            failures = []
            superseded = 0
            for error in errors:
                if _is_bulk_version_conflict(error, versioned_ids=set(chunk)):
                    superseded += 1
                else:
                    failures.append(error)
            if failures:
                raise AossProjectionError(
                    f"Vector projection '{name}' deletion reported failures: "
                    f'{_sanitize_bulk_failures(failures)}'
                )
            deleted += int(success) + superseded
        return deleted


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
