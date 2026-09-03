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

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.neptune.projection_versions import (
    clear_projection_sync_pending,
    defer_cancellation_until_complete,
    reserve_projection_versions,
    validate_batch_size,
    validate_projection_attributes,
)
from graphiti_core.driver.operations.entity_node_ops import EntityNodeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.driver.record_parsers import (
    entity_node_from_neptune_record as entity_node_from_record,
)
from graphiti_core.errors import NodeGroupMismatchError, NodeNotFoundError
from graphiti_core.helpers import get_neptune_projection_versions, query_result_record_count
from graphiti_core.models.nodes.node_db_queries import (
    get_entity_node_return_query,
    get_entity_node_save_bulk_query,
    get_entity_node_save_query,
)
from graphiti_core.nodes import EntityNode

if TYPE_CHECKING:
    from graphiti_core.driver.neptune_driver import NeptuneDriver

logger = logging.getLogger(__name__)

_NODE_CANONICAL_PROPERTIES = frozenset(
    {
        'uuid',
        'name',
        'name_embedding',
        'group_id',
        'summary',
        'created_at',
        'labels',
    }
)


class NeptuneEntityNodeOperations(EntityNodeOperations):
    def __init__(self, driver: NeptuneDriver | None = None):
        self._driver = driver

    async def _sync_vector_projections(
        self,
        executor: QueryExecutor,
        nodes: list[EntityNode],
        batch_size: int,
        projection_versions: dict[str, int],
        tx: Transaction | None,
    ) -> None:
        if self._driver is None:
            return

        persisted = [node for node in nodes if node.uuid in projection_versions]
        if persisted:
            self._driver.save_to_aoss(
                'node_name_and_summary',
                [
                    {
                        'uuid': node.uuid,
                        'name': node.name,
                        'summary': node.summary,
                        'group_id': node.group_id,
                    }
                    for node in persisted
                ],
            )
        if getattr(self._driver, 'vector_projection_enabled', True) is False:
            # Keep the exact graph generation pending so enabling projections can repair every
            # write that occurred while the rollout was disabled.
            return
        for i in range(0, len(persisted), batch_size):
            chunk = persisted[i : i + batch_size]
            documents: list[dict[str, Any]] = []
            for node in chunk:
                document: dict[str, Any] = {
                    'uuid': node.uuid,
                    'group_id': node.group_id,
                    '_version': projection_versions[node.uuid],
                }
                if node.name_embedding is not None:
                    document['embedding'] = node.name_embedding
                documents.append(document)
            indexed = await self._driver.save_vector_to_aoss_async('node_name_embedding', documents)
            if indexed != len(documents):
                raise RuntimeError(
                    'OpenSearch node vector projection is incomplete: '
                    f'indexed {indexed}/{len(documents)} documents'
                )
            await clear_projection_sync_pending(
                executor,
                'node',
                {node.uuid: projection_versions[node.uuid] for node in chunk},
                tx,
                batch_size=batch_size,
            )

    @defer_cancellation_until_complete
    async def save(
        self,
        executor: QueryExecutor,
        node: EntityNode,
        tx: Transaction | None = None,
    ) -> None:
        validate_projection_attributes(node.attributes, _NODE_CANONICAL_PROPERTIES)
        entity_data: dict[str, Any] = {
            'uuid': node.uuid,
            'name': node.name,
            'name_embedding': node.name_embedding,
            'group_id': node.group_id,
            'summary': node.summary,
            'created_at': node.created_at,
        }
        entity_data.update(node.attributes or {})
        labels = ':'.join(list(set(node.labels + ['Entity'])))

        query = get_entity_node_save_query(GraphProvider.NEPTUNE, labels)

        if tx is not None:
            result = await tx.run(query, entity_data=entity_data)
        else:
            result = await executor.execute_query(query, entity_data=entity_data)
        if await query_result_record_count(result) != 1:
            raise NodeGroupMismatchError()

        projection_versions = get_neptune_projection_versions(result)
        await self._sync_vector_projections(
            executor,
            [node],
            batch_size=1,
            projection_versions=projection_versions,
            tx=tx,
        )

        logger.debug(f'Saved Node to Graph: {node.uuid}')

    @defer_cancellation_until_complete
    async def save_bulk(
        self,
        executor: QueryExecutor,
        nodes: list[EntityNode],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        validate_batch_size(batch_size)
        for node in nodes:
            validate_projection_attributes(node.attributes, _NODE_CANONICAL_PROPERTIES)

        # A single UUID represents one materialized projection. Collapse duplicate caller
        # entries before either store is mutated so Neptune and OpenSearch deterministically
        # observe the same last-write-wins value and generation.
        unique_nodes = list({node.uuid: node for node in nodes}.values())
        prepared: list[dict[str, Any]] = []
        for node in unique_nodes:
            entity_data: dict[str, Any] = {
                'uuid': node.uuid,
                'name': node.name,
                'group_id': node.group_id,
                'summary': node.summary,
                'created_at': node.created_at,
                'name_embedding': node.name_embedding,
                'labels': list(set(node.labels + ['Entity'])),
            }
            entity_data.update(node.attributes or {})
            prepared.append(entity_data)

        if not prepared:
            return

        queries = get_entity_node_save_bulk_query(GraphProvider.NEPTUNE, prepared)

        # Neptune bakes each node's labels into its query text (openCypher has
        # no equivalent to Neo4j's `SET n:$(node.labels)`), so query text is a
        # pure function of a node's label set. Group nodes by that query text
        # so a node only gets UNWOUND through the query built for its own
        # labels, then chunk each group so a single request's payload stays
        # bounded regardless of how many nodes share a label combination.
        grouped: dict[str, list[dict[str, Any]]] = {}
        for query, node_data in zip(queries, prepared, strict=True):
            grouped.setdefault(query, []).append(node_data)

        projection_versions: dict[str, int] = {}
        for query, group_nodes in grouped.items():
            for i in range(0, len(group_nodes), batch_size):
                chunk = group_nodes[i : i + batch_size]
                if tx is not None:
                    result = await tx.run(query, nodes=chunk)
                else:
                    result = await executor.execute_query(query, nodes=chunk)
                projection_versions.update(get_neptune_projection_versions(result))

        if set(projection_versions) != {node.uuid for node in unique_nodes}:
            raise NodeGroupMismatchError()

        await self._sync_vector_projections(
            executor,
            unique_nodes,
            batch_size=batch_size,
            projection_versions=projection_versions,
            tx=tx,
        )

    @defer_cancellation_until_complete
    async def delete(
        self,
        executor: QueryExecutor,
        node: EntityNode,
        tx: Transaction | None = None,
    ) -> None:
        await self._delete_uuids(executor, [node.uuid], tx=tx, batch_size=100)
        logger.debug(f'Deleted Node: {node.uuid}')

    @staticmethod
    def _record_uuids(result: Any) -> list[str]:
        records = result[0] if isinstance(result, tuple) else result
        if not isinstance(records, list):
            return []
        return list(
            dict.fromkeys(
                record['uuid']
                for record in records
                if isinstance(record, dict) and isinstance(record.get('uuid'), str)
            )
        )

    async def _run_query(
        self,
        executor: QueryExecutor,
        query: str,
        tx: Transaction | None,
        **kwargs: Any,
    ) -> Any:
        if tx is not None:
            return await tx.run(query, **kwargs)
        return await executor.execute_query(query, **kwargs)

    async def _delete_incident_edges(
        self,
        executor: QueryExecutor,
        deletions: list[dict[str, Any]],
        tx: Transaction | None,
        batch_size: int,
    ) -> None:
        # Marking every node pending takes the same endpoint write locks used by edge saves.
        # New incident edges are therefore rejected while these bounded pages are drained.
        query = """
            UNWIND $deletions AS deletion
            MATCH (n:Entity {uuid: deletion.uuid})-[e:RELATES_TO]-()
            WHERE coalesce(n._graphiti_vector_delete_pending, false) = true
              AND n._graphiti_projection_version = deletion.projection_version
            RETURN DISTINCT e.uuid AS uuid
            ORDER BY uuid
            LIMIT $batch_size
        """
        from graphiti_core.driver.neptune.operations.entity_edge_ops import (
            NeptuneEntityEdgeOperations,
        )

        edge_ops = NeptuneEntityEdgeOperations(self._driver)
        while True:
            result = await self._run_query(
                executor,
                query,
                tx,
                deletions=deletions,
                batch_size=batch_size,
            )
            incident_edge_uuids = self._record_uuids(result)
            if not incident_edge_uuids:
                return
            await edge_ops.delete_by_uuids(
                executor,
                incident_edge_uuids,
                tx=tx,
                batch_size=batch_size,
            )

    async def _delete_chunk(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        tx: Transaction | None,
        batch_size: int,
    ) -> None:
        node_versions = await reserve_projection_versions(
            executor,
            'node',
            uuids,
            tx,
            batch_size=batch_size,
        )
        deletions = [{'uuid': uuid, 'projection_version': node_versions[uuid]} for uuid in uuids]
        prepare_query = """
            UNWIND $deletions AS deletion
            MATCH (n:Entity {uuid: deletion.uuid})
            SET n._graphiti_endpoint_lock = true
            REMOVE n._graphiti_endpoint_lock
            WITH n, deletion
            WHERE coalesce(n._graphiti_projection_version, 0) < deletion.projection_version
            SET n._graphiti_vector_delete_pending = true
            SET n._graphiti_projection_version = deletion.projection_version
            RETURN n.uuid AS uuid
        """
        await self._run_query(executor, prepare_query, tx, deletions=deletions)
        await self._delete_incident_edges(executor, deletions, tx, batch_size)

        if self._driver is not None:
            await self._driver.delete_from_aoss_async(
                'node_name_embedding',
                uuids=uuids,
                versions=node_versions,
            )

        finalize_query = """
            UNWIND $deletions AS deletion
            MATCH (n:Entity)
            WHERE n.uuid = deletion.uuid
              AND coalesce(n._graphiti_vector_delete_pending, false) = true
              AND n._graphiti_projection_version = deletion.projection_version
            DETACH DELETE n
        """
        await self._run_query(executor, finalize_query, tx, deletions=deletions)

    async def _delete_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        tx: Transaction | None,
        batch_size: int,
    ) -> None:
        validate_batch_size(batch_size)
        unique_uuids = list(dict.fromkeys(uuids))
        for start in range(0, len(unique_uuids), batch_size):
            await self._delete_chunk(
                executor,
                unique_uuids[start : start + batch_size],
                tx,
                batch_size,
            )

    @defer_cancellation_until_complete
    async def delete_by_group_id(
        self,
        executor: QueryExecutor,
        group_id: str,
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        validate_batch_size(batch_size)
        query = """
            MATCH (n:Entity {group_id: $group_id})
            RETURN n.uuid AS uuid
            ORDER BY uuid
            LIMIT $batch_size
        """
        while True:
            result = await self._run_query(
                executor,
                query,
                tx,
                group_id=group_id,
                batch_size=batch_size,
            )
            uuids = self._record_uuids(result)
            if not uuids:
                return
            await self._delete_uuids(executor, uuids, tx, batch_size)

    @defer_cancellation_until_complete
    async def delete_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        await self._delete_uuids(executor, uuids, tx, batch_size)

    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
    ) -> EntityNode:
        query = """
            MATCH (n:Entity {uuid: $uuid})
            WHERE coalesce(n._graphiti_vector_delete_pending, false) = false
            RETURN
            """ + get_entity_node_return_query(GraphProvider.NEPTUNE)
        records, _, _ = await executor.execute_query(query, uuid=uuid)
        nodes = [entity_node_from_record(r) for r in records]
        if len(nodes) == 0:
            raise NodeNotFoundError(uuid)
        return nodes[0]

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[EntityNode]:
        query = """
            MATCH (n:Entity)
            WHERE n.uuid IN $uuids
              AND coalesce(n._graphiti_vector_delete_pending, false) = false
            RETURN
            """ + get_entity_node_return_query(GraphProvider.NEPTUNE)
        records, _, _ = await executor.execute_query(query, uuids=uuids)
        return [entity_node_from_record(r) for r in records]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EntityNode]:
        cursor_clause = 'AND n.uuid < $uuid' if uuid_cursor else ''
        limit_clause = 'LIMIT $limit' if limit is not None else ''
        query = (
            """
            MATCH (n:Entity)
            WHERE n.group_id IN $group_ids
              AND coalesce(n._graphiti_vector_delete_pending, false) = false
            """
            + cursor_clause
            + """
            RETURN
            """
            + get_entity_node_return_query(GraphProvider.NEPTUNE)
            + """
            ORDER BY n.uuid DESC
            """
            + limit_clause
        )
        records, _, _ = await executor.execute_query(
            query,
            group_ids=group_ids,
            uuid=uuid_cursor,
            limit=limit,
        )
        return [entity_node_from_record(r) for r in records]

    async def load_embeddings(
        self,
        executor: QueryExecutor,
        node: EntityNode,
    ) -> None:
        query = """
            MATCH (n:Entity {uuid: $uuid})
            WHERE coalesce(n._graphiti_vector_delete_pending, false) = false
            RETURN [x IN split(n.name_embedding, ",") | toFloat(x)] AS name_embedding
        """
        records, _, _ = await executor.execute_query(query, uuid=node.uuid)
        if len(records) == 0:
            raise NodeNotFoundError(node.uuid)
        node.name_embedding = records[0]['name_embedding']

    async def load_embeddings_bulk(
        self,
        executor: QueryExecutor,
        nodes: list[EntityNode],
        batch_size: int = 100,
    ) -> None:
        uuids = [n.uuid for n in nodes]
        query = """
            MATCH (n:Entity)
            WHERE n.uuid IN $uuids
              AND coalesce(n._graphiti_vector_delete_pending, false) = false
            RETURN DISTINCT n.uuid AS uuid, [x IN split(n.name_embedding, ",") | toFloat(x)] AS name_embedding
        """
        embedding_map: dict[str, list[float]] = {}
        for i in range(0, len(uuids), batch_size):
            chunk = uuids[i : i + batch_size]
            records, _, _ = await executor.execute_query(query, uuids=chunk)
            embedding_map.update({r['uuid']: r['name_embedding'] for r in records})
        for node in nodes:
            if node.uuid in embedding_map:
                node.name_embedding = embedding_map[node.uuid]
