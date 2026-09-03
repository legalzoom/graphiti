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
from graphiti_core.driver.operations.entity_edge_ops import EntityEdgeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.driver.record_parsers import (
    entity_edge_from_neptune_record as entity_edge_from_record,
)
from graphiti_core.edges import EntityEdge
from graphiti_core.errors import EdgeNotFoundError, NodeGroupMismatchError
from graphiti_core.helpers import get_neptune_projection_versions
from graphiti_core.models.edges.edge_db_queries import (
    get_entity_edge_return_query,
    get_entity_edge_save_bulk_query,
    get_entity_edge_save_query,
)

if TYPE_CHECKING:
    from graphiti_core.driver.neptune_driver import NeptuneDriver

logger = logging.getLogger(__name__)

_EDGE_CANONICAL_PROPERTIES = frozenset(
    {
        'uuid',
        'source_uuid',
        'target_uuid',
        'source_node_uuid',
        'target_node_uuid',
        'name',
        'fact',
        'fact_embedding',
        'group_id',
        'episodes',
        'created_at',
        'expired_at',
        'valid_at',
        'invalid_at',
        'reference_time',
    }
)


class NeptuneEntityEdgeOperations(EntityEdgeOperations):
    def __init__(self, driver: NeptuneDriver | None = None):
        self._driver = driver

    async def _sync_vector_projections(
        self,
        executor: QueryExecutor,
        edges: list[EntityEdge],
        batch_size: int,
        projection_versions: dict[str, int],
        tx: Transaction | None,
    ) -> None:
        if self._driver is None:
            return

        persisted = [edge for edge in edges if edge.uuid in projection_versions]
        if persisted:
            self._driver.save_to_aoss(
                'edge_name_and_fact',
                [
                    {
                        'uuid': edge.uuid,
                        'name': edge.name,
                        'fact': edge.fact,
                        'group_id': edge.group_id,
                    }
                    for edge in persisted
                ],
            )
        if getattr(self._driver, 'vector_projection_enabled', True) is False:
            # Keep the exact graph generation pending so enabling projections can repair every
            # write that occurred while the rollout was disabled.
            return
        for i in range(0, len(persisted), batch_size):
            chunk = persisted[i : i + batch_size]
            documents: list[dict[str, Any]] = []
            for edge in chunk:
                document: dict[str, Any] = {
                    'uuid': edge.uuid,
                    'group_id': edge.group_id,
                    '_version': projection_versions[edge.uuid],
                }
                if edge.fact_embedding is not None:
                    document['embedding'] = edge.fact_embedding
                documents.append(document)
            indexed = await self._driver.save_vector_to_aoss_async('edge_fact_embedding', documents)
            if indexed != len(documents):
                raise RuntimeError(
                    'OpenSearch edge vector projection is incomplete: '
                    f'indexed {indexed}/{len(documents)} documents'
                )
            await clear_projection_sync_pending(
                executor,
                'edge',
                {edge.uuid: projection_versions[edge.uuid] for edge in chunk},
                tx,
                batch_size=batch_size,
            )

    @staticmethod
    def _persisted_edge_uuids(result: Any) -> list[str]:
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

    @defer_cancellation_until_complete
    async def save(
        self,
        executor: QueryExecutor,
        edge: EntityEdge,
        tx: Transaction | None = None,
    ) -> None:
        validate_projection_attributes(edge.attributes, _EDGE_CANONICAL_PROPERTIES)
        edge_data: dict[str, Any] = {
            'uuid': edge.uuid,
            'source_uuid': edge.source_node_uuid,
            'target_uuid': edge.target_node_uuid,
            'name': edge.name,
            'fact': edge.fact,
            'fact_embedding': edge.fact_embedding,
            'group_id': edge.group_id,
            'episodes': edge.episodes,
            'created_at': edge.created_at,
            'expired_at': edge.expired_at,
            'valid_at': edge.valid_at,
            'invalid_at': edge.invalid_at,
            'reference_time': edge.reference_time,
        }
        edge_data.update(edge.attributes or {})

        query = get_entity_edge_save_query(GraphProvider.NEPTUNE)
        if tx is not None:
            result = await tx.run(query, edge_data=edge_data)
        else:
            result = await executor.execute_query(query, edge_data=edge_data)

        projection_versions = get_neptune_projection_versions(result)
        if list(projection_versions) != [edge.uuid]:
            raise NodeGroupMismatchError()

        await self._sync_vector_projections(
            executor,
            [edge],
            batch_size=1,
            projection_versions=projection_versions,
            tx=tx,
        )

        logger.debug(f'Saved Edge to Graph: {edge.uuid}')

    @defer_cancellation_until_complete
    async def save_bulk(
        self,
        executor: QueryExecutor,
        edges: list[EntityEdge],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        validate_batch_size(batch_size)
        for edge in edges:
            validate_projection_attributes(edge.attributes, _EDGE_CANONICAL_PROPERTIES)

        # Keep graph and vector state aligned when callers supply the same UUID more than once.
        # Dict replacement retains the UUID's original position while selecting its last value.
        unique_edges = list({edge.uuid: edge for edge in edges}.values())
        prepared: list[dict[str, Any]] = []
        for edge in unique_edges:
            edge_data: dict[str, Any] = {
                'uuid': edge.uuid,
                'source_node_uuid': edge.source_node_uuid,
                'target_node_uuid': edge.target_node_uuid,
                'name': edge.name,
                'fact': edge.fact,
                'fact_embedding': edge.fact_embedding,
                'group_id': edge.group_id,
                'episodes': edge.episodes,
                'created_at': edge.created_at,
                'expired_at': edge.expired_at,
                'valid_at': edge.valid_at,
                'invalid_at': edge.invalid_at,
                'reference_time': edge.reference_time,
            }
            edge_data.update(edge.attributes or {})
            prepared.append(edge_data)

        if not prepared:
            return

        query = get_entity_edge_save_bulk_query(GraphProvider.NEPTUNE)
        projection_versions: dict[str, int] = {}
        for i in range(0, len(prepared), batch_size):
            chunk = prepared[i : i + batch_size]
            if tx is not None:
                result = await tx.run(query, entity_edges=chunk)
            else:
                result = await executor.execute_query(query, entity_edges=chunk)
            projection_versions.update(get_neptune_projection_versions(result))

        if set(projection_versions) != {edge.uuid for edge in unique_edges}:
            raise NodeGroupMismatchError()

        await self._sync_vector_projections(
            executor,
            unique_edges,
            batch_size=batch_size,
            projection_versions=projection_versions,
            tx=tx,
        )

    @defer_cancellation_until_complete
    async def delete(
        self,
        executor: QueryExecutor,
        edge: EntityEdge,
        tx: Transaction | None = None,
    ) -> None:
        await self._delete_uuids_chunk(executor, [edge.uuid], tx=tx)

        logger.debug(f'Deleted Edge: {edge.uuid}')

    async def _delete_uuids_chunk(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        tx: Transaction | None = None,
    ) -> None:
        versions = await reserve_projection_versions(
            executor,
            'edge',
            uuids,
            tx,
            batch_size=len(uuids),
        )
        prepare_query = """
            MATCH ()-[e:RELATES_TO {uuid: $uuid}]->()
            SET e._graphiti_projection_lock = true
            REMOVE e._graphiti_projection_lock
            WITH e
            WHERE coalesce(e._graphiti_projection_version, 0) < $projection_version
            SET e._graphiti_vector_delete_pending = true
            SET e._graphiti_projection_version = $projection_version
            RETURN e.uuid AS uuid, $projection_version AS projection_version
        """
        if len(uuids) == 1:
            if tx is not None:
                await tx.run(
                    prepare_query,
                    uuid=uuids[0],
                    projection_version=versions[uuids[0]],
                )
            else:
                await executor.execute_query(
                    prepare_query,
                    uuid=uuids[0],
                    projection_version=versions[uuids[0]],
                )
        else:
            await self._prepare_delete_bulk(executor, uuids, versions, tx)

        if self._driver is not None:
            await self._driver.delete_from_aoss_async(
                'edge_fact_embedding',
                uuids=uuids,
                versions=versions,
            )

        if len(uuids) == 1:
            query = """
                MATCH ()-[e:RELATES_TO {uuid: $uuid}]->()
                WHERE coalesce(e._graphiti_vector_delete_pending, false) = true
                  AND e._graphiti_projection_version = $projection_version
                DELETE e
            """
            if tx is not None:
                await tx.run(query, uuid=uuids[0], projection_version=versions[uuids[0]])
            else:
                await executor.execute_query(
                    query,
                    uuid=uuids[0],
                    projection_version=versions[uuids[0]],
                )
        else:
            await self._finalize_delete_bulk(executor, uuids, versions, tx=tx)

    async def _prepare_delete_bulk(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        versions: dict[str, int],
        tx: Transaction | None,
    ) -> None:
        deletions = [{'uuid': uuid, 'projection_version': versions[uuid]} for uuid in uuids]
        query = """
            UNWIND $deletions AS deletion
            MATCH ()-[e:RELATES_TO {uuid: deletion.uuid}]->()
            SET e._graphiti_projection_lock = true
            REMOVE e._graphiti_projection_lock
            WITH e, deletion
            WHERE coalesce(e._graphiti_projection_version, 0) < deletion.projection_version
            SET e._graphiti_vector_delete_pending = true
            SET e._graphiti_projection_version = deletion.projection_version
            RETURN e.uuid AS uuid, deletion.projection_version AS projection_version
        """
        if tx is not None:
            await tx.run(query, deletions=deletions)
        else:
            await executor.execute_query(query, deletions=deletions)

    async def _finalize_delete_bulk(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        versions: dict[str, int],
        tx: Transaction | None,
    ) -> None:
        deletions = [{'uuid': uuid, 'projection_version': versions[uuid]} for uuid in uuids]
        query = """
            UNWIND $deletions AS deletion
            MATCH ()-[e:RELATES_TO]->()
            WHERE e.uuid = deletion.uuid
              AND coalesce(e._graphiti_vector_delete_pending, false) = true
              AND e._graphiti_projection_version = deletion.projection_version
            DELETE e
        """
        if tx is not None:
            await tx.run(query, deletions=deletions)
        else:
            await executor.execute_query(query, deletions=deletions)

    @defer_cancellation_until_complete
    async def delete_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        validate_batch_size(batch_size)
        unique_uuids = list(dict.fromkeys(uuids))
        if not unique_uuids:
            return
        for start in range(0, len(unique_uuids), batch_size):
            chunk = unique_uuids[start : start + batch_size]
            versions = await reserve_projection_versions(
                executor,
                'edge',
                chunk,
                tx,
                batch_size=batch_size,
            )
            await self._prepare_delete_bulk(executor, chunk, versions, tx)
            if self._driver is not None:
                await self._driver.delete_from_aoss_async(
                    'edge_fact_embedding',
                    uuids=chunk,
                    versions=versions,
                )
            await self._finalize_delete_bulk(executor, chunk, versions, tx)

    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
    ) -> EntityEdge:
        query = """
            MATCH (n:Entity)-[e:RELATES_TO {uuid: $uuid}]->(m:Entity)
            WHERE coalesce(e._graphiti_vector_delete_pending, false) = false
              AND coalesce(n._graphiti_vector_delete_pending, false) = false
              AND coalesce(m._graphiti_vector_delete_pending, false) = false
            RETURN
            """ + get_entity_edge_return_query(GraphProvider.NEPTUNE)
        records, _, _ = await executor.execute_query(query, uuid=uuid)
        edges = [entity_edge_from_record(r) for r in records]
        if len(edges) == 0:
            raise EdgeNotFoundError(uuid)
        return edges[0]

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[EntityEdge]:
        if not uuids:
            return []
        query = """
            MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
            WHERE e.uuid IN $uuids
              AND coalesce(e._graphiti_vector_delete_pending, false) = false
              AND coalesce(n._graphiti_vector_delete_pending, false) = false
              AND coalesce(m._graphiti_vector_delete_pending, false) = false
            RETURN
            """ + get_entity_edge_return_query(GraphProvider.NEPTUNE)
        records, _, _ = await executor.execute_query(query, uuids=uuids)
        return [entity_edge_from_record(r) for r in records]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EntityEdge]:
        cursor_clause = 'AND e.uuid < $uuid' if uuid_cursor else ''
        limit_clause = 'LIMIT $limit' if limit is not None else ''
        query = (
            """
            MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
            WHERE e.group_id IN $group_ids
              AND coalesce(e._graphiti_vector_delete_pending, false) = false
              AND coalesce(n._graphiti_vector_delete_pending, false) = false
              AND coalesce(m._graphiti_vector_delete_pending, false) = false
            """
            + cursor_clause
            + """
            RETURN
            """
            + get_entity_edge_return_query(GraphProvider.NEPTUNE)
            + """
            ORDER BY e.uuid DESC
            """
            + limit_clause
        )
        records, _, _ = await executor.execute_query(
            query,
            group_ids=group_ids,
            uuid=uuid_cursor,
            limit=limit,
        )
        return [entity_edge_from_record(r) for r in records]

    async def get_between_nodes(
        self,
        executor: QueryExecutor,
        source_node_uuid: str,
        target_node_uuid: str,
    ) -> list[EntityEdge]:
        query = """
            MATCH (n:Entity {uuid: $source_node_uuid})-[e:RELATES_TO]->(m:Entity {uuid: $target_node_uuid})
            WHERE coalesce(e._graphiti_vector_delete_pending, false) = false
              AND coalesce(n._graphiti_vector_delete_pending, false) = false
              AND coalesce(m._graphiti_vector_delete_pending, false) = false
            RETURN
            """ + get_entity_edge_return_query(GraphProvider.NEPTUNE)
        records, _, _ = await executor.execute_query(
            query,
            source_node_uuid=source_node_uuid,
            target_node_uuid=target_node_uuid,
        )
        return [entity_edge_from_record(r) for r in records]

    async def get_by_node_uuid(
        self,
        executor: QueryExecutor,
        node_uuid: str,
    ) -> list[EntityEdge]:
        query = """
            MATCH (n:Entity {uuid: $node_uuid})-[e:RELATES_TO]-(m:Entity)
            WHERE coalesce(e._graphiti_vector_delete_pending, false) = false
              AND coalesce(n._graphiti_vector_delete_pending, false) = false
              AND coalesce(m._graphiti_vector_delete_pending, false) = false
            RETURN
            """ + get_entity_edge_return_query(GraphProvider.NEPTUNE)
        records, _, _ = await executor.execute_query(query, node_uuid=node_uuid)
        return [entity_edge_from_record(r) for r in records]

    async def load_embeddings(
        self,
        executor: QueryExecutor,
        edge: EntityEdge,
    ) -> None:
        query = """
            MATCH (n:Entity)-[e:RELATES_TO {uuid: $uuid}]->(m:Entity)
            WHERE coalesce(e._graphiti_vector_delete_pending, false) = false
              AND coalesce(n._graphiti_vector_delete_pending, false) = false
              AND coalesce(m._graphiti_vector_delete_pending, false) = false
            RETURN [x IN split(e.fact_embedding, ",") | toFloat(x)] AS fact_embedding
        """
        records, _, _ = await executor.execute_query(query, uuid=edge.uuid)
        if len(records) == 0:
            raise EdgeNotFoundError(edge.uuid)
        edge.fact_embedding = records[0]['fact_embedding']

    async def load_embeddings_bulk(
        self,
        executor: QueryExecutor,
        edges: list[EntityEdge],
        batch_size: int = 100,
    ) -> None:
        uuids = [e.uuid for e in edges]
        query = """
            MATCH (n:Entity)-[e:RELATES_TO]-(m:Entity)
            WHERE e.uuid IN $edge_uuids
              AND coalesce(e._graphiti_vector_delete_pending, false) = false
              AND coalesce(n._graphiti_vector_delete_pending, false) = false
              AND coalesce(m._graphiti_vector_delete_pending, false) = false
            RETURN DISTINCT e.uuid AS uuid, [x IN split(e.fact_embedding, ",") | toFloat(x)] AS fact_embedding
        """
        embedding_map: dict[str, list[float]] = {}
        for i in range(0, len(uuids), batch_size):
            chunk = uuids[i : i + batch_size]
            records, _, _ = await executor.execute_query(query, edge_uuids=chunk)
            embedding_map.update({r['uuid']: r['fact_embedding'] for r in records})
        for edge in edges:
            if edge.uuid in embedding_map:
                edge.fact_embedding = embedding_map[edge.uuid]
