import hashlib
import json
import logging
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException
from graphiti_core import Graphiti  # type: ignore
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.record_parsers import episodic_node_from_record
from graphiti_core.edges import EntityEdge  # type: ignore
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.helpers import EPISODE_AOSS_TOMBSTONE_VERSION
from graphiti_core.llm_client import LLMClient  # type: ignore
from graphiti_core.models.nodes.node_db_queries import (
    EPISODIC_NODE_RETURN,
    EPISODIC_NODE_RETURN_NEPTUNE,
)
from graphiti_core.nodes import EntityNode, EpisodicNode  # type: ignore

from graph_service.config import ZepEnvDep
from graph_service.dto import FactResult

logger = logging.getLogger(__name__)


def _conditional_episode_identity_digest(
    uuid: str, group_id: str, name: str, content: str, source_description: str
) -> str:
    canonical = json.dumps(
        [uuid, group_id, name, hashlib.sha256(content.encode()).hexdigest(), source_description],
        ensure_ascii=False,
        separators=(',', ':'),
    )
    return hashlib.sha256(f'opr:conditional-episode-delete:v1\0{canonical}'.encode()).hexdigest()


class ZepGraphiti(Graphiti):
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        llm_client: LLMClient | None = None,
        **kwargs,
    ):
        super().__init__(uri, user, password, llm_client, **kwargs)  # type: ignore

    async def save_entity_node(self, name: str, uuid: str, group_id: str, summary: str = ''):
        new_node = EntityNode(
            name=name,
            uuid=uuid,
            group_id=group_id,
            summary=summary,
        )
        await new_node.generate_name_embedding(self.embedder)
        await new_node.save(self.driver)
        return new_node

    async def retrieve_episodes_for_reconciliation(
        self, group_id: str, last_n: int
    ) -> list[EpisodicNode]:
        """Return the raw group ledger, including an in-progress retirement.

        Normal recall/read APIs filter retirement tombstones. This explicitly
        administrative listing remains complete so a caller that received a
        transient cross-store failure can retry the exact conditional retire
        until both the graph and search projection are durably fenced.
        """
        query_driver = (
            self.driver.with_database(group_id)
            if self.driver.provider == GraphProvider.FALKORDB
            else self.driver
        )
        episode_return = (
            EPISODIC_NODE_RETURN_NEPTUNE
            if self.driver.provider == GraphProvider.NEPTUNE
            else EPISODIC_NODE_RETURN
        )
        records, _, _ = await query_driver.execute_query(
            """
            MATCH (e:Episodic)
            WHERE e.group_id = $group_id
              AND coalesce(e.opr_episode_reservation, false) = false
            RETURN DISTINCT
            """
            + episode_return
            + """
            ORDER BY uuid DESC
            LIMIT $limit
            """,
            group_id=group_id,
            limit=last_n,
            routing_='r',
        )
        return [episodic_node_from_record(record) for record in records]

    async def get_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            return edge
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_group(self, group_id: str):
        try:
            edges = await EntityEdge.get_by_group_ids(self.driver, [group_id])
        except GroupsEdgesNotFoundError:
            logger.warning(f'No edges found for group {group_id}')
            edges = []

        nodes = await EntityNode.get_by_group_ids(self.driver, [group_id])

        episodes = await EpisodicNode.get_by_group_ids(self.driver, [group_id])

        for edge in edges:
            await edge.delete(self.driver)

        for node in nodes:
            await node.delete(self.driver)

        for episode in episodes:
            await episode.delete(self.driver)

    async def delete_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            await edge.delete(self.driver)
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_episodic_node(self, uuid: str):
        try:
            episode = await EpisodicNode.get_by_uuid(self.driver, uuid)
            await episode.delete(self.driver)
        except NodeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_episodic_node_if_matches(
        self,
        uuid: str,
        *,
        group_id: str,
        name: str,
        content: str,
        source_description: str,
    ) -> bool:
        """Atomically compare the complete episode identity and retire it.

        A separate read followed by ``delete_episodic_node`` is unsafe because
        episode writes MERGE on caller-controlled UUIDs. This single graph
        statement locks the episode before comparison and leaves a permanent
        tombstone that every episode writer checks under the same lock. A
        same-UUID replacement therefore either wins before the comparison
        (and fails it) or observes the tombstone and cannot recreate the
        retired projection.
        """
        # The transient property follows Neo4j's documented explicit-lock
        # pattern: acquire the node write lock before reading the compared
        # fields, then immediately remove the property. Without this ordering,
        # read-committed isolation can admit a lost-update style TOCTOU race.
        # https://neo4j.com/docs/operations-manual/current/database-internals/concurrent-data-access/
        try:
            canonical_uuid = str(UUID(uuid))
        except (AttributeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail='episode UUID is invalid') from exc
        if self.driver.provider == GraphProvider.KUZU:
            raise HTTPException(
                status_code=501,
                detail='conditional episode retirement is unsupported by the Kuzu backend',
            )
        identity_digest = _conditional_episode_identity_digest(
            canonical_uuid, group_id, name, content, source_description
        )
        query_driver = self.driver
        if self.driver.provider == GraphProvider.FALKORDB:
            # Route to the request graph without constructing a second driver;
            # FalkorDriver.clone() starts index-initialization work and makes
            # this one-shot request responsible for another driver's lifecycle.
            query_driver = self.driver.with_database(group_id)

        records, _, _ = await query_driver.execute_query(
            """
            MATCH (episode:Episodic {uuid: $uuid})
            SET episode._opr_conditional_delete_lock = true
            REMOVE episode._opr_conditional_delete_lock
            WITH episode, coalesce(episode.opr_deleted, false) AS was_deleted
            WHERE (
                was_deleted = false
                AND episode.uuid = $uuid
                AND episode.group_id = $group_id
                AND episode.name = $name
                AND episode.content = $content
                AND episode.source_description = $source_description
            ) OR (
                was_deleted = true
                AND episode.opr_deleted_identity_digest = $identity_digest
            )
            OPTIONAL MATCH (episode)-[relationship]-()
            DELETE relationship
            WITH DISTINCT episode, was_deleted
            SET episode.opr_deleted = true,
                episode.opr_deleted_group_id = $group_id,
                episode.opr_deleted_identity_digest = $identity_digest,
                episode.opr_generation = CASE
                    WHEN was_deleted THEN coalesce(episode.opr_generation, 1)
                    ELSE coalesce(episode.opr_generation, 0) + 1
                END,
                episode.opr_aoss_fenced = CASE
                    WHEN was_deleted THEN coalesce(episode.opr_aoss_fenced, false)
                    ELSE false
                END
            RETURN $uuid AS uuid, episode.opr_aoss_fenced AS aoss_fenced
            """,
            uuid=canonical_uuid,
            group_id=group_id,
            name=name,
            content=content,
            source_description=source_description,
            identity_digest=identity_digest,
        )
        if not records:
            return False

        aoss_fenced = bool(records[0].get('aoss_fenced', False))
        if self.driver.provider == GraphProvider.NEPTUNE and not aoss_fenced:
            indexed = self.driver.save_to_aoss(  # pyright: ignore[reportAttributeAccessIssue]
                'episode_content',
                [
                    {
                        'uuid': canonical_uuid,
                        'content': '',
                        'source': '',
                        'source_description': 'opr_conditional_delete',
                        'group_id': '__opr_deleted__',
                        '_version': EPISODE_AOSS_TOMBSTONE_VERSION,
                    }
                ],
            )
            if indexed != 1:
                raise HTTPException(
                    status_code=503,
                    detail='episode tombstone search projection is not durable',
                )

        if aoss_fenced:
            return True

        entity_edges: str | list[str] = '' if self.driver.provider == GraphProvider.NEPTUNE else []
        finalized, _, _ = await query_driver.execute_query(
            """
            MATCH (episode:Episodic {uuid: $uuid})
            SET episode._opr_conditional_delete_lock = true
            REMOVE episode._opr_conditional_delete_lock
            WITH episode
            WHERE coalesce(episode.opr_deleted, false) = true
              AND episode.opr_deleted_identity_digest = $identity_digest
            SET episode.opr_aoss_fenced = true,
                episode.group_id = '__opr_deleted__',
                episode.name = '__opr_deleted__',
                episode.content = '',
                episode.source_description = 'opr_conditional_delete',
                episode.entity_edges = $entity_edges
            RETURN $uuid AS uuid
            """,
            uuid=canonical_uuid,
            identity_digest=identity_digest,
            entity_edges=entity_edges,
        )
        if not finalized:
            raise HTTPException(
                status_code=503,
                detail='episode tombstone finalization is not durable',
            )
        return True


def _create_graphiti_client(settings: ZepEnvDep) -> ZepGraphiti:
    """Create a ZepGraphiti client based on the configured database backend."""
    if settings.db_backend == 'falkordb':
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        driver = FalkorDriver(  # type: ignore
            host=settings.falkordb_host or 'localhost',  # type: ignore
            port=settings.falkordb_port or 6379,  # type: ignore
            database=settings.falkordb_database or 'default_db',  # type: ignore
        )
        return ZepGraphiti(graph_driver=driver)  # type: ignore
    elif settings.db_backend == 'neptune':
        from graphiti_core.driver.neptune_driver import NeptuneDriver

        if not settings.neptune_host or not settings.aoss_host:
            raise ValueError('NEPTUNE_HOST and AOSS_HOST are required when db_backend is "neptune"')
        driver = NeptuneDriver(  # type: ignore
            host=settings.neptune_host,
            aoss_host=settings.aoss_host,
            port=settings.neptune_port or 8182,
            aoss_port=settings.aoss_port or 443,
        )
        return ZepGraphiti(graph_driver=driver)  # type: ignore
    elif settings.db_backend == 'kuzu':
        from graphiti_core.driver.kuzu_driver import KuzuDriver

        driver = KuzuDriver(  # type: ignore
            db=settings.kuzu_db or ':memory:',
            max_concurrent_queries=settings.kuzu_max_concurrent_queries or 1,
        )
        return ZepGraphiti(graph_driver=driver)  # type: ignore
    else:
        # Validate Neo4j settings are present
        if not all([settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password]):
            raise ValueError(
                'Neo4j configuration (neo4j_uri, neo4j_user, neo4j_password) is required '
                "when db_backend is 'neo4j'"
            )
        return ZepGraphiti(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
        )


async def get_graphiti(settings: ZepEnvDep):
    client = _create_graphiti_client(settings)
    if settings.openai_base_url is not None:
        client.llm_client.config.base_url = settings.openai_base_url
    if settings.openai_api_key is not None:
        client.llm_client.config.api_key = settings.openai_api_key
    if settings.model_name is not None:
        client.llm_client.model = settings.model_name

    try:
        yield client
    finally:
        await client.close()


async def initialize_graphiti(settings: ZepEnvDep):
    client = _create_graphiti_client(settings)
    try:
        await client.build_indices_and_constraints()
    finally:
        await client.close()


def get_fact_result_from_edge(edge: EntityEdge):
    return FactResult(
        uuid=edge.uuid,
        name=edge.name,
        fact=edge.fact,
        valid_at=edge.valid_at,
        invalid_at=edge.invalid_at,
        created_at=edge.created_at,
        expired_at=edge.expired_at,
    )


ZepGraphitiDep = Annotated[ZepGraphiti, Depends(get_graphiti)]
