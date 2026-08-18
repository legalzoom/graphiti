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
from graph_service.protocol import GRAPHITI_RECONCILIATION_PROTOCOL

logger = logging.getLogger(__name__)

_CONDITIONAL_EPISODE_IDENTITY_DOMAIN = 'opr:conditional-episode-delete:v2'
_RETIREMENT_RECEIPT_PROTOCOL = GRAPHITI_RECONCILIATION_PROTOCOL


def _conditional_episode_identity_digest(
    uuid: str,
    group_id: str,
    name: str,
    content: str,
    source: str,
    source_description: str,
) -> str:
    canonical = json.dumps(
        [
            uuid,
            group_id,
            name,
            hashlib.sha256(content.encode()).hexdigest(),
            source,
            source_description,
        ],
        ensure_ascii=False,
        separators=(',', ':'),
    )
    return hashlib.sha256(
        f'{_CONDITIONAL_EPISODE_IDENTITY_DOMAIN}\0{canonical}'.encode()
    ).hexdigest()


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
        source: str,
        source_description: str,
        retirement_request_id: str,
    ) -> bool | None:
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
            canonical_request_id = str(UUID(retirement_request_id))
        except (AttributeError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail='episode or retirement request UUID is invalid',
            ) from exc
        if self.driver.provider in {GraphProvider.KUZU, GraphProvider.FALKORDB}:
            raise HTTPException(
                status_code=501,
                detail=(
                    'conditional episode retirement requires a backend with '
                    'request-receipt uniqueness'
                ),
            )
        identity_digest = _conditional_episode_identity_digest(
            canonical_uuid, group_id, name, content, source, source_description
        )
        query_driver = self.driver
        if self.driver.provider == GraphProvider.NEPTUNE:
            receipt_node_id = f'opr-retirement-receipt:{canonical_request_id}'
            receipts, _, _ = await query_driver.execute_query(
                """
                MERGE (receipt:OPRRetirementReceipt {`~id`: $receipt_node_id})
                ON CREATE SET receipt.request_id = $retirement_request_id,
                              receipt.episode_uuid = $uuid,
                              receipt.group_id = $group_id,
                              receipt.identity_digest = $identity_digest,
                              receipt.protocol = $receipt_protocol,
                              receipt.outcome = 'pending',
                              receipt.opr_deleted = true
                SET receipt._opr_conditional_delete_lock = true
                REMOVE receipt._opr_conditional_delete_lock
                RETURN receipt.request_id = $retirement_request_id
                       AND receipt.episode_uuid = $uuid
                       AND receipt.group_id = $group_id
                       AND receipt.identity_digest = $identity_digest
                       AND receipt.protocol = $receipt_protocol AS bound,
                       receipt.outcome AS outcome
                """,
                receipt_node_id=receipt_node_id,
                retirement_request_id=canonical_request_id,
                uuid=canonical_uuid,
                group_id=group_id,
                identity_digest=identity_digest,
                receipt_protocol=_RETIREMENT_RECEIPT_PROTOCOL,
            )
            if not receipts or receipts[0].get('bound') is not True:
                return None
            receipt_outcome = receipts[0].get('outcome')
            if receipt_outcome == 'not_applied':
                return False
            if receipt_outcome not in {'pending', 'retired'}:
                return None

            records, _, _ = await query_driver.execute_query(
                """
                MATCH (receipt:OPRRetirementReceipt {`~id`: $receipt_node_id})
                SET receipt._opr_conditional_delete_lock = true
                REMOVE receipt._opr_conditional_delete_lock
                WITH receipt
                WHERE receipt.request_id = $retirement_request_id
                  AND receipt.episode_uuid = $uuid
                  AND receipt.group_id = $group_id
                  AND receipt.identity_digest = $identity_digest
                  AND receipt.protocol = $receipt_protocol
                  AND receipt.outcome <> 'not_applied'
                MATCH (episode:Episodic {uuid: $uuid})
                SET episode._opr_conditional_delete_lock = true
                REMOVE episode._opr_conditional_delete_lock
                WITH receipt, episode,
                     coalesce(episode.opr_deleted, false) AS was_deleted
                WHERE (
                    receipt.outcome = 'pending'
                    AND coalesce(episode.opr_deleted, false) = false
                    AND episode.uuid = $uuid
                    AND episode.group_id = $group_id
                    AND episode.name = $name
                    AND episode.content = $content
                    AND episode.source = $source
                    AND episode.source_description = $source_description
                ) OR (
                    receipt.outcome = 'retired'
                    AND coalesce(episode.opr_deleted, false) = true
                    AND episode.opr_deleted_identity_digest = $identity_digest
                    AND episode.opr_retirement_request_id = $retirement_request_id
                )
                SET receipt.outcome = 'retired',
                    episode.opr_deleted = true,
                    episode.opr_deleted_group_id = $group_id,
                    episode.opr_deleted_identity_digest = $identity_digest,
                    episode.opr_retirement_request_id = $retirement_request_id,
                    episode.opr_generation = CASE
                        WHEN was_deleted THEN coalesce(episode.opr_generation, 1)
                        ELSE coalesce(episode.opr_generation, 0) + 1
                    END,
                    episode.opr_aoss_fenced = CASE
                        WHEN was_deleted THEN coalesce(episode.opr_aoss_fenced, false)
                        ELSE false
                    END
                RETURN receipt.outcome AS outcome,
                       true AS applied,
                       coalesce(episode.opr_aoss_fenced, false) AS aoss_fenced
                """,
                receipt_node_id=receipt_node_id,
                retirement_request_id=canonical_request_id,
                uuid=canonical_uuid,
                group_id=group_id,
                name=name,
                content=content,
                source=source,
                source_description=source_description,
                identity_digest=identity_digest,
                receipt_protocol=_RETIREMENT_RECEIPT_PROTOCOL,
            )
            if not records:
                decisions, _, _ = await query_driver.execute_query(
                    """
                    MATCH (receipt:OPRRetirementReceipt {`~id`: $receipt_node_id})
                    SET receipt._opr_conditional_delete_lock = true
                    REMOVE receipt._opr_conditional_delete_lock
                    WITH receipt
                    WHERE receipt.request_id = $retirement_request_id
                      AND receipt.episode_uuid = $uuid
                      AND receipt.group_id = $group_id
                      AND receipt.identity_digest = $identity_digest
                      AND receipt.protocol = $receipt_protocol
                    SET receipt.outcome = CASE
                        WHEN receipt.outcome = 'pending' THEN 'not_applied'
                        ELSE receipt.outcome
                    END
                    RETURN receipt.outcome AS outcome
                    """,
                    receipt_node_id=receipt_node_id,
                    retirement_request_id=canonical_request_id,
                    uuid=canonical_uuid,
                    group_id=group_id,
                    identity_digest=identity_digest,
                    receipt_protocol=_RETIREMENT_RECEIPT_PROTOCOL,
                )
                if not decisions:
                    return None
                decided_outcome = decisions[0].get('outcome')
                if decided_outcome == 'not_applied':
                    return False
                if decided_outcome != 'retired':
                    return None
                aoss_fenced = False
            else:
                aoss_fenced = bool(records[0].get('aoss_fenced', False))
        else:
            records, _, _ = await query_driver.execute_query(
                """
            MERGE (receipt:OPRRetirementReceipt {
                request_id: $retirement_request_id
            })
            ON CREATE SET receipt.episode_uuid = $uuid,
                          receipt.group_id = $group_id,
                          receipt.identity_digest = $identity_digest,
                          receipt.protocol = $receipt_protocol,
                          receipt.outcome = 'pending',
                          receipt.opr_deleted = true
            SET receipt._opr_conditional_delete_lock = true
            REMOVE receipt._opr_conditional_delete_lock
            WITH receipt
            WHERE receipt.episode_uuid = $uuid
              AND receipt.group_id = $group_id
              AND receipt.identity_digest = $identity_digest
              AND receipt.protocol = $receipt_protocol
            OPTIONAL MATCH (episode:Episodic {uuid: $uuid})
            FOREACH (_ IN CASE WHEN episode IS NULL THEN [] ELSE [1] END |
                SET episode._opr_conditional_delete_lock = true
            )
            FOREACH (_ IN CASE WHEN episode IS NULL THEN [] ELSE [1] END |
                SET episode._opr_conditional_delete_lock = NULL
            )
            WITH receipt, episode,
                 coalesce(episode.opr_deleted, false) AS was_deleted,
                 receipt.outcome <> 'not_applied' AND episode IS NOT NULL AND (
                    (
                        receipt.outcome = 'pending'
                        AND coalesce(episode.opr_deleted, false) = false
                        AND episode.uuid = $uuid
                        AND episode.group_id = $group_id
                        AND episode.name = $name
                        AND episode.content = $content
                        AND episode.source = $source
                        AND episode.source_description = $source_description
                    ) OR (
                        receipt.outcome = 'retired'
                        AND coalesce(episode.opr_deleted, false) = true
                        AND episode.opr_deleted_identity_digest = $identity_digest
                        AND episode.opr_retirement_request_id = $retirement_request_id
                    )
                 ) AS can_apply
            SET receipt.outcome = CASE
                WHEN can_apply THEN 'retired'
                WHEN receipt.outcome = 'pending' THEN 'not_applied'
                ELSE receipt.outcome
            END
            WITH receipt, episode, was_deleted, can_apply
            OPTIONAL MATCH (episode)-[relationship]-()
            FOREACH (_ IN CASE
                WHEN can_apply AND relationship IS NOT NULL THEN [1] ELSE [] END |
                DELETE relationship
            )
            FOREACH (_ IN CASE WHEN can_apply THEN [1] ELSE [] END |
                SET episode.opr_deleted = true,
                    episode.opr_deleted_group_id = $group_id,
                    episode.opr_deleted_identity_digest = $identity_digest,
                    episode.opr_retirement_request_id = $retirement_request_id,
                    episode.opr_generation = CASE
                        WHEN was_deleted THEN coalesce(episode.opr_generation, 1)
                        ELSE coalesce(episode.opr_generation, 0) + 1
                    END,
                    episode.opr_aoss_fenced = CASE
                        WHEN was_deleted THEN coalesce(episode.opr_aoss_fenced, false)
                        ELSE false
                    END
            )
            RETURN receipt.outcome AS outcome,
                   can_apply AS applied,
                   coalesce(episode.opr_aoss_fenced, false) AS aoss_fenced
                """,
                uuid=canonical_uuid,
                group_id=group_id,
                name=name,
                content=content,
                source=source,
                source_description=source_description,
                identity_digest=identity_digest,
                retirement_request_id=canonical_request_id,
                receipt_protocol=_RETIREMENT_RECEIPT_PROTOCOL,
            )
            if not records:
                return None
            outcome = records[0].get('outcome')
            applied = bool(records[0].get('applied', False))
            if outcome == 'not_applied':
                return False
            if outcome != 'retired' or not applied:
                raise HTTPException(
                    status_code=503,
                    detail='episode retirement receipt is inconsistent',
                )
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
            MATCH (receipt:OPRRetirementReceipt {
                request_id: $retirement_request_id
            })
            SET receipt._opr_conditional_delete_lock = true
            REMOVE receipt._opr_conditional_delete_lock
            WITH receipt
            WHERE receipt.episode_uuid = $uuid
              AND receipt.group_id = $group_id
              AND receipt.identity_digest = $identity_digest
              AND receipt.protocol = $receipt_protocol
              AND receipt.outcome = 'retired'
            MATCH (episode:Episodic {uuid: $uuid})
            SET episode._opr_conditional_delete_lock = true
            REMOVE episode._opr_conditional_delete_lock
            WITH episode
            WHERE coalesce(episode.opr_deleted, false) = true
              AND episode.opr_deleted_identity_digest = $identity_digest
            OPTIONAL MATCH (episode)-[relationship]-()
            DELETE relationship
            WITH DISTINCT episode
            SET episode.opr_aoss_fenced = true,
                episode.group_id = '__opr_deleted__',
                episode.name = '__opr_deleted__',
                episode.content = '',
                episode.source_description = 'opr_conditional_delete',
                episode.entity_edges = $entity_edges
            RETURN $uuid AS uuid
            """,
            uuid=canonical_uuid,
            group_id=group_id,
            identity_digest=identity_digest,
            retirement_request_id=canonical_request_id,
            receipt_protocol=_RETIREMENT_RECEIPT_PROTOCOL,
            entity_edges=entity_edges,
        )
        if not finalized:
            raise HTTPException(
                status_code=503,
                detail='episode tombstone finalization is not durable',
            )
        return True

    async def episode_retirement_outcome(
        self,
        uuid: str,
        *,
        group_id: str,
        retirement_request_id: str,
    ) -> str | None:
        """Return a durable request-bound ``retired`` or ``not_applied`` receipt.

        The transient write makes this a writer-endpoint mutation query rather
        than a potentially lagging replica read. The provider-specific receipt
        is uniquely keyed and its terminal outcome is serialized with the
        episode identity decision, so this is an idempotency receipt rather
        than a generic absence check.
        """
        try:
            canonical_uuid = str(UUID(uuid))
            canonical_request_id = str(UUID(retirement_request_id))
        except (AttributeError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail='episode or retirement request UUID is invalid',
            ) from exc
        if self.driver.provider in {GraphProvider.KUZU, GraphProvider.FALKORDB}:
            raise HTTPException(
                status_code=501,
                detail=(
                    'episode retirement status requires a backend with request-receipt uniqueness'
                ),
            )
        query_driver = self.driver
        if self.driver.provider == GraphProvider.NEPTUNE:
            receipt_node_id = f'opr-retirement-receipt:{canonical_request_id}'
            # Neptune cannot use FOREACH for a conditional write. Alias the
            # present episode as the second lock target, or the already-locked
            # receipt on absence, so the decision stays in one writer query.
            receipts, _, _ = await query_driver.execute_query(
                """
                MATCH (receipt:OPRRetirementReceipt {`~id`: $receipt_node_id})
                SET receipt._opr_conditional_delete_lock = true
                REMOVE receipt._opr_conditional_delete_lock
                WITH receipt
                WHERE receipt.request_id = $retirement_request_id
                  AND receipt.episode_uuid = $uuid
                  AND receipt.group_id = $group_id
                  AND receipt.protocol = $receipt_protocol
                OPTIONAL MATCH (episode:Episodic {uuid: $uuid})
                WITH receipt, episode,
                     CASE WHEN episode IS NULL THEN receipt ELSE episode END AS decision_lock
                SET decision_lock._opr_conditional_delete_lock = true
                REMOVE decision_lock._opr_conditional_delete_lock
                WITH receipt, episode
                SET receipt.outcome = CASE
                    WHEN receipt.outcome = 'pending' AND episode IS NULL THEN 'not_applied'
                    ELSE receipt.outcome
                END
                RETURN true AS bound,
                       receipt.outcome AS outcome,
                       receipt.outcome = 'retired'
                       AND episode IS NOT NULL
                       AND coalesce(episode.opr_deleted, false) = true
                       AND episode.opr_deleted_group_id = $group_id
                       AND episode.opr_retirement_request_id = $retirement_request_id
                       AND coalesce(episode.opr_aoss_fenced, false) = true AS durable
                """,
                receipt_node_id=receipt_node_id,
                uuid=canonical_uuid,
                group_id=group_id,
                retirement_request_id=canonical_request_id,
                receipt_protocol=_RETIREMENT_RECEIPT_PROTOCOL,
            )
            if not receipts or receipts[0].get('bound') is not True:
                return None
            outcome = receipts[0].get('outcome')
            if outcome == 'not_applied':
                return 'not_applied'
            if outcome == 'retired' and receipts[0].get('durable') is True:
                return 'retired'
            return None
        records, _, _ = await query_driver.execute_query(
            """
            MATCH (receipt:OPRRetirementReceipt {
                request_id: $retirement_request_id
            })
            SET receipt._opr_conditional_delete_lock = true
            REMOVE receipt._opr_conditional_delete_lock
            WITH receipt
            WHERE receipt.episode_uuid = $uuid
              AND receipt.group_id = $group_id
              AND receipt.protocol = $receipt_protocol
            OPTIONAL MATCH (episode:Episodic {uuid: $uuid})
            FOREACH (_ IN CASE WHEN episode IS NULL THEN [] ELSE [1] END |
                SET episode._opr_conditional_delete_lock = true
            )
            FOREACH (_ IN CASE WHEN episode IS NULL THEN [] ELSE [1] END |
                SET episode._opr_conditional_delete_lock = NULL
            )
            RETURN receipt.outcome AS outcome,
                   receipt.outcome = 'retired'
                   AND episode IS NOT NULL
                   AND coalesce(episode.opr_deleted, false) = true
                   AND episode.opr_deleted_group_id = $group_id
                   AND episode.opr_retirement_request_id = $retirement_request_id
                   AND coalesce(episode.opr_aoss_fenced, false) = true AS durable
            """,
            uuid=canonical_uuid,
            group_id=group_id,
            retirement_request_id=canonical_request_id,
            receipt_protocol=_RETIREMENT_RECEIPT_PROTOCOL,
        )
        if not records:
            return None
        outcome = records[0].get('outcome')
        if outcome == 'not_applied':
            return 'not_applied'
        if outcome == 'retired' and records[0].get('durable') is True:
            return 'retired'
        return None


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
