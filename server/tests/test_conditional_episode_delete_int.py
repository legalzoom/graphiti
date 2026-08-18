import asyncio
import os
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.driver.neo4j.operations.episode_node_ops import Neo4jEpisodeNodeOperations
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.errors import EpisodeTombstonedError, NodeNotFoundError
from graphiti_core.nodes import EpisodeType, EpisodicNode

from graph_service.zep_graphiti import ZepGraphiti

pytestmark = pytest.mark.integration


def _neo4j_driver() -> Neo4jDriver:
    return Neo4jDriver(
        os.environ.get('NEO4J_URI', 'bolt://127.0.0.1:7687'),
        os.environ.get('NEO4J_USER', 'neo4j'),
        os.environ.get('NEO4J_PASSWORD', 'testpass'),
    )


@pytest.mark.asyncio
async def test_conditional_delete_rechecks_group_after_waiting_for_writer():
    driver = _neo4j_driver()
    service = cast(ZepGraphiti, SimpleNamespace(driver=driver))
    episode_uuid = str(uuid4())
    await driver.execute_query(
        """
        CREATE (:Episodic {
            uuid: $uuid, group_id: 'opr', name: 'curated:test.md',
            content: 'stored content', source_description: 'publish'
        })
        """,
        uuid=episode_uuid,
    )

    try:
        async with driver.client.session(database=driver._database) as session:
            tx = await session.begin_transaction()
            await tx.run(
                "MATCH (episode:Episodic {uuid: $uuid}) SET episode.group_id = 'other'",
                uuid=episode_uuid,
            )

            delete_task = asyncio.create_task(
                service.delete_episodic_node_if_matches(
                    episode_uuid,
                    group_id='opr',
                    name='curated:test.md',
                    content='stored content',
                    source_description='publish',
                )
            )
            await asyncio.sleep(0.1)
            assert not delete_task.done()
            await tx.commit()

        assert await asyncio.wait_for(delete_task, timeout=5) is False
        records, _, _ = await driver.execute_query(
            'MATCH (episode:Episodic {uuid: $uuid}) RETURN episode.group_id AS group_id',
            uuid=episode_uuid,
        )
        assert records[0]['group_id'] == 'other'
    finally:
        await driver.execute_query('MATCH (n {uuid: $uuid}) DETACH DELETE n', uuid=episode_uuid)
        await driver.close()


@pytest.mark.asyncio
async def test_conditional_delete_tombstone_blocks_same_uuid_recreation():
    driver = _neo4j_driver()
    service = cast(ZepGraphiti, SimpleNamespace(driver=driver))
    episode_uuid = str(uuid4())
    await driver.execute_query(
        """
        CREATE (:Episodic {
            uuid: $uuid, group_id: 'opr', name: 'curated:test.md',
            content: 'stored content', source_description: 'publish'
        })
        """,
        uuid=episode_uuid,
    )

    try:
        assert await service.delete_episodic_node_if_matches(
            episode_uuid,
            group_id='opr',
            name='curated:test.md',
            content='stored content',
            source_description='publish',
        )

        replacement = EpisodicNode(
            uuid=episode_uuid,
            group_id='opr',
            name='curated:replacement.md',
            content='replacement content',
            source_description='publish',
            source=EpisodeType.message,
            created_at=datetime.now(timezone.utc),
            valid_at=datetime.now(timezone.utc),
        )
        with pytest.raises(EpisodeTombstonedError):
            await replacement.save(driver)

        # The legacy UUID-only delete path must not erase the permanent fence.
        await replacement.delete(driver)
        with pytest.raises(EpisodeTombstonedError):
            await replacement.save(driver)

        records, _, _ = await driver.execute_query(
            """
            MATCH (episode:Episodic {uuid: $uuid})
            RETURN episode.opr_deleted AS deleted,
                   episode.group_id AS group_id,
                   episode.content AS content
            """,
            uuid=episode_uuid,
        )
        assert records == [{'deleted': True, 'group_id': '__opr_deleted__', 'content': ''}]
    finally:
        await driver.execute_query('MATCH (n {uuid: $uuid}) DETACH DELETE n', uuid=episode_uuid)
        await driver.close()


@pytest.mark.asyncio
async def test_bulk_save_cannot_overwrite_concurrently_created_tombstone():
    driver = _neo4j_driver()
    episode_uuid = str(uuid4())
    fresh_uuid = str(uuid4())
    episode = EpisodicNode(
        uuid=episode_uuid,
        group_id='opr',
        name='curated:replacement.md',
        content='replacement content',
        source_description='publish',
        source=EpisodeType.message,
        created_at=datetime.now(timezone.utc),
        valid_at=datetime.now(timezone.utc),
    )
    fresh = episode.model_copy(
        update={
            'uuid': fresh_uuid,
            'name': 'curated:fresh.md',
            'content': 'fresh content',
        }
    )

    try:
        async with driver.client.session(database=driver._database) as session:
            tx = await session.begin_transaction()
            await tx.run(
                """
                MERGE (episode:Episodic {uuid: $uuid})
                SET episode.opr_deleted = true,
                    episode.opr_deleted_identity_digest = 'retired'
                """,
                uuid=episode_uuid,
            )

            save_task = asyncio.create_task(
                Neo4jEpisodeNodeOperations().save_bulk(driver, [fresh, episode])
            )
            await asyncio.sleep(0.1)
            assert not save_task.done()
            await tx.commit()

        with pytest.raises(EpisodeTombstonedError):
            await asyncio.wait_for(save_task, timeout=5)
        records, _, _ = await driver.execute_query(
            """
            MATCH (episode:Episodic {uuid: $uuid})
            RETURN episode.opr_deleted AS deleted,
                   episode.opr_deleted_identity_digest AS digest,
                   episode.content AS content
            """,
            uuid=episode_uuid,
        )
        assert records == [{'deleted': True, 'digest': 'retired', 'content': None}]
        fresh_records, _, _ = await driver.execute_query(
            """
            MATCH (episode:Episodic {uuid: $uuid})
            RETURN episode.name AS name,
                   episode.content AS content,
                   episode.opr_episode_reservation AS reservation
            """,
            uuid=fresh_uuid,
        )
        assert fresh_records == [{'name': None, 'content': None, 'reservation': True}]
        with pytest.raises(NodeNotFoundError):
            await EpisodicNode.get_by_uuid(driver, fresh_uuid)
    finally:
        await driver.execute_query(
            'MATCH (n) WHERE n.uuid IN $uuids DETACH DELETE n',
            uuids=[episode_uuid, fresh_uuid],
        )
        await driver.close()


@pytest.mark.asyncio
async def test_legacy_delete_rechecks_tombstone_after_waiting_for_retirement():
    driver = _neo4j_driver()
    episode_uuid = str(uuid4())
    episode = EpisodicNode(
        uuid=episode_uuid,
        group_id='opr',
        name='curated:test.md',
        content='stored content',
        source_description='publish',
        source=EpisodeType.message,
        created_at=datetime.now(timezone.utc),
        valid_at=datetime.now(timezone.utc),
    )
    await episode.save(driver)

    try:
        async with driver.client.session(database=driver._database) as session:
            tx = await session.begin_transaction()
            await tx.run(
                """
                MATCH (episode:Episodic {uuid: $uuid})
                SET episode._opr_conditional_delete_lock = true
                """,
                uuid=episode_uuid,
            )

            delete_task = asyncio.create_task(episode.delete(driver))
            await asyncio.sleep(0.1)
            assert not delete_task.done()
            await tx.run(
                """
                MATCH (episode:Episodic {uuid: $uuid})
                SET episode.opr_deleted = true,
                    episode.group_id = '__opr_deleted__',
                    episode.content = ''
                REMOVE episode._opr_conditional_delete_lock
                """,
                uuid=episode_uuid,
            )
            await tx.commit()

        await asyncio.wait_for(delete_task, timeout=5)
        records, _, _ = await driver.execute_query(
            """
            MATCH (episode:Episodic {uuid: $uuid})
            RETURN episode.opr_deleted AS deleted,
                   episode.group_id AS group_id
            """,
            uuid=episode_uuid,
        )
        assert records == [{'deleted': True, 'group_id': '__opr_deleted__'}]
    finally:
        await driver.execute_query('MATCH (n {uuid: $uuid}) DETACH DELETE n', uuid=episode_uuid)
        await driver.close()


@pytest.mark.asyncio
async def test_falkor_conditional_delete_routes_group_and_preserves_tombstone():
    group_id = f'conditional{uuid4().hex}'
    base_driver = FalkorDriver(
        host=os.environ.get('FALKORDB_HOST', '127.0.0.1'),
        port=int(os.environ.get('FALKORDB_PORT', '6379')),
        database='default_db',
    )
    group_driver = base_driver.with_database(group_id)
    service = cast(ZepGraphiti, SimpleNamespace(driver=base_driver))
    episode_uuid = str(uuid4())
    episode = EpisodicNode(
        uuid=episode_uuid,
        group_id=group_id,
        name='curated:test.md',
        content='stored content',
        source_description='publish',
        source=EpisodeType.message,
        created_at=datetime.now(timezone.utc),
        valid_at=datetime.now(timezone.utc),
    )
    await episode.save(group_driver)

    try:
        assert not await service.delete_episodic_node_if_matches(
            episode_uuid,
            group_id=group_id,
            name=episode.name,
            content='changed content',
            source_description=episode.source_description,
        )
        assert await service.delete_episodic_node_if_matches(
            episode_uuid,
            group_id=group_id,
            name=episode.name,
            content=episode.content,
            source_description=episode.source_description,
        )

        await episode.delete(group_driver)
        with pytest.raises(EpisodeTombstonedError):
            await episode.save(group_driver)
        records, _, _ = await group_driver.execute_query(
            """
            MATCH (episode:Episodic {uuid: $uuid})
            RETURN episode.opr_deleted AS deleted,
                   episode.group_id AS group_id,
                   episode.content AS content
            """,
            uuid=episode_uuid,
        )
        assert records == [{'deleted': True, 'group_id': '__opr_deleted__', 'content': ''}]
    finally:
        await group_driver.execute_query(
            'MATCH (n:Episodic {uuid: $uuid}) DETACH DELETE n',
            uuid=episode_uuid,
        )
        await base_driver.close()
