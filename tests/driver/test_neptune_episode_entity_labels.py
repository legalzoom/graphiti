"""Synthetic storage contracts, not incident data or LLM-quality tests.

Exercise the episode producer and real Neptune query/session path. Only database and
search I/O are substituted, so the direct-operations helper cannot mask ingestion bugs.
"""

from collections import Counter
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.driver.neptune.operations.entity_node_ops import NeptuneEntityNodeOperations
from graphiti_core.driver.neptune_driver import NeptuneDriver
from graphiti_core.errors import EpisodeTombstonedError, NodeGroupMismatchError
from graphiti_core.graphiti import Graphiti
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def entity(uuid: str, labels: list[str]) -> EntityNode:
    return EntityNode(
        uuid=uuid,
        name=uuid,
        labels=labels,
        name_embedding=[1.0, 0.0],
        group_id='test-group',
        created_at=NOW,
    )


@pytest.fixture
def episode_storage():
    driver = object.__new__(NeptuneDriver)
    driver.graph_operations_interface = None
    driver.vector_projection_enabled = True
    driver.client = MagicMock()
    driver.save_to_aoss = MagicMock()
    driver.save_vector_to_aoss_async = AsyncMock(side_effect=lambda _, docs: len(docs))
    writer = object.__new__(Graphiti)
    writer.driver = driver
    writer.embedder = MagicMock()
    writer.store_raw_episode_content = True
    generations = Counter()

    def query_result(query, *, params):
        if 'nodes' in params:
            results = []
            for node in params['nodes']:
                generations[node['uuid']] += 1
                results.append(
                    {'uuid': node['uuid'], 'projection_version': generations[node['uuid']]}
                )
            return results
        return [{'uuid': episode['uuid']} for episode in params.get('episodes', [])]

    driver.client.query.side_effect = query_result
    return writer, driver


async def write_episode(writer: Graphiti, nodes: list[EntityNode]):
    episode = EpisodicNode(
        uuid='test-episode',
        name='synthetic storage contract',
        group_id='test-group',
        source=EpisodeType.text,
        source_description='synthetic',
        content='synthetic',
        created_at=NOW,
        valid_at=NOW,
    )
    return await writer._process_episode_data(episode, nodes, [], NOW, 'test-group')


def entity_writes(driver):
    return [call for call in driver.client.query.call_args_list if 'nodes' in call.kwargs['params']]


@pytest.mark.asyncio
@pytest.mark.parametrize('write_path', ['episode', 'operations', 'operations_session'])
async def test_write_isolates_labels_and_preserves_all_projection_generations(
    episode_storage,
    write_path,
):
    writer, driver = episode_storage
    nodes = [entity('node-a', ['CategoryA']), entity('node-b', ['CategoryB'])]

    if write_path == 'episode':
        await write_episode(writer, nodes)
    else:
        tx = driver.session() if write_path == 'operations_session' else None
        await NeptuneEntityNodeOperations(driver).save_bulk(driver, nodes, tx=tx)

    stored_labels = {node.uuid: set() for node in nodes}
    write_counts = Counter()
    for call in entity_writes(driver):
        labels = {
            line.strip().removeprefix('SET n:')
            for line in call.args[0].splitlines()
            if line.strip().startswith('SET n:')
        }
        for node in call.kwargs['params']['nodes']:
            stored_labels[node['uuid']].update(labels)
            write_counts[node['uuid']] += 1
    assert stored_labels == {
        'node-a': {'Entity', 'CategoryA'},
        'node-b': {'Entity', 'CategoryB'},
    }
    assert write_counts == {'node-a': 1, 'node-b': 1}
    documents = driver.save_vector_to_aoss_async.call_args.args[1]
    assert {doc['uuid']: doc['_version'] for doc in documents} == {'node-a': 1, 'node-b': 1}
    completions = [
        call.kwargs['params']['completed']
        for call in driver.client.query.call_args_list
        if 'completed' in call.kwargs['params']
    ]
    assert completions == [
        [
            {'uuid': 'node-a', 'projection_version': 1},
            {'uuid': 'node-b', 'projection_version': 1},
        ]
    ]


@pytest.mark.asyncio
async def test_episode_write_groups_equivalent_labels_without_repeated_writes(episode_storage):
    writer, driver = episode_storage
    nodes = [
        entity('node-a', ['CategoryA', 'CategoryB']),
        entity('node-b', ['CategoryB', 'CategoryA', 'CategoryA']),
    ]

    await write_episode(writer, nodes)

    writes = entity_writes(driver)
    assert len(writes) == 1
    assert [node['uuid'] for node in writes[0].kwargs['params']['nodes']] == ['node-a', 'node-b']


@pytest.mark.asyncio
async def test_episode_write_without_entities_has_no_entity_query_or_projection(episode_storage):
    writer, driver = episode_storage

    await write_episode(writer, [])

    assert entity_writes(driver) == []
    driver.save_vector_to_aoss_async.assert_not_awaited()
    assert [call.args[0] for call in driver.save_to_aoss.call_args_list] == ['episode_content']


@pytest.mark.asyncio
async def test_episode_write_duplicate_uuid_keeps_only_last_label_set(episode_storage):
    writer, driver = episode_storage

    await write_episode(writer, [entity('node-a', ['CategoryA']), entity('node-a', ['CategoryB'])])

    writes = entity_writes(driver)
    assert len(writes) == 1
    assert 'SET n:CategoryA' not in writes[0].args[0]
    assert 'SET n:CategoryB' in writes[0].args[0]
    assert len(writes[0].kwargs['params']['nodes']) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('rejected_uuid', ['node-a', 'node-b', 'node-c'])
async def test_episode_write_refuses_projection_when_a_label_group_is_rejected(
    episode_storage, rejected_uuid
):
    writer, driver = episode_storage
    original_query = driver.client.query.side_effect

    def reject_node(query, *, params):
        records = original_query(query, params=params)
        return [record for record in records if record['uuid'] != rejected_uuid]

    driver.client.query.side_effect = reject_node

    with pytest.raises(NodeGroupMismatchError):
        await write_episode(
            writer,
            [
                entity('node-a', ['CategoryA']),
                entity('node-b', ['CategoryB']),
                entity('node-c', ['CategoryC']),
            ],
        )

    driver.save_to_aoss.assert_not_called()
    driver.save_vector_to_aoss_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_episode_rejection_prevents_entity_queries(episode_storage):
    writer, driver = episode_storage
    driver.client.query.return_value = []
    driver.client.query.side_effect = None

    with pytest.raises(EpisodeTombstonedError):
        await write_episode(writer, [entity('node-a', ['CategoryA'])])

    assert entity_writes(driver) == []
    driver.save_to_aoss.assert_not_called()
    driver.save_vector_to_aoss_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_episode_write_query_failure_stops_later_groups_and_projection(episode_storage):
    writer, driver = episode_storage
    original_query = driver.client.query.side_effect

    def fail_entity_write(query, *, params):
        if 'nodes' in params:
            raise RuntimeError('synthetic database failure')
        return original_query(query, params=params)

    driver.client.query.side_effect = fail_entity_write

    with pytest.raises(RuntimeError, match='synthetic database failure'):
        await write_episode(
            writer, [entity('node-a', ['CategoryA']), entity('node-b', ['CategoryB'])]
        )

    assert len(entity_writes(driver)) == 1
    driver.save_to_aoss.assert_not_called()
    driver.save_vector_to_aoss_async.assert_not_awaited()
