from __future__ import annotations

import traceback
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.driver import neptune_driver as neptune_driver_module
from graphiti_core.driver.neptune.operations.entity_edge_ops import (
    NeptuneEntityEdgeOperations,
)
from graphiti_core.driver.neptune.operations.entity_node_ops import (
    NeptuneEntityNodeOperations,
)
from graphiti_core.driver.neptune_driver import AossProjectionError, NeptuneDriver
from graphiti_core.driver.query_executor import QueryExecutor
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)

NODE_CANONICAL_ATTRIBUTES = (
    'uuid',
    'group_id',
    'name',
    'name_embedding',
    'summary',
    'created_at',
    'labels',
)

EDGE_CANONICAL_ATTRIBUTES = (
    'uuid',
    'group_id',
    'source_uuid',
    'target_uuid',
    'source_node_uuid',
    'target_node_uuid',
    'name',
    'fact',
    'fact_embedding',
    'episodes',
    'created_at',
    'expired_at',
    'valid_at',
    'invalid_at',
    'reference_time',
)


def _projection_driver() -> MagicMock:
    driver = MagicMock(spec=NeptuneDriver)
    driver.save_to_aoss = MagicMock(return_value=1)
    driver.save_vector_to_aoss_async = AsyncMock(return_value=1)
    return driver


def _executor(uuid: str) -> MagicMock:
    executor = MagicMock(spec=QueryExecutor)
    executor.execute_query = AsyncMock(
        return_value=([{'uuid': uuid, 'projection_version': 1}], None, None)
    )
    return executor


def _node(attribute_key: str) -> EntityNode:
    return EntityNode(
        uuid='node-1',
        name='node',
        group_id='group-1',
        labels=['Person'],
        created_at=NOW,
        name_embedding=[1.0, 0.0],
        attributes={attribute_key: 'attacker-controlled-value'},
    )


def _edge(attribute_key: str) -> EntityEdge:
    return EntityEdge(
        uuid='edge-1',
        name='RELATES_TO',
        fact='a fact',
        group_id='group-1',
        source_node_uuid='source-1',
        target_node_uuid='target-1',
        created_at=NOW,
        fact_embedding=[1.0, 0.0],
        attributes={attribute_key: 'attacker-controlled-value'},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('method_name', ['save', 'save_bulk'])
@pytest.mark.parametrize('attribute_key', NODE_CANONICAL_ATTRIBUTES)
async def test_node_save_rejects_canonical_attribute_collisions_before_writes(
    method_name: str,
    attribute_key: str,
) -> None:
    driver = _projection_driver()
    executor = _executor('node-1')
    operations = NeptuneEntityNodeOperations(driver)
    node = _node(attribute_key)

    with pytest.raises(ValueError, match=attribute_key):
        if method_name == 'save':
            await operations.save(executor, node)
        else:
            await operations.save_bulk(executor, [node])

    executor.execute_query.assert_not_awaited()
    driver.save_to_aoss.assert_not_called()
    driver.save_vector_to_aoss_async.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('method_name', ['save', 'save_bulk'])
@pytest.mark.parametrize('attribute_key', EDGE_CANONICAL_ATTRIBUTES)
async def test_edge_save_rejects_canonical_attribute_collisions_before_writes(
    method_name: str,
    attribute_key: str,
) -> None:
    driver = _projection_driver()
    executor = _executor('edge-1')
    operations = NeptuneEntityEdgeOperations(driver)
    edge = _edge(attribute_key)

    with pytest.raises(ValueError, match=attribute_key):
        if method_name == 'save':
            await operations.save(executor, edge)
        else:
            await operations.save_bulk(executor, [edge])

    executor.execute_query.assert_not_awaited()
    driver.save_to_aoss.assert_not_called()
    driver.save_vector_to_aoss_async.assert_not_awaited()


def _text_aoss_driver() -> NeptuneDriver:
    driver = object.__new__(NeptuneDriver)
    driver._aoss_indices = [
        {
            'index_name': 'secure_text',
            'body': {
                'mappings': {
                    'properties': {
                        'uuid': {'type': 'keyword'},
                        'content': {'type': 'text'},
                    }
                }
            },
        }
    ]
    driver.aoss_client = MagicMock()
    return driver


def test_save_to_aoss_logs_only_sanitized_bulk_failure_metadata(caplog) -> None:
    driver = _text_aoss_driver()
    document_secret = 'document-body-secret-7c53'
    reason_secret = 'mapping-reason-secret-a8d1'
    failure = {
        'index': {
            '_id': 'safe-document-id',
            'status': 400,
            'error': {
                'type': 'mapper_parsing_exception',
                'reason': reason_secret,
            },
        },
        'data': {'content': document_secret},
    }
    caplog.set_level('ERROR', logger='graphiti_core.driver.neptune_driver')

    with patch.object(
        neptune_driver_module.helpers,
        'bulk',
        return_value=(0, [failure]),
    ):
        saved = driver.save_to_aoss(
            'secure_text',
            [{'uuid': 'safe-document-id', 'content': document_secret}],
        )

    assert saved == 0
    assert 'safe-document-id' in caplog.text
    assert '400' in caplog.text
    assert 'mapper_parsing_exception' in caplog.text
    assert document_secret not in caplog.text
    assert reason_secret not in caplog.text


class SecretTransportFailure(RuntimeError):
    pass


def test_save_to_aoss_logs_transport_type_without_exception_or_document_secrets(caplog) -> None:
    driver = _text_aoss_driver()
    document_secret = 'document-body-secret-91bf'
    transport_secret = 'transport-credential-secret-c4e2'
    caplog.set_level('ERROR', logger='graphiti_core.driver.neptune_driver')

    with patch.object(
        neptune_driver_module.helpers,
        'bulk',
        side_effect=SecretTransportFailure(transport_secret),
    ):
        saved = driver.save_to_aoss(
            'secure_text',
            [{'uuid': 'safe-document-id', 'content': document_secret}],
        )

    assert saved == 0
    assert 'error_type=SecretTransportFailure' in caplog.text
    assert document_secret not in caplog.text
    assert transport_secret not in caplog.text


def _vector_aoss_driver(*, index_exists: bool) -> NeptuneDriver:
    driver = object.__new__(NeptuneDriver)
    driver.vector_projection_enabled = False
    driver._vector_aoss_indices = neptune_driver_module._vector_aoss_indices(2)
    driver.aoss_client = MagicMock()
    driver.vector_aoss_client = MagicMock()
    driver.vector_aoss_client.indices.exists.return_value = index_exists
    return driver


@pytest.mark.parametrize(
    ('index_name', 'document_id'),
    [
        ('node_name_embedding', 'node-1'),
        ('edge_fact_embedding', 'edge-1'),
    ],
)
def test_disabled_projection_delete_still_writes_exact_version_tombstone(
    index_name: str,
    document_id: str,
) -> None:
    driver = _vector_aoss_driver(index_exists=True)

    with patch.object(
        neptune_driver_module.helpers,
        'bulk',
        return_value=(1, []),
    ) as bulk:
        deleted = driver.delete_from_aoss(
            index_name,
            uuids=[document_id],
            versions={document_id: 17},
        )

    assert deleted == 1
    assert bulk.call_args.args[0] is driver.vector_aoss_client
    assert bulk.call_args.args[1] == [
        {
            '_op_type': 'index',
            '_index': index_name,
            '_id': document_id,
            '_version': neptune_driver_module.vector_aoss_external_version(17),
            '_version_type': 'external',
            'uuid': document_id,
            neptune_driver_module.VECTOR_AOSS_TOMBSTONE_FIELD: True,
        }
    ]


def test_delete_from_missing_vector_index_is_already_cleaned() -> None:
    driver = _vector_aoss_driver(index_exists=False)

    with patch.object(neptune_driver_module.helpers, 'bulk') as bulk:
        deleted = driver.delete_from_aoss(
            'node_name_embedding',
            uuids=['node-1', 'node-2'],
            versions={'node-1': 4, 'node-2': 9},
        )

    assert deleted == 2
    driver.vector_aoss_client.indices.exists.assert_called_once_with(index='node_name_embedding')
    bulk.assert_not_called()


def test_delete_failure_does_not_expose_bulk_document_or_reason() -> None:
    driver = _vector_aoss_driver(index_exists=True)
    document_secret = 'delete-document-secret-61c2'
    reason_secret = 'delete-reason-secret-bd39'
    failure = {
        'index': {
            '_id': 'node-1',
            'status': 400,
            'error': {'type': 'mapper_parsing_exception', 'reason': reason_secret},
        },
        'data': {'embedding': document_secret},
    }

    with (
        patch.object(neptune_driver_module.helpers, 'bulk', return_value=(0, [failure])),
        pytest.raises(AossProjectionError) as exc_info,
    ):
        driver.delete_from_aoss(
            'node_name_embedding',
            uuids=['node-1'],
            versions={'node-1': 1},
        )

    message = str(exc_info.value)
    assert 'node-1' in message
    assert 'mapper_parsing_exception' in message
    assert document_secret not in message
    assert reason_secret not in message


def test_quiesced_purge_failure_does_not_expose_bulk_document_or_reason() -> None:
    driver = _vector_aoss_driver(index_exists=True)
    document_secret = 'purge-document-secret-735b'
    reason_secret = 'purge-reason-secret-51ae'
    driver.vector_aoss_client.search.return_value = {
        'hits': {'hits': [{'_id': 'node-1', 'sort': ['node-1']}]}
    }
    failure = {
        'delete': {
            '_id': 'node-1',
            'status': 400,
            'error': {'type': 'illegal_argument_exception', 'reason': reason_secret},
        },
        'data': {'embedding': document_secret},
    }

    with (
        patch.object(neptune_driver_module.helpers, 'bulk', return_value=(0, [failure])),
        pytest.raises(AossProjectionError) as exc_info,
    ):
        driver._purge_aoss_query('node_name_embedding', {'match_all': {}})

    message = str(exc_info.value)
    assert 'node-1' in message
    assert 'illegal_argument_exception' in message
    assert document_secret not in message
    assert reason_secret not in message


def test_delete_transport_failure_does_not_survive_exception_chaining() -> None:
    driver = _vector_aoss_driver(index_exists=True)
    transport_secret = 'delete-transport-credential-secret-8fd1'

    with (
        patch.object(
            neptune_driver_module.helpers,
            'bulk',
            side_effect=SecretTransportFailure(transport_secret),
        ),
        pytest.raises(AossProjectionError) as exc_info,
    ):
        driver.delete_from_aoss(
            'node_name_embedding',
            uuids=['node-1'],
            versions={'node-1': 1},
        )

    rendered = ''.join(traceback.format_exception(exc_info.value))
    assert 'error_type=SecretTransportFailure' in rendered
    assert transport_secret not in rendered
