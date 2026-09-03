import sys
from types import ModuleType
from unittest.mock import AsyncMock, Mock

import pytest

from graph_service import config, verify_vector_indices
from graph_service.config import Settings


class _FakeVectorIndexUnsupportedError(RuntimeError):
    pass


def _settings(**overrides) -> Settings:
    values = {
        'openai_api_key': 'test-key',
        'ingest_queue_maxsize': 1000,
        'db_backend': 'neptune',
        'neptune_host': 'neptune-db://cluster.example',
        'aoss_host': 'search.example',
    }
    values.update(overrides)
    return Settings.model_validate(values)


def _install_fake_driver(monkeypatch, driver) -> Mock:
    # graphiti_core.driver.neptune_driver requires boto3/opensearch-py, which are not
    # installed in this package's dev environment. Replace the module wholesale, the
    # same way test_neptune_config.py does for _create_graphiti_client.
    constructor = Mock(return_value=driver)
    module = ModuleType('graphiti_core.driver.neptune_driver')
    module.NeptuneDriver = constructor  # type: ignore[attr-defined]
    module.VectorIndexUnsupportedError = _FakeVectorIndexUnsupportedError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return constructor


@pytest.mark.asyncio
async def test_pass_when_vector_indices_are_created(monkeypatch):
    driver = Mock()
    driver.create_vector_aoss_indices = AsyncMock(return_value=None)
    driver.close = AsyncMock(return_value=None)
    constructor = _install_fake_driver(monkeypatch, driver)
    monkeypatch.setattr(config, 'get_settings', lambda: _settings())

    exit_code = await verify_vector_indices._main_async()

    assert exit_code == 0
    constructor.assert_called_once_with(
        host='neptune-db://cluster.example',
        aoss_host='search.example',
        port=8182,
        aoss_port=443,
    )
    driver.create_vector_aoss_indices.assert_awaited_once()
    driver.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_fail_reports_exact_rejection_and_still_closes(monkeypatch):
    driver = Mock()
    driver.create_vector_aoss_indices = AsyncMock(
        side_effect=_FakeVectorIndexUnsupportedError(
            "OpenSearch host 'search.example' rejected creating vector index "
            "'node_name_embedding': illegal_argument_exception. The AOSS collection "
            'must be of type VECTORSEARCH.'
        )
    )
    driver.close = AsyncMock(return_value=None)
    _install_fake_driver(monkeypatch, driver)
    monkeypatch.setattr(config, 'get_settings', lambda: _settings())

    exit_code = await verify_vector_indices._main_async()

    assert exit_code == 1
    driver.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_requires_neptune_backend(monkeypatch):
    monkeypatch.setattr(config, 'get_settings', lambda: _settings(db_backend='neo4j'))

    exit_code = await verify_vector_indices._main_async()

    assert exit_code == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('neptune_host', 'aoss_host'),
    [(None, 'search.example'), ('neptune-db://cluster.example', None)],
)
async def test_requires_both_endpoints(monkeypatch, neptune_host, aoss_host):
    monkeypatch.setattr(
        config,
        'get_settings',
        lambda: _settings(neptune_host=neptune_host, aoss_host=aoss_host),
    )

    exit_code = await verify_vector_indices._main_async()

    assert exit_code == 1
