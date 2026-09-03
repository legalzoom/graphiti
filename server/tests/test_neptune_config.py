import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from graph_service import zep_graphiti
from graph_service.config import Settings


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


def _install_fake_neptune_module(monkeypatch, constructor: Mock) -> None:
    module = ModuleType('graphiti_core.driver.neptune_driver')
    module.NeptuneDriver = constructor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_neptune_backend_builds_driver_with_configured_endpoints(monkeypatch):
    driver = object()
    driver_constructor = Mock(return_value=driver)
    graphiti_constructor = Mock(return_value=object())
    _install_fake_neptune_module(monkeypatch, driver_constructor)
    monkeypatch.setattr(zep_graphiti, 'ZepGraphiti', graphiti_constructor)

    zep_graphiti._create_graphiti_client(
        _settings(
            neptune_port=8282,
            aoss_port=8443,
            vector_aoss_host='vector-search.example',
            vector_aoss_port=9443,
            neptune_vector_projection_enabled=True,
            neptune_vector_search_enabled=True,
        )
    )

    driver_constructor.assert_called_once_with(
        host='neptune-db://cluster.example',
        aoss_host='search.example',
        port=8282,
        aoss_port=8443,
        vector_aoss_host='vector-search.example',
        vector_aoss_port=9443,
        vector_search_enabled=True,
        vector_projection_enabled=True,
    )
    graphiti_constructor.assert_called_once_with(graph_driver=driver)


@pytest.mark.parametrize(
    ('neptune_host', 'aoss_host'),
    [(None, 'search.example'), ('neptune-db://cluster.example', None)],
)
def test_neptune_backend_requires_both_endpoints(monkeypatch, neptune_host, aoss_host):
    _install_fake_neptune_module(monkeypatch, Mock())

    with pytest.raises(ValueError, match='NEPTUNE_HOST and AOSS_HOST are required'):
        zep_graphiti._create_graphiti_client(
            _settings(neptune_host=neptune_host, aoss_host=aoss_host)
        )
