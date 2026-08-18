import sys
from types import ModuleType
from unittest.mock import AsyncMock, Mock

import pytest

import graphiti_mcp_server
from config.schema import (
    DatabaseConfig,
    DatabaseProvidersConfig,
    GraphitiConfig,
    NeptuneProviderConfig,
)
from services.factories import (
    CrossEncoderFactory,
    DatabaseDriverFactory,
    EmbedderFactory,
    LLMClientFactory,
)


def _database_config() -> DatabaseConfig:
    return DatabaseConfig(
        provider='neptune',
        providers=DatabaseProvidersConfig(
            neptune=NeptuneProviderConfig(
                host='neptune-db://cluster.example',
                aoss_host='search.example',
                port=8282,
                aoss_port=8443,
            )
        ),
    )


def test_neptune_factory_preserves_configured_endpoints(monkeypatch):
    for name in ('NEPTUNE_HOST', 'AOSS_HOST', 'NEPTUNE_PORT', 'AOSS_PORT'):
        monkeypatch.delenv(name, raising=False)

    assert DatabaseDriverFactory.create_config(_database_config()) == {
        'driver': 'neptune',
        'host': 'neptune-db://cluster.example',
        'aoss_host': 'search.example',
        'port': 8282,
        'aoss_port': 8443,
    }


def test_neptune_factory_applies_environment_overrides(monkeypatch):
    monkeypatch.setenv('NEPTUNE_HOST', 'neptune-graph://override')
    monkeypatch.setenv('AOSS_HOST', 'override-search.example')
    monkeypatch.setenv('NEPTUNE_PORT', '8183')
    monkeypatch.setenv('AOSS_PORT', '444')

    assert DatabaseDriverFactory.create_config(_database_config()) == {
        'driver': 'neptune',
        'host': 'neptune-graph://override',
        'aoss_host': 'override-search.example',
        'port': 8183,
        'aoss_port': 444,
    }


@pytest.mark.asyncio
async def test_graphiti_service_wires_neptune_driver_and_main_reranker(monkeypatch):
    llm_client = object()
    embedder_client = object()
    cross_encoder_client = object()
    driver = object()
    client = Mock()
    client.build_indices_and_constraints = AsyncMock()
    driver_constructor = Mock(return_value=driver)
    graphiti_constructor = Mock(return_value=client)

    neptune_module = ModuleType('graphiti_core.driver.neptune_driver')
    neptune_module.NeptuneDriver = driver_constructor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, neptune_module.__name__, neptune_module)
    monkeypatch.setattr(LLMClientFactory, 'create', Mock(return_value=llm_client))
    monkeypatch.setattr(EmbedderFactory, 'create', Mock(return_value=embedder_client))
    monkeypatch.setattr(
        CrossEncoderFactory,
        'create',
        Mock(return_value=cross_encoder_client),
    )
    monkeypatch.setattr(graphiti_mcp_server, 'Graphiti', graphiti_constructor)

    service = graphiti_mcp_server.GraphitiService(
        GraphitiConfig(database=_database_config()), semaphore_limit=7
    )
    await service.initialize()

    driver_constructor.assert_called_once_with(
        host='neptune-db://cluster.example',
        aoss_host='search.example',
        port=8282,
        aoss_port=8443,
    )
    graphiti_constructor.assert_called_once_with(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=embedder_client,
        cross_encoder=cross_encoder_client,
        max_coroutines=7,
    )
    client.build_indices_and_constraints.assert_awaited_once_with()
