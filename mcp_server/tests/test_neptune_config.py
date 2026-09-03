import asyncio
import sys
from types import ModuleType
from unittest.mock import AsyncMock, Mock

import pytest
from graphiti_core.driver.driver import GraphProvider

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
                vector_aoss_host='vector-search.example',
                port=8282,
                aoss_port=8443,
                vector_aoss_port=9443,
            )
        ),
    )


def test_neptune_factory_preserves_configured_endpoints(monkeypatch):
    for name in (
        'NEPTUNE_HOST',
        'AOSS_HOST',
        'VECTOR_AOSS_HOST',
        'NEPTUNE_PORT',
        'AOSS_PORT',
        'VECTOR_AOSS_PORT',
        'NEPTUNE_VECTOR_PROJECTION_ENABLED',
        'NEPTUNE_VECTOR_SEARCH_ENABLED',
        'NEPTUNE_VECTOR_RECONCILE_INTERVAL_SECONDS',
    ):
        monkeypatch.delenv(name, raising=False)

    assert DatabaseDriverFactory.create_config(_database_config()) == {
        'driver': 'neptune',
        'host': 'neptune-db://cluster.example',
        'aoss_host': 'search.example',
        'vector_aoss_host': 'vector-search.example',
        'port': 8282,
        'aoss_port': 8443,
        'vector_aoss_port': 9443,
        'vector_projection_enabled': False,
        'vector_search_enabled': False,
        'vector_reconcile_interval_seconds': 30.0,
    }


def test_neptune_factory_applies_environment_overrides(monkeypatch):
    monkeypatch.setenv('NEPTUNE_HOST', 'neptune-graph://override')
    monkeypatch.setenv('AOSS_HOST', 'override-search.example')
    monkeypatch.setenv('VECTOR_AOSS_HOST', 'override-vector.example')
    monkeypatch.setenv('NEPTUNE_PORT', '8183')
    monkeypatch.setenv('AOSS_PORT', '444')
    monkeypatch.setenv('VECTOR_AOSS_PORT', '445')
    monkeypatch.setenv('NEPTUNE_VECTOR_PROJECTION_ENABLED', 'true')
    monkeypatch.setenv('NEPTUNE_VECTOR_SEARCH_ENABLED', 'true')
    monkeypatch.setenv('NEPTUNE_VECTOR_RECONCILE_INTERVAL_SECONDS', '17.5')

    assert DatabaseDriverFactory.create_config(_database_config()) == {
        'driver': 'neptune',
        'host': 'neptune-graph://override',
        'aoss_host': 'override-search.example',
        'vector_aoss_host': 'override-vector.example',
        'port': 8183,
        'aoss_port': 444,
        'vector_aoss_port': 445,
        'vector_projection_enabled': True,
        'vector_search_enabled': True,
        'vector_reconcile_interval_seconds': 17.5,
    }


def test_neptune_factory_inherits_custom_primary_port_without_vector_host(monkeypatch):
    monkeypatch.delenv('AOSS_PORT', raising=False)
    monkeypatch.delenv('VECTOR_AOSS_HOST', raising=False)
    monkeypatch.delenv('VECTOR_AOSS_PORT', raising=False)

    config = _database_config()
    assert config.providers.neptune is not None
    config.providers.neptune.vector_aoss_host = None

    db_config = DatabaseDriverFactory.create_config(config)

    assert db_config['vector_aoss_host'] is None
    assert db_config['vector_aoss_port'] == 8443


def test_neptune_factory_defaults_separate_vector_host_port_to_443(monkeypatch):
    monkeypatch.delenv('VECTOR_AOSS_HOST', raising=False)
    monkeypatch.delenv('VECTOR_AOSS_PORT', raising=False)

    config = _database_config()
    assert config.providers.neptune is not None
    config.providers.neptune.vector_aoss_port = None

    db_config = DatabaseDriverFactory.create_config(config)

    assert db_config['vector_aoss_host'] == 'vector-search.example'
    assert db_config['vector_aoss_port'] == 443


def test_neptune_vector_port_is_optional_in_schema():
    assert NeptuneProviderConfig().vector_aoss_port is None


def test_neptune_factory_uses_separate_vector_host_port_override(monkeypatch):
    monkeypatch.delenv('VECTOR_AOSS_HOST', raising=False)
    monkeypatch.setenv('VECTOR_AOSS_PORT', '9444')

    config = _database_config()
    assert config.providers.neptune is not None
    config.providers.neptune.vector_aoss_port = None

    db_config = DatabaseDriverFactory.create_config(config)

    assert db_config['vector_aoss_host'] == 'vector-search.example'
    assert db_config['vector_aoss_port'] == 9444


@pytest.mark.asyncio
async def test_graphiti_service_wires_neptune_driver_and_main_reranker(monkeypatch):
    llm_client = object()
    embedder_client = object()
    cross_encoder_client = object()
    driver = Mock(provider=GraphProvider.NEPTUNE, vector_projection_enabled=False)
    client = Mock(driver=driver)
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
    reconciler = AsyncMock()
    monkeypatch.setattr(
        graphiti_mcp_server,
        'run_pending_projection_reconciler',
        reconciler,
    )

    service = graphiti_mcp_server.GraphitiService(
        GraphitiConfig(database=_database_config()), semaphore_limit=7
    )
    await service.initialize()
    await asyncio.sleep(0)

    driver_constructor.assert_called_once_with(
        host='neptune-db://cluster.example',
        aoss_host='search.example',
        port=8282,
        aoss_port=8443,
        vector_aoss_host='vector-search.example',
        vector_aoss_port=9443,
        embedding_dim=1536,
        vector_search_enabled=False,
        vector_projection_enabled=False,
    )
    graphiti_constructor.assert_called_once_with(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=embedder_client,
        cross_encoder=cross_encoder_client,
        max_coroutines=7,
    )
    client.build_indices_and_constraints.assert_awaited_once_with()
    assert service.vector_reconciler is not None
    reconciler.assert_awaited_once_with(driver, 30.0)


@pytest.mark.asyncio
async def test_graphiti_service_starts_reconciler_with_interval_and_stops_before_close(
    monkeypatch,
):
    monkeypatch.delenv('NEPTUNE_VECTOR_PROJECTION_ENABLED', raising=False)
    monkeypatch.delenv('NEPTUNE_VECTOR_RECONCILE_INTERVAL_SECONDS', raising=False)
    config = GraphitiConfig(database=_database_config())
    assert config.database.providers.neptune is not None
    config.database.providers.neptune.vector_projection_enabled = False
    config.database.providers.neptune.vector_reconcile_interval_seconds = 12.5

    driver = Mock(provider=GraphProvider.NEPTUNE, vector_projection_enabled=False)
    events: list[str] = []
    started = asyncio.Event()

    async def close_client() -> None:
        events.append('client-closed')

    client = Mock(driver=driver)
    client.build_indices_and_constraints = AsyncMock()
    client.close = AsyncMock(side_effect=close_client)
    driver_constructor = Mock(return_value=driver)
    neptune_module = ModuleType('graphiti_core.driver.neptune_driver')
    neptune_module.NeptuneDriver = driver_constructor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, neptune_module.__name__, neptune_module)
    monkeypatch.setattr(LLMClientFactory, 'create', Mock(return_value=None))
    monkeypatch.setattr(EmbedderFactory, 'create', Mock(return_value=None))
    monkeypatch.setattr(CrossEncoderFactory, 'create', Mock(return_value=object()))
    monkeypatch.setattr(graphiti_mcp_server, 'Graphiti', Mock(return_value=client))
    reconciler_calls: list[tuple[object, float]] = []

    async def run_reconciler(reconciler_driver, interval_seconds: float) -> None:
        reconciler_calls.append((reconciler_driver, interval_seconds))
        events.append('reconciler-started')
        started.set()
        try:
            await asyncio.Future()
        finally:
            events.append('reconciler-cancelled')

    monkeypatch.setattr(
        graphiti_mcp_server,
        'run_pending_projection_reconciler',
        run_reconciler,
    )

    service = graphiti_mcp_server.GraphitiService(config)
    await service.initialize()
    await asyncio.wait_for(started.wait(), timeout=1)

    assert reconciler_calls == [(driver, 12.5)]
    assert service.vector_reconciler is not None
    assert service.vector_reconciler.get_name() == 'neptune-vector-projection-reconciler'

    await service.shutdown()

    assert events == ['reconciler-started', 'reconciler-cancelled', 'client-closed']
    client.close.assert_awaited_once_with()
    assert service.vector_reconciler is None
    assert service.client is None


@pytest.mark.asyncio
async def test_graphiti_service_does_not_start_neptune_reconciler_for_other_providers(
    monkeypatch,
):
    driver = Mock(provider=GraphProvider.NEO4J, vector_projection_enabled=True)
    client = Mock(driver=driver)
    client.build_indices_and_constraints = AsyncMock()
    client.close = AsyncMock()
    monkeypatch.setattr(LLMClientFactory, 'create', Mock(return_value=None))
    monkeypatch.setattr(EmbedderFactory, 'create', Mock(return_value=None))
    monkeypatch.setattr(CrossEncoderFactory, 'create', Mock(return_value=object()))
    monkeypatch.setattr(
        DatabaseDriverFactory,
        'create_config',
        Mock(return_value={'uri': 'bolt://neo4j', 'user': 'neo4j', 'password': 'secret'}),
    )
    monkeypatch.setattr(graphiti_mcp_server, 'Graphiti', Mock(return_value=client))
    reconciler = AsyncMock()
    monkeypatch.setattr(
        graphiti_mcp_server,
        'run_pending_projection_reconciler',
        reconciler,
    )
    config = GraphitiConfig()
    config.database.provider = 'neo4j'

    service = graphiti_mcp_server.GraphitiService(config)
    await service.initialize()

    assert service.vector_reconciler is None
    reconciler.assert_not_awaited()
    await service.shutdown()


@pytest.mark.asyncio
async def test_graphiti_service_leaves_client_open_for_detached_queue_worker(
    monkeypatch,
):
    driver = Mock(provider=GraphProvider.NEO4J, vector_projection_enabled=False)
    client = Mock(driver=driver)
    client.build_indices_and_constraints = AsyncMock()
    client.close = AsyncMock()
    monkeypatch.setattr(LLMClientFactory, 'create', Mock(return_value=None))
    monkeypatch.setattr(EmbedderFactory, 'create', Mock(return_value=None))
    monkeypatch.setattr(CrossEncoderFactory, 'create', Mock(return_value=object()))
    monkeypatch.setattr(
        DatabaseDriverFactory,
        'create_config',
        Mock(return_value={'uri': 'bolt://neo4j', 'user': 'neo4j', 'password': 'secret'}),
    )
    monkeypatch.setattr(graphiti_mcp_server, 'Graphiti', Mock(return_value=client))
    config = GraphitiConfig()
    config.database.provider = 'neo4j'

    service = graphiti_mcp_server.GraphitiService(config)
    await service.initialize()
    await service.shutdown(close_client=False)

    client.close.assert_not_awaited()
    assert service.client is None
