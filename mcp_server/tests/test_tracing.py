"""Synthetic MCP composition tests using real Graphiti and in-memory SDK spans."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.graphiti import Graphiti
from graphiti_core.llm_client.client import LLMClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import graphiti_mcp_server
from config.schema import DatabaseConfig, GraphitiConfig


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('provider_name', 'driver_module', 'driver_name'),
    [
        ('neo4j', 'graphiti_core.graphiti', 'Neo4jDriver'),
        ('falkordb', 'graphiti_core.driver.falkordb_driver', 'FalkorDriver'),
        ('neptune', 'graphiti_core.driver.neptune_driver', 'NeptuneDriver'),
    ],
)
async def test_service_emits_native_search_spans(
    monkeypatch,
    provider_name,
    driver_module,
    driver_name,
):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, 'get_tracer', provider.get_tracer)

    driver = Mock(spec=GraphDriver)
    driver.provider = GraphProvider(provider_name)
    driver.embedding_dim = None
    driver.clone.return_value = driver
    driver.search_interface = SimpleNamespace(
        edge_fulltext_search=AsyncMock(return_value=[]),
        edge_similarity_search=AsyncMock(return_value=[]),
    )
    # Optional database SDKs are outside this composition test. Keep Graphiti
    # itself real, but replace driver imports before loading their dependencies.
    if driver_module == 'graphiti_core.graphiti':
        module = sys.modules[driver_module]
    else:
        module = ModuleType(driver_module)
        monkeypatch.setitem(sys.modules, driver_module, module)
    monkeypatch.setattr(module, driver_name, Mock(return_value=driver), raising=False)
    llm = Mock(spec=LLMClient)
    embedder = Mock(spec=EmbedderClient)
    embedder.create = AsyncMock(return_value=[0.1, 0.2])
    cross_encoder = Mock(spec=CrossEncoderClient)
    monkeypatch.setattr(graphiti_mcp_server.LLMClientFactory, 'create', Mock(return_value=llm))
    monkeypatch.setattr(graphiti_mcp_server.EmbedderFactory, 'create', Mock(return_value=embedder))
    monkeypatch.setattr(
        graphiti_mcp_server.CrossEncoderFactory,
        'create',
        Mock(return_value=cross_encoder),
    )
    monkeypatch.setattr(
        graphiti_mcp_server.DatabaseDriverFactory,
        'create_config',
        Mock(
            return_value={
                'uri': 'bolt://database.example',
                'user': 'test',
                'password': 'test',
                'host': 'database.example',
                'port': 8182,
                'database': 'test',
                'aoss_host': 'search.example',
                'aoss_port': 443,
                'vector_aoss_host': 'vector.example',
                'vector_aoss_port': 443,
                'vector_search_enabled': False,
                'vector_projection_enabled': False,
            }
        ),
    )
    monkeypatch.setattr(Graphiti, 'build_indices_and_constraints', AsyncMock())

    try:
        service = graphiti_mcp_server.GraphitiService(
            GraphitiConfig(database=DatabaseConfig(provider=provider_name)),
            start_background_tasks=False,
        )
        await service.initialize()
        client = await service.get_client()
        llm.set_tracer.assert_called_once_with(client.tracer)
        assert client.clients.tracer is client.tracer

        with provider.get_tracer('test-request').start_as_current_span('request') as parent:
            assert await client.search('synthetic private query', group_ids=['test-group']) == []

        spans = exporter.get_finished_spans()
        edge_span = next(span for span in spans if span.name == 'graphiti.search.edge_search')
        assert edge_span.context is not None
        assert edge_span.attributes is not None
        assert edge_span.context.trace_id == parent.get_span_context().trace_id
        assert edge_span.attributes['candidate_count'] == 0
        assert edge_span.attributes['returned_count'] == 0
        assert all('synthetic private query' not in str(span.attributes) for span in spans)
    finally:
        provider.shutdown()
