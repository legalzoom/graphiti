"""Synthetic SDK-boundary tests with no exporter, model or database traffic."""

import asyncio

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from graphiti_core.tracer import OpenTelemetryTracer


@pytest.fixture
def tracing():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider, exporter
    provider.shutdown()


def test_native_span_preserves_parent_and_attributes(tracing):
    provider, exporter = tracing
    sdk_tracer = provider.get_tracer('test')
    tracer = OpenTelemetryTracer(sdk_tracer)
    with sdk_tracer.start_as_current_span('request') as parent:
        with tracer.start_span('search') as span:
            span.add_attributes({'candidate_count': 2})
            span.set_status('ok')
        assert trace.get_current_span() is parent

    child, completed_parent = exporter.get_finished_spans()
    assert child.name == 'graphiti.search'
    assert child.parent.span_id == completed_parent.context.span_id
    assert child.attributes == {'candidate_count': 2}
    assert child.status.status_code == trace.StatusCode.OK


@pytest.mark.parametrize('error', [ValueError('synthetic failure'), asyncio.CancelledError()])
def test_native_span_preserves_original_operation_failure(tracing, error):
    provider, exporter = tracing
    tracer = OpenTelemetryTracer(provider.get_tracer('test'))

    with pytest.raises(type(error)) as raised, tracer.start_span('operation'):
        raise error

    assert raised.value is error
    assert [span.name for span in exporter.get_finished_spans()] == ['graphiti.operation']
    assert not trace.get_current_span().get_span_context().is_valid


@pytest.mark.asyncio
async def test_concurrent_operations_keep_separate_trace_parents(tracing):
    provider, exporter = tracing
    sdk_tracer = provider.get_tracer('test')
    tracer = OpenTelemetryTracer(sdk_tracer)

    async def operation(name):
        with sdk_tracer.start_as_current_span(name), tracer.start_span(name):
            await asyncio.sleep(0)

    await asyncio.gather(operation('request-a'), operation('request-b'))
    spans = {span.name: span for span in exporter.get_finished_spans()}
    for name in ('request-a', 'request-b'):
        assert spans[f'graphiti.{name}'].parent.span_id == spans[name].context.span_id
    assert spans['request-a'].context.trace_id != spans['request-b'].context.trace_id
