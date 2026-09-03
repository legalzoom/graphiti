from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient


@pytest.mark.asyncio
async def test_configured_dimension_is_exposed_and_sent_to_azure() -> None:
    client = MagicMock()
    client.embeddings.create = AsyncMock(
        return_value=SimpleNamespace(
            data=[SimpleNamespace(embedding=[1.0, 0.0])],
        )
    )
    embedder = AzureOpenAIEmbedderClient(
        azure_client=client,
        model='text-embedding-3-small',
        embedding_dim=2,
        send_dimensions=True,
    )

    result = await embedder.create(['hello'])

    assert result == [1.0, 0.0]
    assert embedder.config is not None
    assert embedder.config.embedding_dim == 2
    client.embeddings.create.assert_awaited_once_with(
        model='text-embedding-3-small',
        input=['hello'],
        dimensions=2,
    )


@pytest.mark.asyncio
async def test_legacy_construction_does_not_add_a_dimensions_argument() -> None:
    client = MagicMock()
    client.embeddings.create = AsyncMock(
        return_value=SimpleNamespace(
            data=[SimpleNamespace(embedding=[1.0, 0.0, 0.5])],
        )
    )
    embedder = AzureOpenAIEmbedderClient(azure_client=client)

    result = await embedder.create_batch(['hello'])

    assert result == [[1.0, 0.0, 0.5]]
    assert embedder.config is None
    client.embeddings.create.assert_awaited_once_with(
        model='text-embedding-3-small',
        input=['hello'],
    )


@pytest.mark.asyncio
async def test_configured_dimension_is_enforced_without_sending_optional_argument() -> None:
    client = MagicMock()
    client.embeddings.create = AsyncMock(
        return_value=SimpleNamespace(
            data=[SimpleNamespace(embedding=[1.0, 0.0, 0.5])],
        )
    )
    embedder = AzureOpenAIEmbedderClient(
        azure_client=client,
        model='legacy-deployment',
        embedding_dim=2,
    )

    result = await embedder.create_batch(['hello'])

    assert result == [[1.0, 0.0]]
    assert embedder.config is not None
    assert embedder.config.embedding_dim == 2
    client.embeddings.create.assert_awaited_once_with(
        model='legacy-deployment',
        input=['hello'],
    )


@pytest.mark.asyncio
async def test_short_azure_embedding_fails_before_reaching_vector_storage() -> None:
    client = MagicMock()
    client.embeddings.create = AsyncMock(
        return_value=SimpleNamespace(
            data=[SimpleNamespace(embedding=[1.0])],
        )
    )
    embedder = AzureOpenAIEmbedderClient(
        azure_client=client,
        embedding_dim=2,
    )

    with pytest.raises(ValueError, match='returned 1 embedding dimensions'):
        await embedder.create('hello')
