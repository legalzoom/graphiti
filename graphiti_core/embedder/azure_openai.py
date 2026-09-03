"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from typing import Any

from openai import AsyncAzureOpenAI, AsyncOpenAI

from .client import EmbedderClient, EmbedderConfig

logger = logging.getLogger(__name__)


class AzureOpenAIEmbedderClient(EmbedderClient):
    """Wrapper class for Azure OpenAI that implements the EmbedderClient interface.

    Supports both AsyncAzureOpenAI and AsyncOpenAI (with Azure v1 API endpoint).
    """

    def __init__(
        self,
        azure_client: AsyncAzureOpenAI | AsyncOpenAI,
        model: str = 'text-embedding-3-small',
        embedding_dim: int | None = None,
        send_dimensions: bool = False,
    ):
        if embedding_dim is not None and embedding_dim <= 0:
            raise ValueError('embedding_dim must be greater than zero')
        self.azure_client = azure_client
        self.model = model
        self.embedding_dim = embedding_dim
        self.send_dimensions = send_dimensions
        # Preserve the historical direct-construction behavior when no dimension is supplied,
        # while exposing the configured output contract to Graphiti when one is explicit.
        self.config = (
            EmbedderConfig(embedding_dim=embedding_dim) if embedding_dim is not None else None
        )

    async def _create_embeddings(self, input_data: list[str]):
        kwargs: dict[str, Any] = {'model': self.model, 'input': input_data}
        if self.embedding_dim is not None and self.send_dimensions:
            kwargs['dimensions'] = self.embedding_dim
        return await self.azure_client.embeddings.create(**kwargs)

    def _normalize_embedding(self, embedding: list[float]) -> list[float]:
        """Enforce the configured output contract without requiring Azure API support."""
        if self.embedding_dim is None:
            return embedding
        if len(embedding) < self.embedding_dim:
            raise ValueError(
                f'Azure OpenAI returned {len(embedding)} embedding dimensions; '
                f'configured dimension is {self.embedding_dim}'
            )
        return embedding[: self.embedding_dim]

    async def create(self, input_data: str | list[str] | Any) -> list[float]:
        """Create embeddings using Azure OpenAI client."""
        try:
            # Handle different input types
            if isinstance(input_data, str):
                text_input = [input_data]
            elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
                text_input = input_data
            else:
                # Convert to string list for other types
                text_input = [str(input_data)]

            response = await self._create_embeddings(text_input)

            # Return the first embedding as a list of floats
            return self._normalize_embedding(response.data[0].embedding)
        except Exception as e:
            logger.error(f'Error in Azure OpenAI embedding: {e}')
            raise

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Create batch embeddings using Azure OpenAI client."""
        try:
            response = await self._create_embeddings(input_data_list)

            return [self._normalize_embedding(embedding.embedding) for embedding in response.data]
        except Exception as e:
            logger.error(f'Error in Azure OpenAI batch embedding: {e}')
            raise
