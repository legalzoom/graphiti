"""
Copyright 2025, Zep Software, Inc.

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

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from logging import INFO

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.driver.neptune_driver import NeptuneDriver
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

#################################################
# CONFIGURATION
#################################################
# Set up logging and environment variables for
# connecting to Neptune database
#################################################

# Configure logging
logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv()


def _boolean_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, str(default)).strip().lower()
    if value not in {'true', 'false'}:
        raise ValueError(f'{name} must be true or false')
    return value == 'true'


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f'{name} must be set')
    return value


# Neptune and OpenSearch connection parameters
neptune_uri = _required_env('NEPTUNE_HOST')
neptune_port = int(os.environ.get('NEPTUNE_PORT', 8182))
aoss_host = _required_env('AOSS_HOST')
aoss_port = int(os.environ.get('AOSS_PORT', 443))
vector_aoss_host = os.environ.get('VECTOR_AOSS_HOST')
vector_aoss_port = int(os.environ.get('VECTOR_AOSS_PORT', aoss_port))
vector_projection_enabled = _boolean_env('NEPTUNE_VECTOR_PROJECTION_ENABLED')
vector_search_enabled = _boolean_env('NEPTUNE_VECTOR_SEARCH_ENABLED')


if (vector_projection_enabled or vector_search_enabled) and not vector_aoss_host:
    raise ValueError(
        'VECTOR_AOSS_HOST must identify a VECTORSEARCH collection when vector projections '
        'or reads are enabled'
    )


async def main():
    #################################################
    # INITIALIZATION
    #################################################
    # Connect to Neptune and set up Graphiti indices
    # This is required before using other Graphiti
    # functionality
    #################################################

    # Initialize Graphiti with Neptune connection
    driver = NeptuneDriver(
        host=neptune_uri,
        aoss_host=aoss_host,
        port=neptune_port,
        aoss_port=aoss_port,
        vector_aoss_host=vector_aoss_host,
        vector_aoss_port=vector_aoss_port,
        vector_projection_enabled=vector_projection_enabled,
        vector_search_enabled=vector_search_enabled,
    )

    graphiti = Graphiti(graph_driver=driver)

    try:
        # Idempotently create missing indexes. The quickstart intentionally preserves existing
        # graph data and indexes; vector index deletion is an explicit rollback operation.
        await graphiti.build_indices_and_constraints()

        #################################################
        # ADDING EPISODES
        #################################################
        # Episodes are the primary units of information
        # in Graphiti. They can be text or structured JSON
        # and are automatically processed to extract entities
        # and relationships.
        #################################################

        # Example: Add Episodes
        # Episodes list containing both text and JSON episodes
        episodes = [
            {
                'content': 'Kamala Harris is the Attorney General of California. She was previously '
                'the district attorney for San Francisco.',
                'type': EpisodeType.text,
                'description': 'podcast transcript',
            },
            {
                'content': 'As AG, Harris was in office from January 3, 2011 – January 3, 2017',
                'type': EpisodeType.text,
                'description': 'podcast transcript',
            },
            {
                'content': {
                    'name': 'Gavin Newsom',
                    'position': 'Governor',
                    'state': 'California',
                    'previous_role': 'Lieutenant Governor',
                    'previous_location': 'San Francisco',
                },
                'type': EpisodeType.json,
                'description': 'podcast metadata',
            },
            {
                'content': {
                    'name': 'Gavin Newsom',
                    'position': 'Governor',
                    'term_start': 'January 7, 2019',
                    'term_end': 'Present',
                },
                'type': EpisodeType.json,
                'description': 'podcast metadata',
            },
        ]

        # Add episodes to the graph
        for i, episode in enumerate(episodes):
            await graphiti.add_episode(
                name=f'Freakonomics Radio {i}',
                episode_body=episode['content']
                if isinstance(episode['content'], str)
                else json.dumps(episode['content']),
                source=episode['type'],
                source_description=episode['description'],
                reference_time=datetime.now(timezone.utc),
            )
            print(f'Added episode: Freakonomics Radio {i} ({episode["type"].value})')

        await graphiti.build_communities()

        #################################################
        # BASIC SEARCH
        #################################################
        # The simplest way to retrieve relationships (edges)
        # from Graphiti is using the search method, which
        # performs a hybrid search combining semantic
        # similarity and BM25 text retrieval.
        #################################################

        # Perform a hybrid search combining semantic similarity and BM25 retrieval
        print("\nSearching for: 'Who was the California Attorney General?'")
        results = await graphiti.search('Who was the California Attorney General?')

        # Print search results
        print('\nSearch Results:')
        for result in results:
            print(f'UUID: {result.uuid}')
            print(f'Fact: {result.fact}')
            if hasattr(result, 'valid_at') and result.valid_at:
                print(f'Valid from: {result.valid_at}')
            if hasattr(result, 'invalid_at') and result.invalid_at:
                print(f'Valid until: {result.invalid_at}')
            print('---')

        #################################################
        # CENTER NODE SEARCH
        #################################################
        # For more contextually relevant results, you can
        # use a center node to rerank search results based
        # on their graph distance to a specific node
        #################################################

        # Use the top search result's UUID as the center node for reranking
        if results and len(results) > 0:
            # Get the source node UUID from the top result
            center_node_uuid = results[0].source_node_uuid

            print('\nReranking search results based on graph distance:')
            print(f'Using center node UUID: {center_node_uuid}')

            reranked_results = await graphiti.search(
                'Who was the California Attorney General?', center_node_uuid=center_node_uuid
            )

            # Print reranked search results
            print('\nReranked Search Results:')
            for result in reranked_results:
                print(f'UUID: {result.uuid}')
                print(f'Fact: {result.fact}')
                if hasattr(result, 'valid_at') and result.valid_at:
                    print(f'Valid from: {result.valid_at}')
                if hasattr(result, 'invalid_at') and result.invalid_at:
                    print(f'Valid until: {result.invalid_at}')
                print('---')
        else:
            print('No results found in the initial search to use as center node.')

        #################################################
        # NODE SEARCH USING SEARCH RECIPES
        #################################################
        # Graphiti provides predefined search recipes
        # optimized for different search scenarios.
        # Here we use NODE_HYBRID_SEARCH_RRF for retrieving
        # nodes directly instead of edges.
        #################################################

        # Example: Perform a node search using _search method with standard recipes
        print(
            '\nPerforming node search using _search method with standard recipe NODE_HYBRID_SEARCH_RRF:'
        )

        # Use a predefined search configuration recipe and modify its limit
        node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        node_search_config.limit = 5  # Limit to 5 results

        # Execute the node search
        node_search_results = await graphiti._search(
            query='California Governor',
            config=node_search_config,
        )

        # Print node search results
        print('\nNode Search Results:')
        for node in node_search_results.nodes:
            print(f'Node UUID: {node.uuid}')
            print(f'Node Name: {node.name}')
            node_summary = node.summary[:100] + '...' if len(node.summary) > 100 else node.summary
            print(f'Content Summary: {node_summary}')
            print(f'Node Labels: {", ".join(node.labels)}')
            print(f'Created At: {node.created_at}')
            if hasattr(node, 'attributes') and node.attributes:
                print('Attributes:')
                for key, value in node.attributes.items():
                    print(f'  {key}: {value}')
            print('---')

    finally:
        #################################################
        # CLEANUP
        #################################################
        # Always close the connection to Neptune when
        # finished to properly release resources
        #################################################

        # Close the connection
        await graphiti.close()
        print('\nConnection closed')


if __name__ == '__main__':
    asyncio.run(main())
