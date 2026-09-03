# Graphiti Quickstart Example

This example demonstrates the basic functionality of Graphiti, including:

1. Connecting to a Neo4j or FalkorDB database
2. Initializing Graphiti indices and constraints
3. Adding episodes to the graph
4. Searching the graph with semantic and keyword matching
5. Exploring graph-based search with reranking using the top search result's source node UUID
6. Performing node search using predefined search recipes

## Prerequisites

- Python 3.9+  
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)  
- **For Neo4j**:
  - Neo4j Desktop installed and running  
  - A local DBMS created and started in Neo4j Desktop  
- **For FalkorDB**:
  - FalkorDB server running (see [FalkorDB documentation](https://docs.falkordb.com) for setup)
- **For Amazon Neptune**:
  - Amazon Neptune running (see [Amazon Neptune documentation](https://aws.amazon.com/neptune/developer-resources/) for setup)
  - An OpenSearch Serverless text collection (an existing collection may be type `SEARCH`)
  - A separate OpenSearch Serverless collection of type `VECTORSEARCH` for vector indexes


## Setup Instructions

1. Install the required dependencies:

```bash
pip install graphiti-core
```

2. Set up environment variables:

```bash
# Required for LLM and embedding
export OPENAI_API_KEY=your_openai_api_key

# Optional Neo4j connection parameters (defaults shown)
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password

# Optional FalkorDB connection parameters (defaults shown)
export FALKORDB_URI=falkor://localhost:6379

# Optional Amazon Neptune connection parameters
export NEPTUNE_HOST=your_neptune_host
export NEPTUNE_PORT=your_port_or_8182
export AOSS_HOST=your_text_search_collection_host
export AOSS_PORT=your_port_or_443
export VECTOR_AOSS_HOST=your_vectorsearch_collection_host
export VECTOR_AOSS_PORT=your_port_or_443

# Safe defaults. Change these only in the staged rollout described below.
export NEPTUNE_VECTOR_PROJECTION_ENABLED=false
export NEPTUNE_VECTOR_SEARCH_ENABLED=false

# To use a different database, modify the driver constructor in the script
```

TIP: For Amazon Neptune host string please use the following formats
* For Neptune Database: `neptune-db://<cluster endpoint>`
* For Neptune Analytics: `neptune-graph://<graph identifier>`

3. Run the example:

```bash
python quickstart_neo4j.py

# For FalkorDB
python quickstart_falkordb.py

# For Amazon Neptune
python quickstart_neptune.py
```

### Neptune vector rollout

The four BM25/text indexes can remain in an existing `SEARCH` collection addressed by
`AOSS_HOST`. OpenSearch Serverless accepts `knn_vector` mappings only in a `VECTORSEARCH`
collection, so `VECTOR_AOSS_HOST` must address that separate collection whenever vector
projections or reads are enabled.

Roll out vector search in this order. Run the service commands below from the repository root so
they use the server project's environment:

1. Provision the `VECTORSEARCH` collection, configure `VECTOR_AOSS_HOST`, and verify its indexes
   with `uv --directory server run python -m graph_service.verify_vector_indices`.
2. Enable projections on all writers while keeping reads off
   (`NEPTUNE_VECTOR_PROJECTION_ENABLED=true`, `NEPTUNE_VECTOR_SEARCH_ENABLED=false`).
3. Quiesce ingestion and deletion for every group; the read flag applies to the whole driver.
4. With both still quiesced, reset only the two vector indexes and run the all-groups initial
   backfill plus its automatic final reconciliation pass:

   ```bash
   uv --directory server run python -m graph_service.backfill_embeddings \
       --all-groups \
       --reset-vector-indices \
       --acknowledge-ingestion-and-deletion-quiesced
   ```

   A later one-group exact repair must use `--group-id <group_id>
   --reset-group-vector-documents --acknowledge-ingestion-and-deletion-quiesced`; it does not, by
   itself, make driver-wide reads safe.
5. Enable reads only after the all-groups reset and both passes succeed, then resume ingestion and
   deletion.

REST and MCP processes with projections enabled automatically retry durable failed saves and
deletes at startup and every 30 seconds by default. Configure the interval with
`NEPTUNE_VECTOR_RECONCILE_INTERVAL_SECONDS`. For a one-off retry that does not require quiescing,
run `uv --directory server run python -m graph_service.backfill_embeddings --all-groups
--pending-only` (or combine `--group-id <group_id>` with `--pending-only`). A nonzero exit means at
least one marker remains.

To roll back, turn reads off but keep projections and projection deletions on. If you revert to
code that does not maintain them, repeat the all-groups reset and quiesced two-pass backfill before
enabling reads again. Do not delete the text indexes for a vector rollback. The quickstart creates
missing indexes idempotently and does not delete existing graph data or indexes.

## What This Example Demonstrates

- **Graph Initialization**: Setting up the Graphiti indices and constraints in Neo4j, Amazon Neptune, or FalkorDB
- **Adding Episodes**: Adding text content that will be analyzed and converted into knowledge graph nodes and edges
- **Edge Search Functionality**: Performing hybrid searches that combine semantic similarity and BM25 retrieval to find relationships (edges)
- **Graph-Aware Search**: Using the source node UUID from the top search result to rerank additional search results based on graph distance
- **Node Search Using Recipes**: Using predefined search configurations like NODE_HYBRID_SEARCH_RRF to directly search for nodes rather than edges
- **Result Processing**: Understanding the structure of search results including facts, nodes, and temporal metadata

## Next Steps

After running this example, you can:

1. Modify the episode content to add your own information
2. Try different search queries to explore the knowledge extraction
3. Experiment with different center nodes for graph-distance-based reranking
4. Try other predefined search recipes from `graphiti_core.search.search_config_recipes`
5. Explore the more advanced examples in the other directories

## Troubleshooting

### "Graph not found: default_db" Error

If you encounter the error `Neo.ClientError.Database.DatabaseNotFound: Graph not found: default_db`, this occurs when the driver is trying to connect to a database that doesn't exist.

**Solution:**
The Neo4j driver defaults to using `neo4j` as the database name. If you need to use a different database, modify the driver constructor in the script:

```python
# In quickstart_neo4j.py, change:
driver = Neo4jDriver(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)

# To specify a different database:
driver = Neo4jDriver(uri=neo4j_uri, user=neo4j_user, password=neo4j_password, database="your_db_name")
```

## Understanding the Output

### Edge Search Results

The edge search results include EntityEdge objects with:

- UUID: Unique identifier for the edge
- Fact: The extracted fact from the episode
- Valid at/invalid at: Time period during which the fact was true (if available)
- Source/target node UUIDs: Connections between entities in the knowledge graph

### Node Search Results

The node search results include EntityNode objects with:

- UUID: Unique identifier for the node
- Name: The name of the entity
- Content Summary: A summary of the node's content
- Node Labels: The types of the node (e.g., Person, Organization)
- Created At: When the node was created
- Attributes: Additional properties associated with the node
