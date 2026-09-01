# graph-service

Graph service is a fast api server implementing the [graphiti](https://github.com/getzep/graphiti) package.

## Container Releases

The FastAPI server container is automatically built and published to Docker Hub when a new `graphiti-core` version is released to PyPI.

**Image:** `zepai/graphiti`

**Available tags:**
- `latest` - Latest stable release
- `0.22.1` - Specific version (matches graphiti-core version)

**Platforms:** linux/amd64, linux/arm64

The automated release workflow:
1. Triggers when `graphiti-core` PyPI release completes
2. Waits for PyPI package availability
3. Builds multi-platform Docker image
4. Tags with version number and `latest`
5. Pushes to Docker Hub

Only stable releases are built automatically (pre-release versions are skipped).

## Running Instructions

1. Ensure you have Docker and Docker Compose installed on your system.

2. Add `zepai/graphiti:latest` to your service setup

3. Make sure to pass the following environment variables to the service

   ```
   OPENAI_API_KEY=your_openai_api_key
   NEO4J_USER=your_neo4j_user
   NEO4J_PASSWORD=your_neo4j_password
   NEO4J_PORT=your_neo4j_port
   ```

4. This service depends on having access to a neo4j instance, you may wish to add a neo4j image to your service setup as well. Or you may wish to use neo4j cloud or a desktop version if running this locally.

   An example of docker compose setup may look like this:

   ```yml
      version: '3.8'

      services:
      graph:
         image: zepai/graphiti:latest
         ports:
            - "8000:8000"
         
         environment:
            - OPENAI_API_KEY=${OPENAI_API_KEY}
            - NEO4J_URI=bolt://neo4j:${NEO4J_PORT}
            - NEO4J_USER=${NEO4J_USER}
            - NEO4J_PASSWORD=${NEO4J_PASSWORD}
      neo4j:
         image: neo4j:5.22.0
         
         ports:
            - "7474:7474"  # HTTP
            - "${NEO4J_PORT}:${NEO4J_PORT}"  # Bolt
         volumes:
            - neo4j_data:/data
         environment:
            - NEO4J_AUTH=${NEO4J_USER}/${NEO4J_PASSWORD}

      volumes:
      neo4j_data:
   ```

5. Once you start the service, it will be available at `http://localhost:8000` (or the port you have specified in the docker compose file).

6. You may access the swagger docs at `http://localhost:8000/docs`. You may also access redocs at `http://localhost:8000/redoc`.

7. You may also access the neo4j browser at `http://localhost:7474` (the port depends on the neo4j instance you are using).

## Protected OPR graph authentication

The OPR-owned graph group (`group_id=opr`) supports two explicit authentication modes. There is no
credential fallback between them.

### Temporary DEV legacy compatibility bridge

The REST service has a narrowly scoped incident bridge for rolling DEV forward before its OPR
callers have migrated credentials. It is disabled by default and must never be enabled in shared
configuration consumed by the MCP server. Enable it only on the DEV REST container with all three
values:

```text
GRAPHITI_DEPLOYMENT_ENVIRONMENT=dev
OPR_DEV_LEGACY_AUTH_COMPATIBILITY_ENABLED=true
OPR_DEV_LEGACY_AUTH_COMPATIBILITY_REMOVE_BY=<future YYYY-MM-DD, at most 14 days away>
```

Startup fails unless the environment is exactly `dev`, `OPR_AUTH_REQUIRED=false`, static auth mode
is selected, and the removal date is future-dated but no more than 14 days away. A restart after the
date refuses the stale bridge, and startup emits a security warning containing the deadline.

While active, only OPR read, write, reconciliation, and retirement caller-identity checks whose
static credential is still empty are bypassed. Configuring a credential immediately restores its
normal enforcement even while the bridge is active. Administrative authorization and every
writer-fleet epoch, group, operation, receipt, and domain-level fence remain enforced. Remove the
bridge after DEV callers send their intended static or LZ JWT credentials; it is not a third
authentication mode.

### Static compatibility mode

`OPR_AUTH_MODE=static` is the default and preserves the legacy deployment contract. When
`OPR_AUTH_REQUIRED=true`, configure six distinct, HTTP-safe values of at least 32 bytes:

- `OPR_READ_TOKEN`
- `OPR_WRITE_TOKEN`
- `OPR_RECONCILIATION_TOKEN`
- `OPR_RETIREMENT_TOKEN`
- `GRAPHITI_ADMIN_TOKEN`
- `OPR_WRITER_FLEET_EPOCH`

Read, write, and admin credentials use `Authorization: Bearer`. Reconciliation and retirement use
their existing `X-OPR-*` capability headers. The writer fleet epoch is a rollout fence rather than
a caller identity.

### LegalZoom JWT mode

`OPR_AUTH_MODE=lz_jwt` validates short-lived LegalZoom Authorization Service access tokens. It
requires `OPR_AUTH_REQUIRED=true` and these resource-server settings:

```text
OPR_JWT_ISSUER=<exact environment issuer, including its trailing slash>
OPR_JWT_JWKS_URL=<environment issuer JWKS HTTPS URL>
OPR_JWT_AUDIENCE=<exact access-token audience>
OPR_JWT_ALLOWED_CLIENT_IDS=<comma-separated registered OPR client IDs>
OPR_JWT_READ_SCOPE=<registered read scope>
OPR_JWT_WRITE_SCOPE=<registered write scope>
OPR_JWT_RECONCILIATION_SCOPE=<registered reconciliation scope>
OPR_JWT_RETIREMENT_SCOPE=<registered retirement scope>
OPR_JWT_ADMIN_SCOPE=<registered admin scope>
OPR_WRITER_FLEET_EPOCH=<deployment fence of at least 32 bytes>
```

Obtain issuer, audience, registered client IDs, and owned scope names from Identity for each
environment. LegalZoom M2M tokens use `gty=client-credentials`, identify the registered client in
`azp`, and carry authorized scopes in the `scope` claim. Graphiti requires those claims, an
allowlisted `azp`, the exact issuer and audience, an unexpired token, and the specific route scope.
`roles` (including `lz_admin`) never grant Graphiti access. Only RS256 is accepted.

The five legacy identity values are not required in `lz_jwt` mode. They may coexist temporarily in
the environment to support old pods during a controlled migration, but JWT-mode pods ignore them.
Reconciliation and retirement use the normal `Authorization: Bearer <access-token>` header in this
mode; their writer-fleet-epoch and operation-binding headers remain required.

JWKS keys are fetched before application startup, cached, and refreshed on expiry or an unknown key
ID. Cache behavior is bounded by `OPR_JWKS_CACHE_TTL_SECONDS`,
`OPR_JWKS_MAX_STALE_SECONDS`, and `OPR_JWKS_REFRESH_MIN_INTERVAL_SECONDS`; JWT clock tolerance uses
`OPR_JWT_CLOCK_SKEW_SECONDS` (maximum 120 seconds). A pod cannot start without an initial valid
JWKS. After the bounded stale period, a refresh outage fails authorization closed.

Do not mix `static` and `lz_jwt` pods behind one Service while a caller sends only one credential
type. Use a separate JWT canary endpoint or a controlled cutover so requests do not alternate
between incompatible authentication envelopes.
