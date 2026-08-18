import importlib
import re
import sys
from pathlib import Path

tomllib = importlib.import_module('tomllib' if sys.version_info >= (3, 11) else 'tomli')

ROOT = Path(__file__).resolve().parents[1]


def _project_version(path: Path) -> str:
    return str(tomllib.loads(path.read_text())['project']['version'])


def _docker_arg(path: Path, name: str) -> str:
    match = re.search(rf'^ARG {re.escape(name)}=(\S+)$', path.read_text(), re.MULTILINE)
    assert match is not None, f'{name} is missing from {path}'
    return match.group(1)


def test_mcp_image_defaults_match_release_projects():
    core_version = _project_version(ROOT / 'pyproject.toml')
    mcp_version = _project_version(ROOT / 'mcp_server/pyproject.toml')

    for dockerfile in (
        ROOT / 'mcp_server/docker/Dockerfile',
        ROOT / 'mcp_server/docker/Dockerfile.standalone',
    ):
        assert _docker_arg(dockerfile, 'GRAPHITI_CORE_VERSION') == core_version
        assert _docker_arg(dockerfile, 'MCP_SERVER_VERSION') == mcp_version

    compose = (ROOT / 'mcp_server/docker/docker-compose.yml').read_text()
    assert f'GRAPHITI_CORE_VERSION:-{core_version}' in compose
    assert f'MCP_SERVER_VERSION:-{mcp_version}' in compose


def test_neptune_release_images_install_the_required_core_extra():
    server_dockerfile = (ROOT / 'Dockerfile').read_text()
    server_release_workflow = (ROOT / '.github/workflows/release-server-container.yml').read_text()
    standalone_dockerfile = (ROOT / 'mcp_server/docker/Dockerfile.standalone').read_text()
    mcp_server_source = (ROOT / 'mcp_server/src/graphiti_mcp_server.py').read_text()

    assert 'ARG INSTALL_NEPTUNE=false' in server_dockerfile
    assert 'elif [ "$INSTALL_NEPTUNE" = "true" ]; then EXTRA="[neptune]"' in server_dockerfile
    assert 'INSTALL_NEPTUNE=true' in server_release_workflow
    assert (
        'graphiti-core[neo4j,falkordb,neptune]==${GRAPHITI_CORE_VERSION}' in standalone_dockerfile
    )
    assert 'RUN echo "${GRAPHITI_CORE_VERSION}" > /app/mcp/.graphiti-core-version' in (
        standalone_dockerfile
    )
    assert "Path('/app/mcp/.graphiti-core-version')" in mcp_server_source
