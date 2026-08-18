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
