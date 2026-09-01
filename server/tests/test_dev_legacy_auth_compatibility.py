import logging
from datetime import date, timedelta
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException, Request

from graph_service import config
from graph_service.auth import GraphitiAuthorizer, Permission
from graph_service.config import Settings
from graph_service.dto import (
    AddEntityNodeRequest,
    AddMessagesRequest,
    GetMemoryRequest,
    Message,
    Result,
    SearchQuery,
)
from graph_service.protocol import GRAPHITI_RECONCILIATION_OPERATION_RETIRE_EPISODE
from graph_service.routers import ingest
from graph_service.routers.ingest import (
    add_entity_node,
    add_messages,
    get_episode_retirement_status,
)
from graph_service.routers.retrieve import get_episodes_for_reconciliation, get_memory, search
from graph_service.zep_graphiti import ZepGraphiti

TODAY = date(2026, 9, 1)
REMOVE_BY = TODAY + timedelta(days=7)
WRITER_FLEET_EPOCH = 'writer-fleet-epoch-' + ('e' * 32)


def _settings_values(**overrides) -> dict:
    values = {
        'openai_api_key': 'test',
        'ingest_queue_maxsize': 1000,
        'graphiti_deployment_environment': 'dev',
        'opr_dev_legacy_auth_compatibility_enabled': True,
        'opr_dev_legacy_auth_compatibility_remove_by': REMOVE_BY.isoformat(),
    }
    values.update(overrides)
    return values


def _settings(monkeypatch: pytest.MonkeyPatch, **overrides) -> Settings:
    monkeypatch.setattr(config, '_utc_today', lambda: TODAY)
    return Settings.model_validate(_settings_values(**overrides))


def _http_request() -> Request:
    return cast(Request, SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace())))


def test_legacy_compatibility_defaults_off_and_auth_stays_fail_closed():
    settings = Settings.model_validate(
        {
            'openai_api_key': 'test',
            'ingest_queue_maxsize': 1000,
        }
    )

    assert settings.opr_dev_legacy_auth_compatibility_enabled is False
    assert settings.opr_dev_legacy_auth_compatibility_remove_by is None


@pytest.mark.asyncio
async def test_auth_stays_fail_closed_when_legacy_compatibility_is_disabled():
    settings = Settings.model_validate(
        {
            'openai_api_key': 'test',
            'ingest_queue_maxsize': 1000,
        }
    )

    with pytest.raises(HTTPException) as exc_info:
        await GraphitiAuthorizer(settings).require(Permission.READ, None)

    assert exc_info.value.status_code == 403


@pytest.mark.parametrize('environment', ['', 'prd', 'DEV', 'development'])
def test_legacy_compatibility_is_rejected_outside_exact_dev_environment(
    monkeypatch: pytest.MonkeyPatch,
    environment: str,
):
    monkeypatch.setattr(config, '_utc_today', lambda: TODAY)

    with pytest.raises(ValueError, match='GRAPHITI_DEPLOYMENT_ENVIRONMENT=dev'):
        Settings.model_validate(_settings_values(graphiti_deployment_environment=environment))


@pytest.mark.parametrize(
    ('remove_by', 'error'),
    [
        (None, 'REMOVE_BY'),
        (TODAY.isoformat(), 'future date'),
        ((TODAY - timedelta(days=1)).isoformat(), 'future date'),
        ((TODAY + timedelta(days=15)).isoformat(), '14 days'),
    ],
)
def test_legacy_compatibility_requires_a_bounded_removal_date(
    monkeypatch: pytest.MonkeyPatch,
    remove_by: str | None,
    error: str,
):
    monkeypatch.setattr(config, '_utc_today', lambda: TODAY)

    with pytest.raises(ValueError, match=error):
        Settings.model_validate(
            _settings_values(opr_dev_legacy_auth_compatibility_remove_by=remove_by)
        )


def test_legacy_compatibility_cannot_override_required_auth(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, '_utc_today', lambda: TODAY)

    with pytest.raises(ValueError, match='OPR_AUTH_REQUIRED=false'):
        Settings.model_validate(_settings_values(opr_auth_required=True))


@pytest.mark.asyncio
async def test_legacy_compatibility_emits_unmistakable_startup_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    authorizer = GraphitiAuthorizer(_settings(monkeypatch))

    with caplog.at_level(logging.WARNING, logger='graph_service.auth'):
        await authorizer.start()

    assert any(
        'OPR DEV LEGACY AUTH COMPATIBILITY IS ENABLED' in record.getMessage()
        and REMOVE_BY.isoformat() in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_legacy_compatibility_allows_unauthenticated_opr_messages(
    monkeypatch: pytest.MonkeyPatch,
):
    queue = Mock()
    monkeypatch.setattr(
        ingest,
        'async_worker',
        SimpleNamespace(
            accepting=True,
            draining=False,
            depth=0,
            capacity=1000,
            queue=queue,
        ),
    )
    graphiti = cast(ZepGraphiti, SimpleNamespace())
    settings = _settings(monkeypatch)

    result = await add_messages(
        AddMessagesRequest(
            group_id='opr',
            messages=[Message(content='content', role_type='user', role=None)],
        ),
        _http_request(),
        graphiti,
        settings,
        GraphitiAuthorizer(settings),
    )

    assert isinstance(result, Result)
    assert result.success is True
    queue.put_nowait.assert_called_once()


@pytest.mark.asyncio
async def test_legacy_compatibility_allows_unauthenticated_opr_entity_write(
    monkeypatch: pytest.MonkeyPatch,
):
    save_entity_node = AsyncMock(return_value={'uuid': 'entity'})
    graphiti = cast(ZepGraphiti, SimpleNamespace(save_entity_node=save_entity_node))
    settings = _settings(monkeypatch)

    await add_entity_node(
        AddEntityNodeRequest(
            uuid='entity',
            group_id='opr',
            name='name',
            summary='summary',
        ),
        graphiti,
        settings,
        GraphitiAuthorizer(settings),
    )

    save_entity_node.assert_awaited_once()


@pytest.mark.asyncio
async def test_legacy_compatibility_allows_unauthenticated_opr_search_and_memory(
    monkeypatch: pytest.MonkeyPatch,
):
    graphiti_search = AsyncMock(return_value=[])
    graphiti = cast(ZepGraphiti, SimpleNamespace(search=graphiti_search))
    settings = _settings(monkeypatch)
    authorizer = GraphitiAuthorizer(settings)

    await search(
        SearchQuery(query='query', group_ids=['opr']),
        graphiti,
        settings,
        authorizer,
    )
    await get_memory(
        GetMemoryRequest(
            group_id='opr',
            center_node_uuid=None,
            messages=[Message(content='query', role_type='user', role=None)],
        ),
        graphiti,
        settings,
        authorizer,
    )

    assert graphiti_search.await_count == 2


@pytest.mark.asyncio
async def test_legacy_compatibility_never_bypasses_admin(
    monkeypatch: pytest.MonkeyPatch,
):
    settings = _settings(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await GraphitiAuthorizer(settings).require(Permission.ADMIN, None)

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_legacy_compatibility_enforces_a_configured_read_credential(
    monkeypatch: pytest.MonkeyPatch,
):
    settings = _settings(monkeypatch, opr_read_token='configured-read-token')
    authorizer = GraphitiAuthorizer(settings)

    with pytest.raises(HTTPException) as exc_info:
        await authorizer.require(Permission.READ, None)
    assert exc_info.value.status_code == 403

    await authorizer.require(Permission.READ, 'Bearer configured-read-token')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('permission', 'setting_name', 'token'),
    [
        (Permission.RECONCILE, 'opr_reconciliation_token', 'configured-reconcile-token'),
        (Permission.RETIRE, 'opr_retirement_token', 'configured-retirement-token'),
    ],
)
async def test_legacy_compatibility_enforces_configured_privileged_credentials(
    monkeypatch: pytest.MonkeyPatch,
    permission: Permission,
    setting_name: str,
    token: str,
):
    settings = _settings(monkeypatch, **{setting_name: token})
    authorizer = GraphitiAuthorizer(settings)

    with pytest.raises(HTTPException) as exc_info:
        await authorizer.require(permission, None, legacy_token=None)
    assert exc_info.value.status_code == 403

    await authorizer.require(permission, None, legacy_token=token)


@pytest.mark.asyncio
async def test_reconciliation_v5_still_requires_the_writer_fleet_fence(
    monkeypatch: pytest.MonkeyPatch,
):
    retrieve_episodes = AsyncMock(return_value=[])
    graphiti = cast(
        ZepGraphiti,
        SimpleNamespace(retrieve_episodes_for_reconciliation=retrieve_episodes),
    )
    settings = _settings(monkeypatch, opr_writer_fleet_epoch=WRITER_FLEET_EPOCH)
    authorizer = GraphitiAuthorizer(settings)

    with pytest.raises(HTTPException) as exc_info:
        await get_episodes_for_reconciliation(
            'opr',
            20,
            graphiti,
            settings,
            authorizer,
            x_opr_writer_fleet_epoch='wrong-epoch',
        )

    assert exc_info.value.status_code == 403
    retrieve_episodes.assert_not_awaited()

    await get_episodes_for_reconciliation(
        'opr',
        20,
        graphiti,
        settings,
        authorizer,
        x_opr_writer_fleet_epoch=WRITER_FLEET_EPOCH,
    )
    retrieve_episodes.assert_awaited_once()


@pytest.mark.asyncio
async def test_retirement_v5_still_requires_operation_fence_and_durable_receipt(
    monkeypatch: pytest.MonkeyPatch,
):
    retirement_outcome = AsyncMock(return_value=None)
    graphiti = cast(
        ZepGraphiti,
        SimpleNamespace(episode_retirement_outcome=retirement_outcome),
    )
    settings = _settings(monkeypatch, opr_writer_fleet_epoch=WRITER_FLEET_EPOCH)
    authorizer = GraphitiAuthorizer(settings)

    with pytest.raises(HTTPException) as operation_error:
        await get_episode_retirement_status(
            'episode',
            'request-id',
            'opr',
            graphiti,
            settings,
            authorizer,
            x_opr_writer_fleet_epoch=WRITER_FLEET_EPOCH,
        )
    assert operation_error.value.status_code == 403
    retirement_outcome.assert_not_awaited()

    with pytest.raises(HTTPException) as receipt_error:
        await get_episode_retirement_status(
            'episode',
            'request-id',
            'opr',
            graphiti,
            settings,
            authorizer,
            x_opr_writer_fleet_epoch=WRITER_FLEET_EPOCH,
            x_opr_reconciliation_operation=(GRAPHITI_RECONCILIATION_OPERATION_RETIRE_EPISODE),
        )
    assert receipt_error.value.status_code == 412
    retirement_outcome.assert_awaited_once()
