import logging
from collections.abc import Callable
from datetime import date, timedelta
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException, Request

from graph_service import auth as auth_module
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
READ_TOKEN = 'configured-read-token-' + ('r' * 32)
WRITE_TOKEN = 'configured-write-token-' + ('w' * 32)
RECONCILIATION_TOKEN = 'configured-reconcile-token-' + ('c' * 32)
RETIREMENT_TOKEN = 'configured-retirement-token-' + ('t' * 32)


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
    monkeypatch.setattr(config, 'utc_today', lambda: TODAY)
    return Settings.model_validate(_settings_values(**overrides))


def _today() -> date:
    return TODAY


def _authorizer(
    settings: Settings,
    *,
    current_date: Callable[[], date] = _today,
) -> GraphitiAuthorizer:
    return GraphitiAuthorizer(settings, current_date=current_date)


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


def test_disabled_legacy_compatibility_loads_blank_removal_date_from_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv('OPENAI_API_KEY', 'test')
    monkeypatch.setenv('INGEST_QUEUE_MAXSIZE', '1000')
    monkeypatch.setenv('OPR_DEV_LEGACY_AUTH_COMPATIBILITY_ENABLED', 'false')
    monkeypatch.setenv('OPR_DEV_LEGACY_AUTH_COMPATIBILITY_REMOVE_BY', '')

    settings = Settings()  # type: ignore[call-arg]

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
        await _authorizer(settings).require(Permission.READ, None)

    assert exc_info.value.status_code == 403


@pytest.mark.parametrize('environment', ['', 'prd', 'DEV', 'development'])
def test_legacy_compatibility_is_rejected_outside_exact_dev_environment(
    monkeypatch: pytest.MonkeyPatch,
    environment: str,
):
    monkeypatch.setattr(config, 'utc_today', lambda: TODAY)

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
    monkeypatch.setattr(config, 'utc_today', lambda: TODAY)

    with pytest.raises(ValueError, match=error):
        Settings.model_validate(
            _settings_values(opr_dev_legacy_auth_compatibility_remove_by=remove_by)
        )


def test_legacy_compatibility_accepts_the_fourteen_day_removal_boundary(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(config, 'utc_today', lambda: TODAY)
    remove_by = TODAY + timedelta(days=14)

    settings = Settings.model_validate(
        _settings_values(opr_dev_legacy_auth_compatibility_remove_by=remove_by.isoformat())
    )

    assert settings.opr_dev_legacy_auth_compatibility_remove_by == remove_by


def test_legacy_compatibility_cannot_override_required_auth(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, 'utc_today', lambda: TODAY)

    with pytest.raises(ValueError, match='OPR_AUTH_REQUIRED=false'):
        Settings.model_validate(_settings_values(opr_auth_required=True))


@pytest.mark.asyncio
async def test_legacy_compatibility_emits_unmistakable_startup_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    authorizer = _authorizer(_settings(monkeypatch))

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
        _authorizer(settings),
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
        _authorizer(settings),
    )

    save_entity_node.assert_awaited_once()


@pytest.mark.asyncio
async def test_legacy_compatibility_allows_unauthenticated_opr_search_and_memory(
    monkeypatch: pytest.MonkeyPatch,
):
    graphiti_search = AsyncMock(return_value=[])
    graphiti = cast(ZepGraphiti, SimpleNamespace(search=graphiti_search))
    settings = _settings(monkeypatch)
    authorizer = _authorizer(settings)

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
        await _authorizer(settings).require(Permission.ADMIN, None)

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('permission', 'setting_name', 'token', 'authorization', 'legacy_token'),
    [
        (
            Permission.READ,
            'opr_read_token',
            READ_TOKEN,
            f'Bearer {READ_TOKEN}',
            None,
        ),
        (
            Permission.WRITE,
            'opr_write_token',
            WRITE_TOKEN,
            f'Bearer {WRITE_TOKEN}',
            None,
        ),
        (
            Permission.RECONCILE,
            'opr_reconciliation_token',
            RECONCILIATION_TOKEN,
            None,
            RECONCILIATION_TOKEN,
        ),
        (
            Permission.RETIRE,
            'opr_retirement_token',
            RETIREMENT_TOKEN,
            None,
            RETIREMENT_TOKEN,
        ),
    ],
)
async def test_legacy_compatibility_enforces_each_configured_credential(
    monkeypatch: pytest.MonkeyPatch,
    permission: Permission,
    setting_name: str,
    token: str,
    authorization: str | None,
    legacy_token: str | None,
):
    settings = _settings(monkeypatch, **{setting_name: token})
    authorizer = _authorizer(settings)

    with pytest.raises(HTTPException) as exc_info:
        await authorizer.require(permission, None, legacy_token=None)
    assert exc_info.value.status_code == 403

    await authorizer.require(permission, authorization, legacy_token=legacy_token)


@pytest.mark.parametrize(
    'setting_name',
    [
        'opr_read_token',
        'opr_write_token',
        'opr_reconciliation_token',
        'opr_retirement_token',
        'opr_writer_fleet_epoch',
        'graphiti_admin_token',
    ],
)
def test_legacy_compatibility_rejects_each_short_non_empty_privileged_value(
    monkeypatch: pytest.MonkeyPatch,
    setting_name: str,
):
    sentinel = 'short-active-secret'

    with pytest.raises(
        ValueError,
        match='configured privileged values must be at least 32 bytes',
    ) as exc_info:
        _settings(monkeypatch, **{setting_name: sentinel})

    assert sentinel not in str(exc_info.value)


@pytest.mark.asyncio
async def test_legacy_compatibility_warns_once_per_bypassed_permission(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    monkeypatch.setattr(auth_module, '_DEV_LEGACY_COMPATIBILITY_WARNED_PERMISSIONS', set())
    settings = _settings(monkeypatch)
    authorizer = _authorizer(settings)

    with caplog.at_level(logging.WARNING, logger='graph_service.auth'):
        for _ in range(3):
            await authorizer.require(Permission.READ, None)
        for _ in range(2):
            await authorizer.require(Permission.WRITE, None)

    bypass_warnings = [
        record.getMessage()
        for record in caplog.records
        if 'legacy auth compatibility bypass used' in record.getMessage()
    ]
    assert len(bypass_warnings) == 2
    assert any('permission=read' in message for message in bypass_warnings)
    assert any('permission=write' in message for message in bypass_warnings)


@pytest.mark.asyncio
async def test_legacy_compatibility_expires_for_live_read_and_write_requests(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    monkeypatch.setattr(auth_module, '_DEV_LEGACY_COMPATIBILITY_WARNED_PERMISSIONS', set())
    monkeypatch.setattr(auth_module, '_DEV_LEGACY_COMPATIBILITY_EXPIRY_WARNED', False)
    current_date = [TODAY]
    settings = _settings(monkeypatch)
    authorizer = _authorizer(settings, current_date=lambda: current_date[0])

    await authorizer.require(Permission.READ, None)
    await authorizer.require(Permission.WRITE, None)

    with caplog.at_level(logging.WARNING, logger='graph_service.auth'):
        for expired_date in (REMOVE_BY, REMOVE_BY + timedelta(days=1)):
            current_date[0] = expired_date
            for permission in (Permission.READ, Permission.WRITE):
                with pytest.raises(HTTPException) as exc_info:
                    await authorizer.require(permission, None)
                assert exc_info.value.status_code == 403

    expiry_warnings = [
        record.getMessage()
        for record in caplog.records
        if 'legacy auth compatibility expired' in record.getMessage()
    ]
    assert len(expiry_warnings) == 1


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
    authorizer = _authorizer(settings)

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
    authorizer = _authorizer(settings)

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
