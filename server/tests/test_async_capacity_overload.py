import json
from unittest.mock import MagicMock

import pytest
from graphiti_core.async_limiter import AsyncCapacityOverloadedError

from graph_service.main import async_capacity_overloaded


@pytest.mark.asyncio
async def test_async_capacity_overload_maps_to_retryable_service_unavailable():
    response = await async_capacity_overloaded(
        MagicMock(), AsyncCapacityOverloadedError(capacity=2, max_waiters=32)
    )

    assert response.status_code == 503
    assert response.headers['Retry-After'] == '1'
    assert json.loads(bytes(response.body)) == {
        'detail': 'capacity is exhausted and all 32 waiting slots are occupied'
    }
