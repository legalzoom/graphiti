from datetime import datetime, timezone

import pytest

from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.nodes import EpisodeType, EpisodicNode


@pytest.mark.asyncio
async def test_kuzu_episode_save_remains_compatible_with_explicit_schema():
    driver = KuzuDriver(':memory:')
    episode = EpisodicNode(
        name='kuzu compatibility',
        group_id='test',
        source=EpisodeType.text,
        source_description='test',
        content='content',
        created_at=datetime.now(timezone.utc),
        valid_at=datetime.now(timezone.utc),
    )

    try:
        await episode.save(driver)
        stored = await EpisodicNode.get_by_uuid(driver, episode.uuid)
        assert stored.uuid == episode.uuid
    finally:
        await driver.close()
