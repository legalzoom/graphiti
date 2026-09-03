"""Queue service for managing episode processing."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from services.graphiti_scope import graphiti_for_group

logger = logging.getLogger(__name__)

DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 30.0
MAX_WORKER_CANCEL_WAIT_SECONDS = 1.0


class QueueService:
    """Service for managing sequential episode processing queues by group_id."""

    def __init__(self):
        """Initialize the queue service."""
        # Dictionary to store queues for each group_id
        self._episode_queues: dict[str, asyncio.Queue] = {}
        # Keep the actual worker task so check-and-reserve is atomic on the event loop.
        self._worker_tasks: dict[str, asyncio.Task[None]] = {}
        # Store the graphiti client after initialization
        self._graphiti_client: Any = None
        self._accepting_tasks = True

    async def add_episode_task(
        self, group_id: str, process_func: Callable[[], Awaitable[None]]
    ) -> int:
        """Add an episode processing task to the queue.

        Args:
            group_id: The group ID for the episode
            process_func: The async function to process the episode

        Returns:
            The position in the queue
        """
        if not self._accepting_tasks:
            raise RuntimeError('Queue service is shutting down and is not accepting new work')

        # Initialize queue for this group_id if it doesn't exist
        if group_id not in self._episode_queues:
            self._episode_queues[group_id] = asyncio.Queue()

        # Add the episode processing function to the queue
        await self._episode_queues[group_id].put(process_func)

        # Store the task before yielding again. Setting a boolean inside the new
        # coroutine leaves a race where simultaneous enqueues can spawn workers
        # that process one group's supposedly sequential queue concurrently.
        worker = self._worker_tasks.get(group_id)
        if worker is None or worker.done():
            self._worker_tasks[group_id] = asyncio.create_task(
                self._process_episode_queue(group_id)
            )

        return self._episode_queues[group_id].qsize()

    async def _process_episode_queue(self, group_id: str) -> None:
        """Process episodes for a specific group_id sequentially.

        This function runs as a long-lived task that processes episodes
        from the queue one at a time.
        """
        logger.info(f'Starting episode queue worker for group_id: {group_id}')
        queue = self._episode_queues[group_id]

        try:
            while True:
                # Get the next episode processing function from the queue
                # This will wait if the queue is empty
                process_func = await queue.get()

                try:
                    # Process the episode
                    await process_func()
                except Exception as e:
                    logger.error(
                        f'Error processing queued episode for group_id {group_id}: {str(e)}'
                    )
                finally:
                    # Mark the task as done regardless of success/failure
                    queue.task_done()
        except asyncio.CancelledError:
            logger.info(f'Episode queue worker for group_id {group_id} was cancelled')
        except Exception as e:
            logger.error(f'Unexpected error in queue worker for group_id {group_id}: {str(e)}')
        finally:
            current_worker = asyncio.current_task()
            if self._worker_tasks.get(group_id) is current_worker:
                self._worker_tasks.pop(group_id, None)
            logger.info(f'Stopped episode queue worker for group_id: {group_id}')

    def get_queue_size(self, group_id: str) -> int:
        """Get the current queue size for a group_id."""
        if group_id not in self._episode_queues:
            return 0
        return self._episode_queues[group_id].qsize()

    def is_worker_running(self, group_id: str) -> bool:
        """Check if a worker is running for a group_id."""
        worker = self._worker_tasks.get(group_id)
        return worker is not None and not worker.done()

    @staticmethod
    def _consume_detached_worker_result(worker: asyncio.Task[None]) -> None:
        """Retrieve a late worker result after bounded shutdown has returned."""
        if worker.cancelled():
            return
        error = worker.exception()
        if error is not None:
            logger.error(
                'Detached episode queue worker failed after shutdown (error_type=%s)',
                type(error).__name__,
            )

    async def shutdown(
        self,
        timeout_seconds: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
    ) -> bool:
        """Stop admission, drain queued work when possible, then stop every worker.

        The queue workers share the Graphiti client owned by ``GraphitiService``. Cooperative
        workers stop before that service closes its graph driver. A worker that ignores repeated
        cancellation is detached after a bounded grace period. The return value is false when any
        detached worker may still be using that shared client, so the caller can leave the client
        open rather than closing its driver underneath in-flight consistency work.
        """
        if timeout_seconds < 0:
            raise ValueError('timeout_seconds must be non-negative')

        self._accepting_tasks = False
        queues = list(self._episode_queues.values())
        if queues:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(queue.join() for queue in queues)),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    'Timed out after %.1fs draining episode queues; cancelling workers',
                    timeout_seconds,
                )

        workers = list(self._worker_tasks.values())
        for worker in workers:
            if not worker.done():
                worker.cancel()
        all_workers_stopped = True
        if workers:
            cancel_wait_seconds = min(timeout_seconds, MAX_WORKER_CANCEL_WAIT_SECONDS)
            done, pending = await asyncio.wait(workers, timeout=cancel_wait_seconds)
            if done:
                await asyncio.gather(*done, return_exceptions=True)
            if pending:
                all_workers_stopped = False
                logger.critical(
                    '%d episode queue worker(s) did not stop within %.1fs after cancellation; '
                    'detaching them and leaving their shared graph client open',
                    len(pending),
                    cancel_wait_seconds,
                )
                for worker in pending:
                    worker.add_done_callback(self._consume_detached_worker_result)
                    # A second request stops jobs that only swallowed the first cancellation.
                    # Do not await again: a hostile job may suppress cancellation repeatedly.
                    worker.cancel()

        # A timeout can leave functions that never started in a queue. Balance their unfinished
        # task counts before releasing the queues; the functions themselves have not created
        # coroutine objects and require no additional cleanup.
        for queue in queues:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                else:
                    queue.task_done()

        self._episode_queues.clear()
        self._worker_tasks.clear()
        self._graphiti_client = None
        return all_workers_stopped

    async def initialize(self, graphiti_client: Any) -> None:
        """Initialize the queue service with a graphiti client.

        Args:
            graphiti_client: The graphiti client instance to use for processing episodes
        """
        self._graphiti_client = graphiti_client
        logger.info('Queue service initialized with graphiti client')

    async def add_episode(
        self,
        group_id: str,
        name: str,
        content: str,
        source_description: str,
        episode_type: Any,
        entity_types: Any,
        uuid: str | None,
        reference_time: datetime | None = None,
        edge_types: Any = None,
        edge_type_map: Any = None,
        excluded_entity_types: list[str] | None = None,
        previous_episode_uuids: list[str] | None = None,
        custom_extraction_instructions: str | None = None,
        update_communities: bool = False,
        saga: str | None = None,
        saga_previous_episode_uuid: str | None = None,
    ) -> int:
        """Add an episode for processing.

        Args:
            group_id: The group ID for the episode
            name: Name of the episode
            content: Episode content
            source_description: Description of the episode source
            episode_type: Type of the episode
            entity_types: Entity types for extraction
            uuid: Episode UUID
            reference_time: Event occurrence time for the episode. Defaults to
                the current UTC time when not provided (bi-temporal model).
            edge_types: Optional mapping of edge (fact) type name to Pydantic model
            edge_type_map: Optional mapping of (source, target) entity type pairs to
                allowed edge type names
            excluded_entity_types: Optional list of entity type names to exclude
                from extraction
            previous_episode_uuids: Optional explicit list of prior episode UUIDs to
                use as context (overrides automatic retrieval)
            custom_extraction_instructions: Optional extra natural-language
                instructions for the extraction LLM
            update_communities: Whether to incrementally update communities after
                ingestion
            saga: Optional saga name/id to attach this episode to
            saga_previous_episode_uuid: Optional UUID of the prior episode in the saga

        Returns:
            The position in the queue
        """
        if self._graphiti_client is None:
            raise RuntimeError('Queue service not initialized. Call initialize() first.')

        async def process_episode():
            """Process the episode using the graphiti client."""
            try:
                logger.info(f'Processing episode {uuid} for group {group_id}')

                # FalkorDB maps each group to its own physical graph. Use a
                # call-scoped Graphiti copy so workers for different groups
                # never race by rebinding the shared client's driver.
                scoped_client = await graphiti_for_group(self._graphiti_client, group_id)
                core_group_id = (
                    None
                    if group_id
                    == getattr(getattr(scoped_client, 'driver', None), 'default_group_id', None)
                    else group_id
                )
                await scoped_client.add_episode(
                    name=name,
                    episode_body=content,
                    source_description=source_description,
                    source=episode_type,
                    # Falkor's logical default group ('_') lives in the configured base
                    # database. Passing '_' explicitly makes core clone
                    # again; None selects the logical default without rebinding.
                    group_id=core_group_id,
                    reference_time=reference_time or datetime.now(timezone.utc),
                    entity_types=entity_types,
                    edge_types=edge_types,
                    edge_type_map=edge_type_map,
                    excluded_entity_types=excluded_entity_types,
                    previous_episode_uuids=previous_episode_uuids,
                    custom_extraction_instructions=custom_extraction_instructions,
                    update_communities=update_communities,
                    saga=saga,
                    saga_previous_episode_uuid=saga_previous_episode_uuid,
                    uuid=uuid,
                )

                logger.info(f'Successfully processed episode {uuid} for group {group_id}')

            except Exception as e:
                logger.error(f'Failed to process episode {uuid} for group {group_id}: {str(e)}')
                raise

        # Use the existing add_episode_task method to queue the processing
        return await self.add_episode_task(group_id, process_episode)
