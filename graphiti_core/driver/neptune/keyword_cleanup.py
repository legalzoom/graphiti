"""Prune orphan keyword documents without changing graph data or live projections."""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Any

from opensearchpy import helpers
from opensearchpy.exceptions import NotFoundError

from graphiti_core.driver.operations.graph_ops import SearchIndexPruneResult
from graphiti_core.helpers import validate_group_id

if TYPE_CHECKING:
    from graphiti_core.driver.neptune_driver import NeptuneDriver

logger = logging.getLogger(__name__)


class KeywordIndexCleanupError(RuntimeError):
    """Cleanup did not complete; earlier pages may already have been pruned."""


def _hits(response: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(response, dict):
        raise KeywordIndexCleanupError('Keyword index search returned an invalid response')
    shards = response.get('_shards')
    if (
        response.get('timed_out') is not False
        or not isinstance(shards, dict)
        or shards.get('failed') != 0
        or response.get('terminated_early') is True
    ):
        raise KeywordIndexCleanupError('Keyword index search did not complete successfully')
    envelope = response.get('hits')
    hits = envelope.get('hits') if isinstance(envelope, dict) else None
    if not isinstance(hits, list) or any(not isinstance(hit, dict) for hit in hits):
        raise KeywordIndexCleanupError('Keyword index search returned invalid hits')
    return hits


def _identity(hit: dict[str, Any]) -> tuple[str, str]:
    source = hit.get('_source')
    if not isinstance(source, dict):
        raise KeywordIndexCleanupError('Keyword document has no source identity')
    uuid, group = source.get('uuid'), source.get('group_id')
    if not isinstance(uuid, str) or not uuid or uuid != hit.get('_id'):
        raise KeywordIndexCleanupError('Keyword document identity does not match its document ID')
    if not isinstance(group, str):
        raise KeywordIndexCleanupError('Keyword document has no explicit group identity')
    return uuid, group


def _delete_action(index: str, hit: dict[str, Any]) -> dict[str, Any]:
    sequence, term = hit.get('_seq_no'), hit.get('_primary_term')
    if type(sequence) is not int or sequence < 0 or type(term) is not int or term < 1:
        raise KeywordIndexCleanupError('Keyword document lacks conditional-deletion metadata')
    return {
        '_op_type': 'delete',
        '_index': index,
        '_id': hit['_id'],
        '_if_seq_no': sequence,
        '_if_primary_term': term,
    }


def _delete_page(driver: NeptuneDriver, actions: list[dict[str, Any]]) -> int:
    try:
        successful, errors = helpers.bulk(
            driver.aoss_client,
            actions,
            stats_only=False,
            raise_on_error=False,
            raise_on_exception=False,
        )
    except Exception as exc:
        raise KeywordIndexCleanupError(
            f'Keyword cleanup failed (error_type={type(exc).__name__}); '
            'earlier documents may have been removed'
        ) from None
    absent = 0
    for error in errors:
        operation = error.get('delete', {})
        if operation.get('status') == 404 and operation.get('result') == 'not_found':
            absent += 1
        else:
            # In particular, a version conflict is not permission for an unconditional delete.
            raise KeywordIndexCleanupError(
                'Conditional keyword cleanup failed; earlier documents may have been removed'
            )
    if successful + absent != len(actions):
        raise KeywordIndexCleanupError('Keyword cleanup did not account for every document')
    return successful


async def prune_keyword_indices(
    driver: NeptuneDriver,
    indices: list[dict[str, Any]],
    *,
    group_id: str | None,
    dry_run: bool,
    page_size: int,
) -> list[SearchIndexPruneResult]:
    """Check current graph existence before conditionally deleting each index document.

    None is reserved for the existing global-clear operation. Public repair requires a group.
    This is a bounded-page traversal, not an atomic snapshot or a fence against future writes.
    """
    if group_id is not None:
        validate_group_id(group_id)
    results = []
    try:
        for index in indices:
            results.append(await _prune_index(driver, index, group_id, dry_run, page_size))
    except KeywordIndexCleanupError as exc:
        logger.error('Keyword cleanup stopped: %s', exc)
        raise
    except Exception as exc:
        # This operation also backs clear_graph, whose existing presenter echoes errors.
        raise KeywordIndexCleanupError(
            f'Keyword cleanup failed (error_type={type(exc).__name__}); '
            'earlier documents may have been removed'
        ) from None
    return results


async def _prune_index(
    driver: NeptuneDriver,
    index: dict[str, Any],
    group_id: str | None,
    dry_run: bool,
    page_size: int,
) -> SearchIndexPruneResult:
    name = index['index_name']
    search_after = None
    examined = retained = orphaned = removed = 0
    while True:
        body: dict[str, Any] = {
            'size': page_size,
            '_source': ['uuid', 'group_id'],
            'seq_no_primary_term': True,
            'sort': [{'uuid': 'asc'}],
            # Older indexes analyze group_id as text. This only discovers candidates;
            # exact _source equality below authorizes a group-scoped deletion.
            'query': {'match_phrase': {'group_id': {'query': group_id, 'zero_terms_query': 'all'}}}
            if group_id is not None
            else {'match_all': {}},
        }
        if search_after is not None:
            body['search_after'] = search_after
        try:
            response = await driver._execute_aoss_search(name, body)
        except NotFoundError:
            break
        hits = _hits(response)
        if not hits:
            break
        last_uuid, _ = _identity(hits[-1])
        next_cursor = hits[-1].get('sort')
        if next_cursor != [last_uuid] or (search_after is not None and next_cursor <= search_after):
            raise KeywordIndexCleanupError('Keyword cleanup search cursor did not advance')
        if len(hits) > page_size:
            raise KeywordIndexCleanupError('Keyword cleanup search exceeded its page size')

        by_group: dict[str, list[dict[str, Any]]] = {}
        for hit in hits:
            if hit.get('_index') != name:
                raise KeywordIndexCleanupError('Keyword search returned a different index')
            uuid, stored_group = _identity(hit)
            if group_id is not None and stored_group != group_id:
                continue
            by_group.setdefault(stored_group, []).append(hit)

        for stored_group, candidates in by_group.items():
            uuids = [hit['_id'] for hit in candidates]
            records, _, _ = await driver.execute_query(
                index['graph_match']
                + ' WHERE record.uuid IN $uuids AND record.group_id = $group_id '
                + 'AND coalesce(record.opr_deleted, false) = false '
                + 'RETURN DISTINCT record.uuid AS uuid',
                uuids=uuids,
                group_id=stored_group,
            )
            if not isinstance(records, list) or any(
                not isinstance(record, dict) or record.get('uuid') not in uuids
                for record in records
            ):
                raise KeywordIndexCleanupError('Graph existence read returned invalid identities')
            existing = {record['uuid'] for record in records}
            actions = [
                _delete_action(name, hit) for hit in candidates if hit['_id'] not in existing
            ]
            examined += len(candidates)
            retained += len(candidates) - len(actions)
            orphaned += len(actions)
            if actions and not dry_run:
                removed += await driver._execute_aoss_mutation(
                    partial(_delete_page, driver, actions)
                )
        search_after = next_cursor
        if len(hits) < page_size:
            break
    return SearchIndexPruneResult(name, examined, retained, orphaned, removed)
