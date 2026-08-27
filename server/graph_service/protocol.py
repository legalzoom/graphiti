"""Versioned contracts for privileged OPR reconciliation operations."""

import hashlib
import hmac

GRAPHITI_RECONCILIATION_PROTOCOL = 'opr.graphiti.reconciliation/v5'
GRAPHITI_RECONCILIATION_OPERATION_RETIRE_EPISODE = 'retire_episode'
GRAPHITI_RECONCILIATION_GROUP_ID = 'opr'


def reconciliation_token_matches(expected: str, supplied: str | None) -> bool:
    """Compare the deployment-managed M2M credential without timing leaks."""
    return bool(
        expected
        and supplied
        and hmac.compare_digest(expected.encode('utf-8'), supplied.encode('utf-8'))
    )


def writer_fleet_epoch_sha256(epoch: str) -> str:
    """Return the non-secret attestation bound to privileged responses."""
    return hashlib.sha256(epoch.encode('utf-8')).hexdigest()


def bearer_token_matches(expected: str, authorization: str | None) -> bool:
    """Validate one configured bearer credential without accepting raw tokens."""
    if not expected or not authorization:
        return False
    scheme, separator, supplied = authorization.partition(' ')
    if (
        separator != ' '
        or scheme.casefold() != 'bearer'
        or not supplied
        or supplied != supplied.strip()
        or any(character.isspace() for character in supplied)
    ):
        return False
    return hmac.compare_digest(expected.encode('utf-8'), supplied.encode('utf-8'))
