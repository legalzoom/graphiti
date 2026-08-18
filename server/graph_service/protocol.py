"""Versioned contracts for privileged OPR reconciliation operations."""

import hmac

GRAPHITI_RECONCILIATION_PROTOCOL = 'opr.graphiti.reconciliation/v3'
GRAPHITI_RECONCILIATION_OPERATION_RETIRE_EPISODE = 'retire_episode'
GRAPHITI_RECONCILIATION_GROUP_ID = 'opr'


def reconciliation_token_matches(expected: str, supplied: str | None) -> bool:
    """Compare the deployment-managed M2M credential without timing leaks."""
    return bool(expected and supplied and hmac.compare_digest(expected, supplied))
