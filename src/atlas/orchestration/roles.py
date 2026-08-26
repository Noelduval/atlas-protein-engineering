"""Interfaces for scientific role implementations called by LangGraph."""

from __future__ import annotations

from typing import Protocol

from atlas.adaptive.ledger import ScientificLedger
from atlas.orchestration.state import OrchestrationState


class RoleExecutor(Protocol):
    def execute(
        self,
        role: str,
        state: OrchestrationState,
        ledger: ScientificLedger,
    ) -> dict[str, object]: ...


class LedgerRoleExecutor:
    """Baseline role executor that persists a real orchestration observation.

    Production pipeline integration supplies evidence-producing role handlers.
    This implementation remains useful for graph readiness and makes no
    scientific claim by itself.
    """

    def execute(
        self,
        role: str,
        state: OrchestrationState,
        ledger: ScientificLedger,
    ) -> dict[str, object]:
        updates: dict[str, object] = {}
        if role == "generation":
            updates["failure_memory_count"] = len(ledger.failures())
        if role == "critic":
            updates["route"] = "PROMOTE"
        ledger.record_event(
            "role_observation",
            {
                "run_id": state["run_id"],
                "role": role,
                "round_index": state["round_index"],
                "active_candidate_id": state["active_candidate_id"],
                "claim_boundary": "routing observation; not scientific evidence",
            },
        )
        return updates

