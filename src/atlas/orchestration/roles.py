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


class EvidenceBackedRoleExecutor:
    """Route over already persisted production evidence without copying payloads.

    Each role verifies the ledger-owned scientific state relevant to that role
    and appends an observation.  The repair role follows a real persisted
    parent/child trajectory; the critic follows an explicit persisted decision.
    """

    _ROLE_AXES = {
        "research_evidence": (),
        "screening": ("stability", "liability"),
        "structure": ("structure_quality",),
        "deeper_evaluation": ("catalytic_geometry", "substrate_interface"),
        "simulation": ("dynamics",),
        "activity_evidence": ("activity_oriented",),
    }

    def execute(
        self,
        role: str,
        state: OrchestrationState,
        ledger: ScientificLedger,
    ) -> dict[str, object]:
        candidate_id = state["active_candidate_id"]
        candidate = ledger.get_candidate(candidate_id) if candidate_id else None
        updates: dict[str, object] = {}
        if role == "generation":
            updates["generated_count"] = ledger.candidate_count()
            updates["failure_memory_count"] = len(ledger.failures())
        elif role == "repair":
            trajectory = ledger.repair_trajectory(candidate_id)
            complete = [item for item in trajectory if item.get("disposition")]
            if not complete:
                updates["route"] = "REJECT"
            else:
                updates["active_candidate_id"] = str(complete[0]["child_id"])
        elif role == "critic":
            decisions = ledger.decisions_for(candidate_id)
            updates["route"] = decisions[-1]["route"] if decisions else state["route"]
        elif role == "coordinator":
            updates["evaluated_count"] = ledger.evidence_candidate_count()

        active_id = str(updates.get("active_candidate_id", candidate_id))
        active_candidate = ledger.get_candidate(active_id) if active_id else candidate
        evidence = () if active_candidate is None else ledger.evidence_for(active_id)
        required_axes = self._ROLE_AXES.get(role, ())
        observed_axes = sorted({record.axis.value for record in evidence})
        ledger.record_event(
            "scientific_role_observation",
            {
                "run_id": state["run_id"],
                "role": role,
                "candidate_id": active_id,
                "candidate_exists": active_candidate is not None,
                "required_axes": list(required_axes),
                "observed_axes": observed_axes,
                "evidence_record_count": len(evidence),
                "decision_count": len(ledger.decisions_for(active_id)) if active_id else 0,
                "claim_boundary": (
                    "LangGraph routes ledger identifiers; scientific payloads remain in the "
                    "typed append-only ledger."
                ),
            },
        )
        return updates
