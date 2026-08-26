from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from atlas.adaptive.ledger import ScientificLedger
from atlas.orchestration.graph import AtlasOrchestrator
from atlas.orchestration.roles import RoleExecutor
from atlas.orchestration.state import OrchestrationState, initial_state


ALLOWED_STATE_FIELDS = {
    "run_id",
    "ledger_path",
    "events_path",
    "round_index",
    "stage",
    "route",
    "active_candidate_id",
    "generated_count",
    "evaluated_count",
    "failure_memory_count",
    "repair_generation",
    "role_step",
    "terminal_reason",
}


class ScriptedRoles(RoleExecutor):
    def __init__(self, critic_routes=("PROMOTE",), fail_once_at: str | None = None):
        self.critic_routes = list(critic_routes)
        self.fail_once_at = fail_once_at
        self.failed = False
        self.calls: list[str] = []

    def execute(self, role, state, ledger):
        self.calls.append(role)
        if self.fail_once_at == role and not self.failed:
            self.failed = True
            raise RuntimeError("simulated interruption")
        updates = {}
        if role == "generation":
            updates["failure_memory_count"] = len(ledger.failures())
        if role == "critic":
            updates["route"] = self.critic_routes.pop(0)
        ledger.record_event(
            "test_role_evidence",
            {"role": role, "run_id": state["run_id"], "step": state["role_step"]},
        )
        return updates


def _orchestrator(tmp_path: Path, roles: RoleExecutor) -> AtlasOrchestrator:
    ledger_path = tmp_path / "scientific.sqlite"
    events_path = tmp_path / "events.jsonl"
    ScientificLedger.create(ledger_path, events_path).close()
    return AtlasOrchestrator.create(
        checkpoint_path=tmp_path / "graph-checkpoints.sqlite",
        role_executor=roles,
    )


def test_graph_state_contains_routing_identifiers_not_scientific_payloads() -> None:
    assert set(OrchestrationState.__annotations__) == ALLOWED_STATE_FIELDS
    assert not {"candidates", "evidence", "failures", "structures"}.intersection(
        OrchestrationState.__annotations__
    )


def test_promote_route_executes_deeper_science_and_portfolio(tmp_path: Path) -> None:
    roles = ScriptedRoles(("PROMOTE",))
    orchestrator = _orchestrator(tmp_path, roles)
    state = initial_state(
        run_id="run-promote",
        ledger_path=tmp_path / "scientific.sqlite",
        events_path=tmp_path / "events.jsonl",
    )
    final = orchestrator.run(state)
    orchestrator.close()

    assert roles.calls == [
        "research_evidence",
        "hypothesis",
        "generation",
        "screening",
        "structure",
        "critic",
        "deeper_evaluation",
        "simulation",
        "activity_evidence",
        "adversarial_critic",
        "portfolio_selection",
        "coordinator",
    ]
    assert final["terminal_reason"] == "portfolio_complete"


def test_revise_route_reenters_evaluation_and_is_bounded(tmp_path: Path) -> None:
    roles = ScriptedRoles(("REVISE", "PROMOTE"))
    orchestrator = _orchestrator(tmp_path, roles)
    final = orchestrator.run(
        initial_state(
            run_id="run-repair",
            ledger_path=tmp_path / "scientific.sqlite",
            events_path=tmp_path / "events.jsonl",
        )
    )
    orchestrator.close()

    counts = Counter(roles.calls)
    assert counts["repair"] == 1
    assert counts["screening"] == 2
    assert counts["structure"] == 2
    assert counts["critic"] == 2
    assert final["repair_generation"] == 1
    assert final["route"] == "PROMOTE"


def test_repeated_revision_cannot_exceed_two_generations(tmp_path: Path) -> None:
    roles = ScriptedRoles(("REVISE", "REVISE", "REVISE"))
    orchestrator = _orchestrator(tmp_path, roles)
    final = orchestrator.run(
        initial_state(
            run_id="run-bounded",
            ledger_path=tmp_path / "scientific.sqlite",
            events_path=tmp_path / "events.jsonl",
        )
    )
    orchestrator.close()

    assert Counter(roles.calls)["repair"] == 2
    assert final["route"] == "REJECT"
    assert final["terminal_reason"] == "candidate_rejected_after_bounded_review"


def test_sqlite_checkpoint_resume_does_not_repeat_completed_roles(tmp_path: Path) -> None:
    roles = ScriptedRoles(("PROMOTE",), fail_once_at="structure")
    orchestrator = _orchestrator(tmp_path, roles)
    state = initial_state(
        run_id="run-resume",
        ledger_path=tmp_path / "scientific.sqlite",
        events_path=tmp_path / "events.jsonl",
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        orchestrator.run(state)
    final = orchestrator.resume("run-resume")
    orchestrator.close()

    counts = Counter(roles.calls)
    assert counts["research_evidence"] == 1
    assert counts["generation"] == 1
    assert counts["structure"] == 2
    assert final["terminal_reason"] == "portfolio_complete"

