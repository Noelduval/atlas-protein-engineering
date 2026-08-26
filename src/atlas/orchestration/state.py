"""Minimal routing state; scientific payloads remain in the Atlas ledger."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict


class OrchestrationState(TypedDict):
    run_id: str
    ledger_path: str
    events_path: str
    round_index: int
    stage: str
    route: Literal["REJECT", "REVISE", "PROMOTE"]
    active_candidate_id: str
    generated_count: int
    evaluated_count: int
    failure_memory_count: int
    repair_generation: int
    role_step: int
    terminal_reason: str


def initial_state(
    *,
    run_id: str,
    ledger_path: str | Path,
    events_path: str | Path,
    round_index: int = 1,
    active_candidate_id: str = "",
) -> OrchestrationState:
    return OrchestrationState(
        run_id=run_id,
        ledger_path=str(ledger_path),
        events_path=str(events_path),
        round_index=round_index,
        stage="initialized",
        route="PROMOTE",
        active_candidate_id=active_candidate_id,
        generated_count=0,
        evaluated_count=0,
        failure_memory_count=0,
        repair_generation=0,
        role_step=0,
        terminal_reason="",
    )

