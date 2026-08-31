"""Checkpointed LangGraph routes over ledger-owned Atlas scientific state."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any, Callable, Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from atlas.adaptive.ledger import ScientificLedger
from atlas.orchestration.roles import LedgerRoleExecutor, RoleExecutor
from atlas.orchestration.state import OrchestrationState


_LINEAR_PREFIX = (
    "research_evidence",
    "hypothesis",
    "generation",
    "screening",
    "structure",
    "critic",
)
_PROMOTION_PATH = (
    "deeper_evaluation",
    "activity_evidence",
    "adversarial_critic",
    "portfolio_selection",
    "coordinator",
)


class AtlasOrchestrator:
    def __init__(
        self,
        *,
        checkpoint_connection: sqlite3.Connection,
        checkpointer: SqliteSaver,
        role_executor: RoleExecutor,
    ) -> None:
        self._connection = checkpoint_connection
        self.checkpointer = checkpointer
        self.role_executor = role_executor
        self.graph = self._build_graph()

    @classmethod
    def create(
        cls,
        *,
        checkpoint_path: str | Path,
        role_executor: RoleExecutor | None = None,
    ) -> AtlasOrchestrator:
        destination = Path(checkpoint_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(destination, check_same_thread=False)
        checkpointer = SqliteSaver(connection)
        checkpointer.setup()
        return cls(
            checkpoint_connection=connection,
            checkpointer=checkpointer,
            role_executor=role_executor or LedgerRoleExecutor(),
        )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> AtlasOrchestrator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _node(self, role: str) -> Callable[[OrchestrationState], dict[str, Any]]:
        def execute(state: OrchestrationState) -> dict[str, Any]:
            ledger = ScientificLedger.open(
                state["ledger_path"],
                state["events_path"],
                synchronize_mirror=False,
            )
            try:
                updates = dict(self.role_executor.execute(role, state, ledger))
                updates.update(
                    {
                        "stage": role,
                        "role_step": state["role_step"] + 1,
                    }
                )
                if role == "repair":
                    updates["repair_generation"] = state["repair_generation"] + 1
                if (
                    role == "critic"
                    and updates.get("route", state["route"]) == "REVISE"
                    and state["repair_generation"] >= 2
                ):
                    updates["route"] = "REJECT"
                if role == "coordinator":
                    route = str(updates.get("route", state["route"]))
                    updates["terminal_reason"] = (
                        "candidate_rejected_after_bounded_review"
                        if route == "REJECT"
                        else "portfolio_complete"
                    )
                ledger.record_event(
                    "orchestration_transition",
                    {
                        "run_id": state["run_id"],
                        "role": role,
                        "next_step": updates["role_step"],
                        "route": updates.get("route", state["route"]),
                    },
                )
                return updates
            finally:
                ledger.close()

        return execute

    @staticmethod
    def _critic_route(
        state: OrchestrationState,
    ) -> Literal["REJECT", "REVISE", "PROMOTE"]:
        return state["route"]

    def _build_graph(self):
        builder = StateGraph(OrchestrationState)
        for role in _LINEAR_PREFIX + ("repair",) + _PROMOTION_PATH:
            builder.add_node(role, self._node(role))
        builder.add_edge(START, _LINEAR_PREFIX[0])
        for left, right in zip(_LINEAR_PREFIX, _LINEAR_PREFIX[1:]):
            if left != "critic":
                builder.add_edge(left, right)
        builder.add_conditional_edges(
            "critic",
            self._critic_route,
            {
                "REJECT": "coordinator",
                "REVISE": "repair",
                "PROMOTE": "deeper_evaluation",
            },
        )
        builder.add_edge("repair", "screening")
        for left, right in zip(_PROMOTION_PATH, _PROMOTION_PATH[1:]):
            builder.add_edge(left, right)
        builder.add_edge("coordinator", END)
        return builder.compile(checkpointer=self.checkpointer)

    @staticmethod
    def _config(run_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": run_id},
            "recursion_limit": 80,
        }

    def run(self, state: OrchestrationState) -> OrchestrationState:
        return self.graph.invoke(state, config=self._config(state["run_id"]))

    def resume(self, run_id: str) -> OrchestrationState:
        return self.graph.invoke(None, config=self._config(run_id))
