"""Diversity-aware Pareto portfolio selection without a universal score."""

from __future__ import annotations

from collections.abc import Mapping

from atlas.adaptive.models import CandidateRecord, EvidenceAxis, EvidenceRecord
from atlas.adaptive.screening import (
    Direction,
    Evaluation,
    Objective,
    select_diverse_survivors,
)


PORTFOLIO_OBJECTIVES = (
    Objective(EvidenceAxis.STABILITY, Direction.MINIMIZE),
    Objective(EvidenceAxis.SUBSTRATE_INTERFACE, Direction.MAXIMIZE),
    Objective(EvidenceAxis.CATALYTIC_GEOMETRY, Direction.MINIMIZE),
    Objective(EvidenceAxis.LIABILITY, Direction.MINIMIZE),
)


def select_experimental_portfolio(
    candidates: tuple[CandidateRecord, ...],
    evidence_by_candidate: Mapping[str, tuple[EvidenceRecord, ...]],
    *,
    target: int = 5,
) -> tuple[CandidateRecord, ...]:
    """Select complementary nondominated experiments, never a weighted score."""
    if target < 1 or target > 5:
        raise ValueError("Experimental portfolio target must be between one and five")
    if not candidates:
        return ()
    evaluations = tuple(
        Evaluation(candidate, evidence_by_candidate.get(candidate.candidate_id, ()))
        for candidate in candidates
    )
    ordered = select_diverse_survivors(
        evaluations,
        PORTFOLIO_OBJECTIVES,
        target=min(len(evaluations), target),
    )
    selected: list[CandidateRecord] = []
    used_positions: set[int] = set()
    used_regions: set[str] = set()
    for evaluation in ordered:
        candidate = evaluation.candidate
        positions = {mutation.position for mutation in candidate.mutations}
        if (
            not selected
            or positions - used_positions
            or candidate.structural_region not in used_regions
        ):
            selected.append(candidate)
            used_positions.update(positions)
            used_regions.add(candidate.structural_region)
    for evaluation in ordered:
        if evaluation.candidate not in selected and len(selected) < target:
            selected.append(evaluation.candidate)
    return tuple(selected[:target])
