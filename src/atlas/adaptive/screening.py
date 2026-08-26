"""Independent-axis Pareto screening and diversity-preserving funnel selection."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from atlas.adaptive.models import (
    CandidateRecord,
    EvidenceAxis,
    EvidenceRecord,
    EvidenceStatus,
    HardViolation,
)


class Direction(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass(frozen=True)
class Objective:
    axis: EvidenceAxis
    direction: Direction


@dataclass(frozen=True)
class Evaluation:
    candidate: CandidateRecord
    evidence: tuple[EvidenceRecord, ...]
    hard_violations: tuple[HardViolation, ...] = ()

    @property
    def hard_rejected(self) -> bool:
        return bool(self.hard_violations)

    def latest_by_axis(self) -> dict[EvidenceAxis, EvidenceRecord]:
        available: dict[EvidenceAxis, EvidenceRecord] = {}
        for record in self.evidence:
            if record.status is EvidenceStatus.AVAILABLE and record.value is not None:
                available[record.axis] = record
        return available

    @property
    def evidence_completeness(self) -> int:
        return len(self.latest_by_axis())


def _conservative_value(record: EvidenceRecord, direction: Direction) -> float:
    value = float(record.value)
    uncertainty = 0.0 if record.uncertainty is None else float(record.uncertainty)
    return value + uncertainty if direction is Direction.MINIMIZE else value - uncertainty


def _dominates(left: Evaluation, right: Evaluation, objectives: tuple[Objective, ...]) -> bool:
    left_evidence, right_evidence = left.latest_by_axis(), right.latest_by_axis()
    if any(
        objective.axis not in left_evidence or objective.axis not in right_evidence
        for objective in objectives
    ):
        return False
    no_worse = True
    strictly_better = False
    for objective in objectives:
        left_value = _conservative_value(left_evidence[objective.axis], objective.direction)
        right_value = _conservative_value(right_evidence[objective.axis], objective.direction)
        if objective.direction is Direction.MINIMIZE:
            no_worse &= left_value <= right_value
            strictly_better |= left_value < right_value
        else:
            no_worse &= left_value >= right_value
            strictly_better |= left_value > right_value
    return no_worse and strictly_better


def pareto_front(
    evaluations: Iterable[Evaluation], objectives: Iterable[Objective]
) -> tuple[Evaluation, ...]:
    objective_tuple = tuple(objectives)
    if not objective_tuple:
        raise ValueError("At least one independent objective is required")
    eligible = tuple(evaluation for evaluation in evaluations if not evaluation.hard_rejected)
    return tuple(
        evaluation
        for evaluation in eligible
        if not any(
            other.candidate.candidate_id != evaluation.candidate.candidate_id
            and _dominates(other, evaluation, objective_tuple)
            for other in eligible
        )
    )


def pareto_layers(
    evaluations: Iterable[Evaluation], objectives: Iterable[Objective]
) -> tuple[tuple[Evaluation, ...], ...]:
    objectives_tuple = tuple(objectives)
    remaining = list(evaluation for evaluation in evaluations if not evaluation.hard_rejected)
    layers: list[tuple[Evaluation, ...]] = []
    while remaining:
        front_ids = {
            evaluation.candidate.candidate_id
            for evaluation in pareto_front(remaining, objectives_tuple)
        }
        if not front_ids:
            break
        layer = tuple(
            sorted(
                (evaluation for evaluation in remaining if evaluation.candidate.candidate_id in front_ids),
                key=lambda evaluation: evaluation.candidate.candidate_id,
            )
        )
        layers.append(layer)
        remaining = [
            evaluation
            for evaluation in remaining
            if evaluation.candidate.candidate_id not in front_ids
        ]
    return tuple(layers)


def _round_robin_diversity(layer: tuple[Evaluation, ...]) -> tuple[Evaluation, ...]:
    buckets: dict[tuple[str, str], deque[Evaluation]] = defaultdict(deque)
    for evaluation in layer:
        candidate = evaluation.candidate
        buckets[(candidate.structural_region, candidate.strategy.value)].append(evaluation)
    ordered: list[Evaluation] = []
    keys = sorted(buckets)
    while any(buckets.values()):
        for key in keys:
            if buckets[key]:
                ordered.append(buckets[key].popleft())
    return tuple(ordered)


def select_diverse_survivors(
    evaluations: Iterable[Evaluation],
    objectives: Iterable[Objective],
    *,
    target: int,
) -> tuple[Evaluation, ...]:
    """Fill the funnel by Pareto layer and region/strategy strata, never a score."""
    if target < 1:
        raise ValueError("target must be positive")
    selected: list[Evaluation] = []
    for layer in pareto_layers(evaluations, objectives):
        for evaluation in _round_robin_diversity(layer):
            selected.append(evaluation)
            if len(selected) == target:
                return tuple(selected)
    return tuple(selected)

