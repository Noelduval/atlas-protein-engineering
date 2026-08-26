"""Deterministic adaptive rounds for broad singles and evidence-guided doubles."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from typing import Iterable

from atlas.adaptive.ledger import DuplicateCandidateError, ScientificLedger
from atlas.adaptive.models import CandidateRecord, DesignStrategy, FailureObservation
from atlas.design.design_space import ResidueClass, ResidueDesignRecord
from atlas.design.strategies import (
    strategy_for_residue,
    strategy_narrative,
    substitution_order,
)


@dataclass(frozen=True)
class SearchBudget:
    candidate_budget: int = 5_000
    round1_target: int = 1_200
    minimum_doubles: int = 750

    def __post_init__(self) -> None:
        if self.candidate_budget < 2:
            raise ValueError("candidate_budget must allow adaptive rounds")
        if not 1 <= self.round1_target < self.candidate_budget:
            raise ValueError("round1_target must be within the candidate budget")
        if not 1 <= self.minimum_doubles < self.candidate_budget:
            raise ValueError("minimum_doubles must be within the candidate budget")
        if self.round1_target + self.minimum_doubles > self.candidate_budget:
            raise ValueError("round1_target leaves no room for the minimum double budget")


@dataclass(frozen=True)
class AdaptiveSearchResult:
    candidates: tuple[CandidateRecord, ...]
    strategy_counts: dict[str, int]
    region_counts: dict[str, int]
    round_counts: dict[int, int]
    failure_memory_deprioritized: int


class AdaptiveGenerator:
    def __init__(self, design_space: Iterable[ResidueDesignRecord], *, seed: int) -> None:
        records = tuple(sorted(design_space, key=lambda record: record.position))
        if [record.position for record in records] != list(range(1, len(records) + 1)):
            raise ValueError("Design space must contain one contiguous position record")
        self.design_space = records
        self.reference_sequence = "".join(record.wildtype for record in records)
        self.seed = int(seed)

    @staticmethod
    def _memory_scopes(
        failures: Iterable[FailureObservation], prefix: str
    ) -> frozenset[str]:
        return frozenset(
            failure.scope.removeprefix(prefix)
            for failure in failures
            if failure.scope.startswith(prefix)
            and failure.evidence_count >= 3
            and failure.confidence >= 0.75
        )

    def _seed_tiebreaker(self, label: str) -> str:
        return hashlib.sha256(f"{self.seed}:{label}".encode()).hexdigest()

    def _single_candidate(
        self, record: ResidueDesignRecord, mutant: str, *, round_index: int
    ) -> CandidateRecord:
        strategy = strategy_for_residue(record)
        hypothesis, upside, risk = strategy_narrative(strategy, record, mutant)
        return CandidateRecord.create(
            reference_sequence=self.reference_sequence,
            mutations=[f"{record.wildtype}{record.position}{mutant}"],
            parents=(),
            strategy=strategy,
            structural_region=record.structural_region,
            round_index=round_index,
            hypothesis=hypothesis,
            intended_upside=upside,
            expected_risk=risk,
        )

    def _single_pool(
        self, penalized_mutations: frozenset[str]
    ) -> list[tuple[ResidueDesignRecord, str, int, bool]]:
        pool: list[tuple[ResidueDesignRecord, str, int, bool]] = []
        for record in self.design_space:
            if record.residue_class not in {
                ResidueClass.DESIGNABLE,
                ResidueClass.CONTEXT_SENSITIVE,
            }:
                continue
            for substitution_rank, mutant in enumerate(substitution_order(record)):
                label = f"{record.wildtype}{record.position}{mutant}"
                pool.append(
                    (record, mutant, substitution_rank, label in penalized_mutations)
                )
        return pool

    def generate_seed_search(
        self,
        budget: SearchBudget,
        *,
        position_priority: Iterable[int] = (),
        failure_memory: Iterable[FailureObservation] = (),
    ) -> AdaptiveSearchResult:
        """Generate adaptive rounds 1–3, reserving repairs for evaluated near-misses."""
        failures = tuple(failure_memory)
        penalized_mutations = self._memory_scopes(failures, "mutation:")
        penalized_combinations = self._memory_scopes(failures, "combination:")
        pool = self._single_pool(penalized_mutations)
        broad = sorted(
            pool,
            key=lambda item: (
                item[3],
                item[2],
                self._seed_tiebreaker(f"{item[0].position}:{item[1]}"),
            ),
        )
        round1_specs = broad[: budget.round1_target]
        round1_keys = {(record.position, mutant) for record, mutant, _, _ in round1_specs}
        priority = tuple(dict.fromkeys(int(position) for position in position_priority))
        priority_rank = {position: index for index, position in enumerate(priority)}
        remaining = [
            item for item in pool if (item[0].position, item[1]) not in round1_keys
        ]
        remaining.sort(
            key=lambda item: (
                item[3],
                0 if item[0].position in priority_rank else 1,
                priority_rank.get(item[0].position, item[2]),
                item[2],
                self._seed_tiebreaker(f"expand:{item[0].position}:{item[1]}"),
            )
        )
        maximum_singles = budget.candidate_budget - budget.minimum_doubles
        round2_capacity = max(0, maximum_singles - len(round1_specs))
        round2_specs = remaining[:round2_capacity]

        round1 = tuple(
            self._single_candidate(record, mutant, round_index=1)
            for record, mutant, _, _ in round1_specs
        )
        round2 = tuple(
            self._single_candidate(record, mutant, round_index=2)
            for record, mutant, _, _ in round2_specs
        )
        singles = round1 + round2
        if len(singles) < 2:
            raise ValueError("Design space cannot support evidence-guided combinations")
        round3_target = budget.candidate_budget - len(singles)
        round3 = self._generate_doubles(
            singles,
            target=round3_target,
            preferred_candidate_ids=(),
            penalized_combinations=penalized_combinations,
        )
        candidates = round1 + round2 + round3
        if len(candidates) != budget.candidate_budget:
            raise RuntimeError(
                f"Legal design space produced {len(candidates)} candidates; "
                f"required {budget.candidate_budget}"
            )
        penalized_count = sum(
            1
            for candidate in candidates
            if any(mutation.label in penalized_mutations for mutation in candidate.mutations)
            or candidate.mutation_set in penalized_combinations
        )
        return AdaptiveSearchResult(
            candidates=candidates,
            strategy_counts=dict(Counter(c.strategy.value for c in candidates)),
            region_counts=dict(Counter(c.structural_region for c in candidates)),
            round_counts=dict(Counter(c.round_index for c in candidates)),
            failure_memory_deprioritized=penalized_count,
        )

    def _generate_doubles(
        self,
        singles: tuple[CandidateRecord, ...],
        *,
        target: int,
        preferred_candidate_ids: Iterable[str],
        penalized_combinations: frozenset[str],
    ) -> tuple[CandidateRecord, ...]:
        preferred_rank = {
            candidate_id: index
            for index, candidate_id in enumerate(dict.fromkeys(preferred_candidate_ids))
        }
        parents = sorted(
            singles,
            key=lambda candidate: (
                0 if candidate.candidate_id in preferred_rank else 1,
                preferred_rank.get(candidate.candidate_id, 0),
                candidate.mutations[0].position,
                self._seed_tiebreaker(candidate.candidate_id),
            ),
        )
        generated: list[CandidateRecord] = []
        seen: set[str] = set()
        deferred: list[CandidateRecord] = []
        count = len(parents)
        for offset in range(1, count):
            for left_index in range(count):
                right_index = (left_index + offset) % count
                if left_index >= right_index:
                    continue
                left, right = parents[left_index], parents[right_index]
                if left.mutations[0].position == right.mutations[0].position:
                    continue
                mutations = tuple(sorted(left.mutations + right.mutations))
                mutation_set = "/".join(mutation.label for mutation in mutations)
                regions = {left.structural_region, right.structural_region}
                region = left.structural_region if len(regions) == 1 else "cross_region"
                candidate = CandidateRecord.create(
                    reference_sequence=self.reference_sequence,
                    mutations=mutations,
                    parents=(left.candidate_id, right.candidate_id),
                    strategy=DesignStrategy.EVIDENCE_GUIDED_COMBINATION,
                    structural_region=region,
                    round_index=3,
                    hypothesis=(
                        f"Combine {left.mutation_set} and {right.mutation_set} as an "
                        "explicit epistasis hypothesis from complementary single-mutant designs."
                    ),
                    intended_upside=(
                        "Retain independent supporting features while testing whether their "
                        "combination broadens stability/function tradeoffs."
                    ),
                    expected_risk=(
                        "Combination-specific epistasis may negate either single-mutant benefit."
                    ),
                )
                if candidate.candidate_id in seen:
                    continue
                seen.add(candidate.candidate_id)
                if mutation_set in penalized_combinations:
                    deferred.append(candidate)
                else:
                    generated.append(candidate)
                if len(generated) == target:
                    return tuple(generated)
            if len(generated) >= target:
                break
        for candidate in deferred:
            generated.append(candidate)
            if len(generated) == target:
                return tuple(generated)
        raise RuntimeError(f"Could generate only {len(generated)}/{target} legal doubles")

    @staticmethod
    def persist_new(
        candidates: Iterable[CandidateRecord], ledger: ScientificLedger
    ) -> int:
        persisted = 0
        for candidate in candidates:
            try:
                ledger.add_candidate(candidate)
            except DuplicateCandidateError:
                continue
            persisted += 1
        return persisted

