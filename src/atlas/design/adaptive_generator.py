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
    SubstitutionProposal,
    strategy_for_residue,
    strategy_narrative,
    substitution_policy,
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
        self,
        record: ResidueDesignRecord,
        proposal: SubstitutionProposal,
        *,
        round_index: int,
    ) -> CandidateRecord:
        strategy = strategy_for_residue(record)
        hypothesis, upside, risk = strategy_narrative(strategy, record, proposal)
        return CandidateRecord.create(
            reference_sequence=self.reference_sequence,
            mutations=[f"{record.wildtype}{record.position}{proposal.mutant}"],
            parents=(),
            strategy=strategy,
            structural_region=record.structural_region,
            round_index=round_index,
            hypothesis=hypothesis,
            intended_upside=upside,
            expected_risk=risk,
            intended_physical_change=proposal.intended_physical_change,
            feature_to_preserve=proposal.feature_to_preserve,
            principal_biochemical_risk=proposal.principal_biochemical_risk,
            metal_liability=proposal.metal_liability,
            requires_candidate_geometry=proposal.requires_candidate_geometry,
            substitution_classes=(proposal.substitution_class,),
        )

    def _single_pool(
        self,
        penalized_mutations: frozenset[str],
        penalized_substitutions: frozenset[str] = frozenset(),
    ) -> list[tuple[ResidueDesignRecord, SubstitutionProposal, int, bool]]:
        pool: list[tuple[ResidueDesignRecord, SubstitutionProposal, int, bool]] = []
        for record in self.design_space:
            if record.residue_class not in {
                ResidueClass.DESIGNABLE,
                ResidueClass.CONTEXT_SENSITIVE,
            }:
                continue
            for substitution_rank, proposal in enumerate(substitution_policy(record)):
                label = f"{record.wildtype}{record.position}{proposal.mutant}"
                pool.append(
                    (
                        record,
                        proposal,
                        substitution_rank,
                        label in penalized_mutations
                        or f"{record.structural_region}:{proposal.mutant}"
                        in penalized_substitutions,
                    )
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
        round1 = self.generate_round1(budget, failure_memory=failure_memory)
        round2 = self.generate_round2(
            budget,
            round1=round1,
            position_priority=position_priority,
            failure_memory=failure_memory,
        )
        round3 = self.generate_round3(
            budget,
            singles=round1 + round2,
            preferred_candidate_ids=(),
            failure_memory=failure_memory,
        )
        candidates = round1 + round2 + round3
        failures = tuple(failure_memory)
        penalized_mutations = self._memory_scopes(failures, "mutation:")
        penalized_substitutions = self._memory_scopes(failures, "substitution:")
        penalized_combinations = self._memory_scopes(failures, "combination:")
        penalized_count = sum(
            1
            for candidate in candidates
            if any(mutation.label in penalized_mutations for mutation in candidate.mutations)
            or any(
                f"{candidate.structural_region}:{mutation.mutant}"
                in penalized_substitutions
                for mutation in candidate.mutations
            )
            or candidate.mutation_set in penalized_combinations
        )
        return AdaptiveSearchResult(
            candidates=candidates,
            strategy_counts=dict(Counter(c.strategy.value for c in candidates)),
            region_counts=dict(Counter(c.structural_region for c in candidates)),
            round_counts=dict(Counter(c.round_index for c in candidates)),
            failure_memory_deprioritized=penalized_count,
        )

    def generate_round1(
        self,
        budget: SearchBudget,
        *,
        failure_memory: Iterable[FailureObservation] = (),
    ) -> tuple[CandidateRecord, ...]:
        """Allocate broad singles before any downstream evidence is available."""
        penalized_mutations = self._memory_scopes(failure_memory, "mutation:")
        penalized_substitutions = self._memory_scopes(
            failure_memory, "substitution:"
        )
        pool = self._single_pool(penalized_mutations, penalized_substitutions)
        broad = sorted(
            pool,
            key=lambda item: (
                item[3],
                item[2],
                self._seed_tiebreaker(f"{item[0].position}:{item[1].mutant}"),
            ),
        )
        round1_specs = broad[: budget.round1_target]
        if len(round1_specs) != budget.round1_target:
            raise RuntimeError(
                f"Legal design space produced {len(round1_specs)}/{budget.round1_target} "
                "Round-1 singles"
            )
        return tuple(
            self._single_candidate(record, proposal, round_index=1)
            for record, proposal, _, _ in round1_specs
        )

    def generate_round2(
        self,
        budget: SearchBudget,
        *,
        round1: tuple[CandidateRecord, ...],
        position_priority: Iterable[int] = (),
        failure_memory: Iterable[FailureObservation] = (),
    ) -> tuple[CandidateRecord, ...]:
        """Allocate targeted singles after Round-1 evidence and failure memory exist."""
        penalized_mutations = self._memory_scopes(failure_memory, "mutation:")
        penalized_substitutions = self._memory_scopes(
            failure_memory, "substitution:"
        )
        pool = self._single_pool(penalized_mutations, penalized_substitutions)
        round1_keys = {
            (candidate.mutations[0].position, candidate.mutations[0].mutant)
            for candidate in round1
        }
        priority = tuple(dict.fromkeys(int(position) for position in position_priority))
        priority_rank = {position: index for index, position in enumerate(priority)}
        remaining = [
            item
            for item in pool
            if (item[0].position, item[1].mutant) not in round1_keys
        ]
        remaining.sort(
            key=lambda item: (
                item[3],
                0 if item[0].position in priority_rank else 1,
                priority_rank.get(item[0].position, item[2]),
                item[2],
                self._seed_tiebreaker(
                    f"expand:{item[0].position}:{item[1].mutant}"
                ),
            )
        )
        maximum_singles = budget.candidate_budget - budget.minimum_doubles
        round2_capacity = max(0, maximum_singles - len(round1))
        round2_specs = remaining[:round2_capacity]
        return tuple(
            self._single_candidate(record, proposal, round_index=2)
            for record, proposal, _, _ in round2_specs
        )

    def generate_round3(
        self,
        budget: SearchBudget,
        *,
        singles: tuple[CandidateRecord, ...],
        preferred_candidate_ids: Iterable[str] = (),
        failure_memory: Iterable[FailureObservation] = (),
    ) -> tuple[CandidateRecord, ...]:
        """Allocate evidence-guided doubles from evaluated single-mutant parents."""
        if len(singles) < 2:
            raise ValueError("Design space cannot support evidence-guided combinations")
        round3_target = budget.candidate_budget - len(singles)
        if round3_target < budget.minimum_doubles:
            raise RuntimeError("Single-mutant allocation consumed the minimum double budget")
        penalized_combinations = self._memory_scopes(
            failure_memory, "combination:"
        )
        return self._generate_doubles(
            singles,
            target=round3_target,
            preferred_candidate_ids=preferred_candidate_ids,
            penalized_combinations=penalized_combinations,
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
        records_by_position = {record.position: record for record in self.design_space}
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
                if mutation_set == "Y91F/D126A":
                    continue
                regions = {left.structural_region, right.structural_region}
                region = left.structural_region if len(regions) == 1 else "cross_region"
                left_record = records_by_position[left.mutations[0].position]
                right_record = records_by_position[right.mutations[0].position]
                direct_contact = (
                    right_record.position in left_record.packing_neighbor_positions
                    or left_record.position in right_record.packing_neighbor_positions
                )
                shared_substrate_network = (
                    left_record.min_substrate_distance_a <= 5.0
                    and right_record.min_substrate_distance_a <= 5.0
                )
                shared_preorganization_network = (
                    left_record.min_zinc_distance_a <= 10.0
                    and right_record.min_zinc_distance_a <= 10.0
                )
                function_regions = {"substrate_interface", "second_shell"}
                stability_regions = {"distal_stability", "scaffold_surface"}
                if direct_contact or shared_substrate_network or shared_preorganization_network:
                    double_category = "LOCAL_COUPLED_DOUBLE"
                    coupling = (
                        "direct heavy-atom packing contact"
                        if direct_contact
                        else "shared deposited substrate/catalytic preorganization network"
                    )
                elif (
                    left.structural_region in function_regions
                    and right.structural_region in stability_regions
                ) or (
                    right.structural_region in function_regions
                    and left.structural_region in stability_regions
                ):
                    double_category = "FUNCTION_STABILITY_RESCUE_DOUBLE"
                    coupling = (
                        "function-oriented mutation paired with a distal/scaffold-support "
                        "hypothesis"
                    )
                else:
                    double_category = "ORTHOGONAL_MECHANISM_DOUBLE"
                    coupling = (
                        "spatially separated positions with distinct structural-region "
                        "mechanisms"
                    )
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
                    intended_physical_change=(
                        f"Combine {left.intended_physical_change} with "
                        f"{right.intended_physical_change}."
                    ),
                    feature_to_preserve=(
                        f"{left.feature_to_preserve}; {right.feature_to_preserve}."
                    ),
                    principal_biochemical_risk=(
                        "High combination-specific epistasis uncertainty; child structure "
                        "must be compared with both parents."
                    ),
                    metal_liability=(
                        "HIGH_RISK_METAL_SITE_HYPOTHESIS"
                        if "HIGH_RISK_METAL_SITE_HYPOTHESIS"
                        in {left.metal_liability, right.metal_liability}
                        else "none_identified"
                    ),
                    requires_candidate_geometry=(
                        left.requires_candidate_geometry
                        or right.requires_candidate_geometry
                    ),
                    double_category=double_category,
                    physical_coupling=coupling,
                    epistasis_uncertainty="high",
                    known_experiment_conflict="none_identified",
                    substitution_classes=(
                        left.substitution_classes[0],
                        right.substitution_classes[0],
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
