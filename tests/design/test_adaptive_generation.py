from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from atlas.adaptive.ledger import ScientificLedger
from atlas.adaptive.models import DesignStrategy, FailureObservation
from atlas.design.adaptive_generator import AdaptiveGenerator, SearchBudget
from atlas.design.design_space import ResidueClass, classify_design_space
from atlas.design.strategies import substitution_policy
from atlas.structure.reconstruct import reconstruct_active_like


SOURCE = Path(__file__).parents[2] / "data" / "23WN.cif"


@pytest.fixture(scope="module")
def design_space(tmp_path_factory):
    directory = tmp_path_factory.mktemp("generation-space")
    pdb = directory / "active_like.pdb"
    reconstruct_active_like(SOURCE, pdb, directory / "map.csv")
    return classify_design_space(pdb, SOURCE)


def test_adaptive_rounds_generate_5000_unique_legal_variants(design_space) -> None:
    generator = AdaptiveGenerator(design_space, seed=622)
    result = generator.generate_seed_search(
        SearchBudget(candidate_budget=5_000, round1_target=1_200, minimum_doubles=750),
        position_priority=(69, 67, 91, 126, 159, 163),
    )

    assert len(result.candidates) == 5_000
    assert len({candidate.sequence for candidate in result.candidates}) == 5_000
    assert len({candidate.candidate_id for candidate in result.candidates}) == 5_000
    assert Counter(candidate.round_index for candidate in result.candidates)[1] == 1_200
    assert Counter(candidate.round_index for candidate in result.candidates)[3] >= 750
    assert all(len(candidate.mutations) in {1, 2} for candidate in result.candidates)

    protected = {
        record.position
        for record in design_space
        if record.residue_class in {ResidueClass.HARD_PROTECTED, ResidueClass.OUT_OF_SCOPE}
    }
    assert not protected.intersection(
        mutation.position
        for candidate in result.candidates
        for mutation in candidate.mutations
    )


def test_generation_uses_multiple_strategies_regions_and_documented_hypotheses(
    design_space,
) -> None:
    result = AdaptiveGenerator(design_space, seed=622).generate_seed_search(
        SearchBudget(candidate_budget=5_000, round1_target=1_200, minimum_doubles=750),
        position_priority=(69, 67, 91, 126),
    )
    assert set(result.strategy_counts) >= {
        DesignStrategy.SUBSTRATE_INTERFACE.value,
        DesignStrategy.STABILITY_SUPPORT.value,
        DesignStrategy.SECOND_SHELL_PREORGANIZATION.value,
        DesignStrategy.CONSERVATIVE.value,
        DesignStrategy.EVIDENCE_GUIDED_COMBINATION.value,
    }
    assert set(result.region_counts) >= {
        "substrate_interface",
        "second_shell",
        "distal_stability",
        "scaffold_surface",
    }
    assert all(candidate.hypothesis for candidate in result.candidates)
    assert all(candidate.intended_upside for candidate in result.candidates)
    assert all(candidate.expected_risk for candidate in result.candidates)
    assert all(candidate.intended_physical_change for candidate in result.candidates)
    assert all(candidate.feature_to_preserve for candidate in result.candidates)
    assert all(candidate.principal_biochemical_risk for candidate in result.candidates)
    assert all(candidate.substitution_classes for candidate in result.candidates)


def test_allowed_substitution_classes_control_proposal_identity(design_space) -> None:
    source = next(
        record
        for record in design_space
        if record.residue_class is ResidueClass.DESIGNABLE
        and record.wildtype not in "STNQ"
    )
    restricted = replace(
        source,
        allowed_substitution_classes=("hydrogen_bond",),
    )

    proposals = substitution_policy(restricted)

    assert proposals
    assert {proposal.substitution_class for proposal in proposals} == {"hydrogen_bond"}
    assert {proposal.mutant for proposal in proposals} <= set("STNQ")


def test_cgp_and_buried_charge_receive_context_specific_handling(design_space) -> None:
    source = next(
        record
        for record in design_space
        if record.residue_class is ResidueClass.DESIGNABLE
        and record.burial_class == "buried"
        and record.secondary_structure in {"helix", "strand"}
        and record.wildtype not in "CGP"
    )
    policy_record = replace(
        source,
        allowed_substitution_classes=(
            "conservative",
            "packing",
            "helix_propensity",
            "charge_balance",
        ),
    )

    proposals = substitution_policy(policy_record)
    mutants = {proposal.mutant for proposal in proposals}

    assert "P" not in mutants
    assert "G" not in mutants
    assert not mutants.intersection("DEKRH")


def test_near_zinc_coordination_capable_substitutions_are_explicitly_flagged(
    design_space,
) -> None:
    source = next(
        record
        for record in design_space
        if record.residue_class is ResidueClass.DESIGNABLE
        and record.structural_region == "second_shell"
        and record.min_zinc_distance_a <= 10.0
        and any(
            proposal.mutant in "HCDE" for proposal in substitution_policy(record)
        )
    )

    risky = [
        proposal
        for proposal in substitution_policy(source)
        if proposal.mutant in "HCDE"
    ]

    assert risky
    assert all(
        proposal.metal_liability == "HIGH_RISK_METAL_SITE_HYPOTHESIS"
        for proposal in risky
    )
    assert all(proposal.requires_candidate_geometry for proposal in risky)


def test_context_sensitive_sites_receive_bounded_context_aware_substitutions(
    design_space,
) -> None:
    result = AdaptiveGenerator(design_space, seed=622).generate_seed_search(
        SearchBudget(candidate_budget=5_000, round1_target=1_200, minimum_doubles=750)
    )
    mutants_by_position: dict[int, set[str]] = {91: set(), 126: set()}
    for candidate in result.candidates:
        for mutation in candidate.mutations:
            if mutation.position in mutants_by_position:
                mutants_by_position[mutation.position].add(mutation.mutant)
    assert mutants_by_position[91] <= {"F", "W", "H"}
    assert mutants_by_position[126] <= {"E", "N", "A"}
    assert mutants_by_position[91]
    assert mutants_by_position[126]


def test_doubles_are_mechanistically_categorized_and_known_construct_is_retrospective_only(
    design_space,
) -> None:
    result = AdaptiveGenerator(design_space, seed=622).generate_seed_search(
        SearchBudget(candidate_budget=5_000, round1_target=1_200, minimum_doubles=750)
    )
    doubles = [candidate for candidate in result.candidates if len(candidate.mutations) == 2]

    assert doubles
    assert {candidate.double_category for candidate in doubles} <= {
        "LOCAL_COUPLED_DOUBLE",
        "FUNCTION_STABILITY_RESCUE_DOUBLE",
        "ORTHOGONAL_MECHANISM_DOUBLE",
    }
    assert all(candidate.double_category for candidate in doubles)
    assert all(candidate.physical_coupling for candidate in doubles)
    assert all(candidate.epistasis_uncertainty == "high" for candidate in doubles)
    assert all(len(candidate.substitution_classes) == 2 for candidate in doubles)
    assert all(candidate.mutation_set != "Y91F/D126A" for candidate in doubles)


def test_high_confidence_failure_memory_deprioritizes_but_does_not_make_soft_law(
    design_space,
) -> None:
    generator = AdaptiveGenerator(design_space, seed=622)
    budget = SearchBudget(candidate_budget=5_000, round1_target=1_200, minimum_doubles=750)
    legal_label = generator.generate_round1(budget)[0].mutations[0].label
    memory = FailureObservation(
        candidate_id=None,
        category="destabilizing_substitution",
        scope=f"mutation:{legal_label}",
        detail="Repeated destabilization in this local context.",
        evidence_count=4,
        confidence=0.9,
        source="ThermoMPNN",
    )
    result = generator.generate_seed_search(
        budget,
        failure_memory=(memory,),
    )
    labels = [
        mutation.label
        for candidate in result.candidates
        for mutation in candidate.mutations
    ]
    assert legal_label in labels
    assert result.failure_memory_deprioritized > 0
    first_penalized = next(
        index
        for index, candidate in enumerate(result.candidates)
        if any(mutation.label == legal_label for mutation in candidate.mutations)
    )
    assert first_penalized >= 1_200


def test_generation_is_resume_safe_against_ledger_identity(design_space, tmp_path: Path) -> None:
    ledger = ScientificLedger.create(tmp_path / "ledger.sqlite", tmp_path / "events.jsonl")
    generator = AdaptiveGenerator(design_space, seed=622)
    budget = SearchBudget(candidate_budget=250, round1_target=150, minimum_doubles=50)
    first = generator.generate_seed_search(budget)
    persisted_first = generator.persist_new(first.candidates, ledger)
    persisted_second = generator.persist_new(first.candidates, ledger)

    assert persisted_first == 250
    assert persisted_second == 0
    assert ledger.candidate_count() == 250


def test_rounds_can_be_allocated_after_real_feedback(design_space) -> None:
    generator = AdaptiveGenerator(design_space, seed=622)
    budget = SearchBudget(candidate_budget=5_000, round1_target=1_200, minimum_doubles=750)
    round1 = generator.generate_round1(budget)
    memory = (
        FailureObservation(
            candidate_id=None,
            category="destabilizing_substitution",
            scope="mutation:A37P",
            detail="Repeated Round-1 stability regression.",
            evidence_count=5,
            confidence=0.9,
            source="ThermoMPNN",
        ),
    )
    round2 = generator.generate_round2(
        budget,
        round1=round1,
        position_priority=(69, 67, 91),
        failure_memory=memory,
    )
    preferred = tuple(candidate.candidate_id for candidate in round1[:20])
    round3 = generator.generate_round3(
        budget,
        singles=round1 + round2,
        preferred_candidate_ids=preferred,
        failure_memory=memory,
    )

    assert len(round1) == 1_200
    assert len(round1 + round2 + round3) == 5_000
    assert all(candidate.round_index == 2 for candidate in round2)
    assert all(candidate.round_index == 3 for candidate in round3)
    assert any(set(candidate.parents).intersection(preferred) for candidate in round3[:250])


def test_round2_consults_scoped_substitution_pattern_memory(design_space) -> None:
    generator = AdaptiveGenerator(design_space, seed=622)
    budget = SearchBudget(candidate_budget=5_000, round1_target=1_200, minimum_doubles=750)
    round1 = generator.generate_round1(budget)
    baseline_round2 = generator.generate_round2(budget, round1=round1)
    penalized_region = baseline_round2[0].structural_region
    penalized_mutant = baseline_round2[0].mutations[0].mutant
    memory = (
        FailureObservation(
            candidate_id=None,
            category="recurrent_stability_regression",
            scope=f"substitution:{penalized_region}:{penalized_mutant}",
            detail="This legal substitution pattern repeatedly regressed in this region.",
            evidence_count=8,
            confidence=0.85,
            source="Round-1 ThermoMPNN aggregation",
        ),
    )
    round2 = generator.generate_round2(
        budget,
        round1=round1,
        failure_memory=memory,
    )
    penalized = [
        index
        for index, candidate in enumerate(round2)
        if candidate.structural_region == penalized_region
        and candidate.mutations[0].mutant == penalized_mutant
    ]
    unpenalized = [
        index
        for index, candidate in enumerate(round2)
        if not (
            candidate.structural_region == penalized_region
            and candidate.mutations[0].mutant == penalized_mutant
        )
    ]
    assert penalized
    assert unpenalized
    assert min(penalized) > min(unpenalized)
