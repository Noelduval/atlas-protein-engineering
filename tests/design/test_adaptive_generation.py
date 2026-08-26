from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from atlas.adaptive.ledger import ScientificLedger
from atlas.adaptive.models import DesignStrategy, FailureObservation
from atlas.design.adaptive_generator import AdaptiveGenerator, SearchBudget
from atlas.design.design_space import ResidueClass, classify_design_space
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


def test_high_confidence_failure_memory_deprioritizes_but_does_not_make_soft_law(
    design_space,
) -> None:
    memory = FailureObservation(
        candidate_id=None,
        category="destabilizing_substitution",
        scope="mutation:A37P",
        detail="Repeated destabilization in this local context.",
        evidence_count=4,
        confidence=0.9,
        source="ThermoMPNN",
    )
    result = AdaptiveGenerator(design_space, seed=622).generate_seed_search(
        SearchBudget(candidate_budget=5_000, round1_target=1_200, minimum_doubles=750),
        failure_memory=(memory,),
    )
    labels = [
        mutation.label
        for candidate in result.candidates
        for mutation in candidate.mutations
    ]
    assert "A37P" in labels
    assert result.failure_memory_deprioritized > 0
    first_penalized = next(
        index
        for index, candidate in enumerate(result.candidates)
        if any(mutation.label == "A37P" for mutation in candidate.mutations)
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

