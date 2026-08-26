from __future__ import annotations

import pytest

from atlas.adaptive.models import (
    CandidateDisposition,
    CandidateRecord,
    CompletionAudit,
    DesignStrategy,
    EvidenceAxis,
    EvidenceRecord,
    HardViolation,
    Mutation,
)


REFERENCE = "ACDEFGHIKLMNPQRSTVWY"


def test_mutations_are_normalized_and_candidate_identity_is_sequence_stable() -> None:
    first = CandidateRecord.create(
        reference_sequence=REFERENCE,
        mutations=["V18A", "C2S"],
        parents=("ATLAS-PARENT",),
        strategy=DesignStrategy.EVIDENCE_GUIDED_COMBINATION,
        structural_region="substrate_interface",
        round_index=3,
        hypothesis="Combine independent interface and packing support.",
        intended_upside="Preserve substrate contacts while improving packing.",
        expected_risk="Epistatic destabilization.",
    )
    second = CandidateRecord.create(
        reference_sequence=REFERENCE,
        mutations=[Mutation.parse("C2S"), Mutation.parse("V18A")],
        parents=("ANOTHER-PARENT",),
        strategy=DesignStrategy.REPAIR_RESCUE,
        structural_region="second_shell",
        round_index=4,
        hypothesis="Different metadata cannot change sequence identity.",
        intended_upside="Repair.",
        expected_risk="None.",
    )

    assert first.mutation_set == "C2S/V18A"
    assert first.sequence == "ASDEFGHIKLMNPQRSTA WY".replace(" ", "")
    assert first.candidate_id == second.candidate_id
    assert first.candidate_id.startswith("ATLAS-")


@pytest.mark.parametrize("mutation", ["", "A0V", "A1A", "X3A", "A999V"])
def test_illegal_mutations_fail_before_candidate_creation(mutation: str) -> None:
    with pytest.raises(ValueError):
        CandidateRecord.create(
            reference_sequence=REFERENCE,
            mutations=[mutation],
            parents=(),
            strategy=DesignStrategy.CONSERVATIVE,
            structural_region="distal_stability",
            round_index=1,
            hypothesis="test",
            intended_upside="test",
            expected_risk="test",
        )


def test_candidate_records_keep_lineage_and_independent_evidence_axes() -> None:
    candidate = CandidateRecord.create(
        reference_sequence=REFERENCE,
        mutations=["C2S"],
        parents=("ATLAS-PARENT",),
        strategy=DesignStrategy.REPAIR_RESCUE,
        structural_region="substrate_interface",
        round_index=4,
        hypothesis="Remove a diagnosed clash.",
        intended_upside="Preserve the parent's substrate contact.",
        expected_risk="May lose packing support.",
        revision_generation=1,
    )
    stability = EvidenceRecord.numeric(
        candidate.candidate_id,
        EvidenceAxis.STABILITY,
        value=-0.2,
        uncertainty=0.3,
        method="ThermoMPNN",
        provenance={"commit": "abc"},
    )
    geometry = EvidenceRecord.numeric(
        candidate.candidate_id,
        EvidenceAxis.CATALYTIC_GEOMETRY,
        value=0.4,
        uncertainty=0.1,
        method="Atlas geometry",
        provenance={"structure": "mutant.pdb"},
    )

    assert candidate.parents == ("ATLAS-PARENT",)
    assert candidate.revision_generation == 1
    assert stability.axis is EvidenceAxis.STABILITY
    assert geometry.axis is EvidenceAxis.CATALYTIC_GEOMETRY
    assert not hasattr(candidate, "ranking_score")


def test_hard_violations_are_not_soft_evidence() -> None:
    violation = HardViolation(
        candidate_id="ATLAS-X",
        code="protected_residue",
        detail="E96 cannot be mutated.",
    )
    assert violation.hard is True
    assert CandidateDisposition.HARD_REJECTED.value == "hard_rejected"


def test_zero_finalists_requires_search_and_repair_exhaustion() -> None:
    incomplete = CompletionAudit(
        unique_legal_evaluated=4_999,
        candidate_budget=5_000,
        strategies_used=frozenset(strategy.value for strategy in DesignStrategy),
        required_strategies=frozenset(strategy.value for strategy in DesignStrategy),
        structural_regions_explored=frozenset({"substrate_interface", "second_shell"}),
        minimum_structural_regions=2,
        pending_candidates=0,
        pending_checkpoints=0,
        eligible_unrepaired_near_misses=0,
        finalists=0,
        near_misses_reported=3,
        promotable_without_hard_relaxation=0,
    )
    assert incomplete.complete is False
    assert "candidate budget" in " ".join(incomplete.blockers).lower()

    complete = CompletionAudit(
        unique_legal_evaluated=5_127,
        candidate_budget=5_000,
        strategies_used=frozenset(strategy.value for strategy in DesignStrategy),
        required_strategies=frozenset(strategy.value for strategy in DesignStrategy),
        structural_regions_explored=frozenset(
            {"substrate_interface", "second_shell", "distal_stability"}
        ),
        minimum_structural_regions=2,
        pending_candidates=0,
        pending_checkpoints=0,
        eligible_unrepaired_near_misses=0,
        finalists=0,
        near_misses_reported=5,
        promotable_without_hard_relaxation=0,
    )
    assert complete.complete is True
    assert complete.blockers == ()

