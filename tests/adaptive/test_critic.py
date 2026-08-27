from __future__ import annotations

from pathlib import Path

import pytest

from atlas.adaptive.critic import CriticPolicy, CriticRoute
from atlas.adaptive.ledger import ScientificLedger
from atlas.adaptive.models import (
    CandidateRecord,
    DesignStrategy,
    EvidenceAxis,
    EvidenceRecord,
    HardViolation,
)
from atlas.adaptive.repair import RepairGenerator
from atlas.adaptive.screening import Evaluation
from atlas.design.design_space import ResidueClass, ResidueDesignRecord


REFERENCE = "ACDEFGHIKLMNPQRSTVWY"


def _candidate(mutation: str = "C2S", *, generation: int = 0) -> CandidateRecord:
    return CandidateRecord.create(
        reference_sequence=REFERENCE,
        mutations=[mutation],
        parents=() if generation == 0 else ("ATLAS-PARENT",),
        strategy=DesignStrategy.SUBSTRATE_INTERFACE,
        structural_region="substrate_interface",
        round_index=3,
        hypothesis="Preserve a useful substrate contact.",
        intended_upside="Improve substrate-interface evidence.",
        expected_risk="Predicted stability regression.",
        revision_generation=generation,
    )


def _record(position: int, wildtype: str, region: str) -> ResidueDesignRecord:
    return ResidueDesignRecord(
        position=position,
        wildtype=wildtype,
        residue_name="ALA",
        residue_class=ResidueClass.DESIGNABLE,
        structural_region=region,
        model_label="active_like_inferred",
        catalytic_role="none_assigned",
        zinc_coordination=False,
        min_zinc_distance_a=12.0,
        min_substrate_distance_a=9.0,
        sasa_a2=20.0,
        relative_sasa=0.2,
        burial_class="partially_buried",
        packing_neighbors=8,
        secondary_structure="helix",
        mean_b_factor=20.0,
        structural_uncertainty="deposited_coordinate_supported",
        experimental_evidence="No position-specific result.",
        protection_reason="",
        allowed_substitution_classes=("conservative", "packing"),
        classification_rule="test fixture",
    )


def _numeric(candidate, axis, value):
    return EvidenceRecord.numeric(
        candidate.candidate_id,
        axis,
        value=value,
        uncertainty=0.2,
        method="test",
        provenance={"test": True},
    )


def test_critic_routes_supported_repairable_near_miss_to_revision() -> None:
    candidate = _candidate()
    evaluation = Evaluation(
        candidate,
        (
            _numeric(candidate, EvidenceAxis.SUBSTRATE_INTERFACE, 0.9),
            _numeric(candidate, EvidenceAxis.STABILITY, 1.8),
        ),
    )
    critique = CriticPolicy().critique(evaluation)
    assert critique.route is CriticRoute.REVISE
    assert critique.supporting_signals
    assert critique.weakness_axis is EvidenceAxis.STABILITY
    assert critique.repairable is True
    assert critique.feature_to_preserve


def test_critic_never_repairs_a_hard_violation() -> None:
    candidate = _candidate()
    evaluation = Evaluation(
        candidate,
        (),
        hard_violations=(
            HardViolation(candidate.candidate_id, "protected_residue", "E96 changed"),
        ),
    )
    critique = CriticPolicy().critique(evaluation)
    assert critique.route is CriticRoute.REJECT
    assert critique.repairable is False


def test_lower_is_better_structure_and_dynamics_do_not_become_weaknesses() -> None:
    candidate = _candidate()
    evaluation = Evaluation(
        candidate,
        (
            _numeric(candidate, EvidenceAxis.STRUCTURE_QUALITY, 0.1),
            _numeric(candidate, EvidenceAxis.DYNAMICS, 0.2),
            _numeric(candidate, EvidenceAxis.SUBSTRATE_INTERFACE, 0.9),
        ),
    )

    critique = CriticPolicy().critique(evaluation)

    assert critique.route is CriticRoute.PROMOTE
    assert critique.weakness_axis is None


def test_repair_children_are_bounded_and_document_preservation_goal() -> None:
    parent = _candidate()
    evaluation = Evaluation(
        parent,
        (
            _numeric(parent, EvidenceAxis.SUBSTRATE_INTERFACE, 0.9),
            _numeric(parent, EvidenceAxis.STABILITY, 1.8),
        ),
    )
    critique = CriticPolicy().critique(evaluation)
    records = tuple(
        _record(position, wildtype, "distal_stability" if position > 10 else "substrate_interface")
        for position, wildtype in enumerate(REFERENCE, start=1)
    )
    generator = RepairGenerator(records, max_children_per_parent_round=3, max_generations=2)
    proposals = generator.propose(parent, critique, round_index=4)

    assert len(proposals) == 3
    assert all(proposal.child.parents == (parent.candidate_id,) for proposal in proposals)
    assert all(proposal.child.revision_generation == 1 for proposal in proposals)
    assert all(proposal.feature_to_preserve == critique.feature_to_preserve for proposal in proposals)
    assert len({proposal.child.sequence for proposal in proposals}) == 3
    assert generator.propose(parent, critique, round_index=4) == ()


def test_repair_generation_stops_after_two_generations() -> None:
    parent = _candidate(generation=2)
    evaluation = Evaluation(
        parent,
        (
            _numeric(parent, EvidenceAxis.SUBSTRATE_INTERFACE, 0.9),
            _numeric(parent, EvidenceAxis.STABILITY, 1.8),
        ),
    )
    critique = CriticPolicy().critique(evaluation)
    records = tuple(
        _record(position, wildtype, "distal_stability")
        for position, wildtype in enumerate(REFERENCE, start=1)
    )
    assert RepairGenerator(records).propose(parent, critique, round_index=5) == ()


def test_real_repair_trajectory_persists_evidence_change(tmp_path: Path) -> None:
    parent = _candidate()
    records = tuple(
        _record(position, wildtype, "distal_stability")
        for position, wildtype in enumerate(REFERENCE, start=1)
    )
    parent_evaluation = Evaluation(
        parent,
        (
            _numeric(parent, EvidenceAxis.SUBSTRATE_INTERFACE, 0.9),
            _numeric(parent, EvidenceAxis.STABILITY, 1.8),
        ),
    )
    critique = CriticPolicy().critique(parent_evaluation)
    proposal = RepairGenerator(records).propose(parent, critique, round_index=4)[0]
    ledger = ScientificLedger.create(tmp_path / "ledger.sqlite", tmp_path / "events.jsonl")
    ledger.add_candidate(parent)
    ledger.add_evidence(parent_evaluation.evidence[0])
    ledger.add_evidence(parent_evaluation.evidence[1])
    ledger.record_decision(
        candidate_id=parent.candidate_id,
        route="REVISE",
        rationale=critique.rationale,
        round_index=3,
    )
    ledger.add_candidate(proposal.child)
    ledger.record_repair(proposal)
    child_stability = _numeric(proposal.child, EvidenceAxis.STABILITY, 0.4)
    ledger.add_evidence(child_stability)
    ledger.record_repair_outcome(
        parent_id=parent.candidate_id,
        child_id=proposal.child.candidate_id,
        evidence_delta={"stability": -1.4},
        disposition="PROMOTE",
    )

    trajectory = ledger.repair_trajectory(parent.candidate_id)
    assert trajectory[0]["child_id"] == proposal.child.candidate_id
    assert trajectory[0]["evidence_delta"] == {"stability": -1.4}
    assert trajectory[0]["disposition"] == "PROMOTE"
