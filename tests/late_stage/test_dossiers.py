from __future__ import annotations

from pathlib import Path

import pytest

from atlas.adaptive.models import CandidateRecord, DesignStrategy
from atlas.late_stage.dossiers import (
    REQUIRED_HYPOTHESIS_LABEL,
    build_finalist_dossier,
    write_finalist_dossiers,
)


def _candidate(mutation: str = "A1V") -> CandidateRecord:
    reference = "ACDEFGHIKLMNPQRSTVWY"
    return CandidateRecord.create(
        reference_sequence=reference,
        mutations=[mutation],
        parents=(),
        strategy=DesignStrategy.SUBSTRATE_INTERFACE,
        structural_region="substrate_interface",
        round_index=3,
        hypothesis="Preserve the resolved catalytic arrangement while changing a substrate-facing contact.",
        intended_upside="Retain intended-substrate contact compatibility.",
        expected_risk="The local structural model may not predict experimental behavior.",
    )


def _evidence() -> dict[str, object]:
    return {
        "why_atlas_proposed": {
            "strategy": "substrate_interface_explorer",
            "rationale": "Complementary Pareto support without combining raw model scales.",
        },
        "thermompnn": {
            "ThermoMPNN": {"raw_value": -0.31, "status": "available"},
            "ThermoMPNN-D": {"raw_value": 0.42, "status": "available"},
            "interpretation": "Separate stability-oriented model outputs.",
        },
        "orthogonal_structure": {
            "status": "unavailable",
            "reason": "External predictor did not execute.",
        },
        "catalytic_preorganization": {
            "status": "available",
            "maximum_grounded_distance_deviation_a": 0.24,
        },
        "abeta_contacts": {
            "status": "available",
            "resolved_pose_contact_retention": 0.88,
        },
        "specificity": {
            "status": "available",
            "intended_context_rank": 1,
            "scope": "bounded peptide panel",
        },
        "pose_robustness": {
            "classification": "robust",
            "scope": "bounded local perturbations of the resolved pose",
        },
        "developability": {
            "status": "available",
            "risks": ["one exposed hydrophobic substitution"],
        },
        "disagreements_uncertainty": [
            "Orthogonal sequence-to-structure evidence is unavailable.",
            "The resolved Aβ fragment is not a full Aβ42 ensemble.",
        ],
        "reason_it_beat_near_misses": (
            "Retained catalytic geometry and intended-context compatibility while the "
            "near-miss had an unresolved contact warning."
        ),
        "falsification_criteria": [
            "Expression or folding is worse than the matched active-like reference.",
            "Cleavage-site-resolved assays do not support the proposed substrate-context behavior.",
            "Zn-dependence controls contradict the proposed catalytic mechanism.",
        ],
    }


def test_dossier_contains_complete_bounded_experimental_handoff() -> None:
    candidate = _candidate()
    text = build_finalist_dossier(
        candidate,
        candidate_structure="artifacts/ATLAS-TEST/mutant_complex.pdb",
        late_stage_evidence=_evidence(),
        near_misses=[
            {
                "candidate_id": "ATLAS-NEAR-MISS",
                "exact_mutations": "C2S",
                "comparison": "Unresolved substrate-contact warning.",
            }
        ],
    )

    required_sections = (
        "Candidate ID",
        "Exact mutations",
        "FASTA",
        "Candidate-specific structure",
        "Mechanistic hypothesis",
        "Why Atlas proposed it",
        "ThermoMPNN/ThermoMPNN-D evidence",
        "Orthogonal structure evidence",
        "Catalytic/preorganization evidence",
        "Aβ contact evidence",
        "Specificity evidence",
        "Pose robustness",
        "Developability risks",
        "Disagreements/uncertainty",
        "Why it beat near-misses",
        "Near-miss comparison",
        "Explicit falsification criteria",
        "Recommended wet-lab sequence",
    )
    assert all(section in text for section in required_sections)
    assert REQUIRED_HYPOTHESIS_LABEL in text
    assert f">{candidate.candidate_id}|{candidate.mutation_set}" in text
    assert candidate.sequence in text
    assert "ATLAS-NEAR-MISS" in text
    assert "External predictor did not execute" in text
    assert (
        "expression/purification\n"
        "→ folding/stability\n"
        "→ cleavage-site-resolved Aβ assay\n"
        "→ matched kinetic characterization\n"
        "→ Zn-dependence controls\n"
        "→ non-cognate specificity profiling\n"
        "→ structural confirmation if warranted"
    ) in text

    forbidden_claims = (
        "better catalysis",
        "measured specificity",
        "therapeutic efficacy",
        "plaque clearance",
        "experimental validation",
        "BBB penetration",
        "favorable PK",
        "non-immunogenicity",
        "clinical developability",
    )
    lowered = text.casefold()
    assert all(claim.casefold() not in lowered for claim in forbidden_claims)


def test_writer_rejects_more_than_five_finalists(tmp_path: Path) -> None:
    candidates = tuple(_candidate(mutation) for mutation in ("A1V", "C2S", "D3N", "E4Q", "F5Y", "G6A"))
    structures = {candidate.candidate_id: "mutant.pdb" for candidate in candidates}
    evidence = {candidate.candidate_id: _evidence() for candidate in candidates}

    with pytest.raises(ValueError, match="at most five finalists"):
        write_finalist_dossiers(
            output_dir=tmp_path,
            finalists=candidates,
            candidate_structures=structures,
            late_stage_evidence=evidence,
            near_misses=(),
        )

    assert list(tmp_path.iterdir()) == []


def test_dossier_rejects_inputs_that_assert_forbidden_claims() -> None:
    evidence = _evidence()
    evidence["specificity"] = {"claim": "Measured specificity is superior."}

    with pytest.raises(ValueError, match="forbidden scientific claim"):
        build_finalist_dossier(
            _candidate(),
            candidate_structure="mutant.pdb",
            late_stage_evidence=evidence,
            near_misses=(),
        )
