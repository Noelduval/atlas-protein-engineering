from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlas.late_stage.developability import screen_developability


def _write_peptide_pdb(path: Path, residues: tuple[str, ...]) -> None:
    lines = []
    serial = 1
    for residue_index, residue in enumerate(residues, start=1):
        base = (residue_index - 1) * 3.8
        atoms = [
            ("N", "N", base, 0.0, 0.0),
            ("CA", "C", base + 1.45, 0.0, 0.0),
            ("C", "C", base + 2.90, 0.0, 0.0),
            ("O", "O", base + 3.35, -1.1, 0.0),
            ("CB", "C", base + 1.45, 1.5, 0.0),
        ]
        if residue == "VAL":
            atoms.extend(
                [
                    ("CG1", "C", base + 2.45, 2.45, 0.7),
                    ("CG2", "C", base + 0.35, 2.25, -0.7),
                ]
            )
        for atom_name, element, x, y, z in atoms:
            lines.append(
                f"ATOM  {serial:5d} {atom_name:^4s} {residue:>3s} A{residue_index:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2s}"
            )
            serial += 1
    path.write_text("\n".join(lines) + "\nTER\nEND\n")


def test_missing_structure_requires_review_and_persists_claim_boundary(
    tmp_path: Path,
) -> None:
    result = screen_developability(
        candidate_id="ATLAS-MISSING",
        sequence="ACDEFGHIKLMNPQRSTVWY",
        output_dir=tmp_path,
    )

    assert result.status == "indeterminate"
    assert result.outcome == "review_required"
    assert result.uncertainty["level"] == "high"
    assert "structure_evidence_unavailable" in {flag.code for flag in result.flags}
    assert result.artifact_path == tmp_path / "developability.json"

    persisted = json.loads(result.artifact_path.read_text())
    assert persisted["provenance"]["method"] == (
        "deterministic_sequence_structure_heuristics"
    )
    assert persisted["provenance"]["excluded_claims"] == [
        "safety",
        "blood-brain barrier penetration",
        "pharmacokinetics",
        "immunogenicity",
        "clinical developability",
    ]
    assert "score" not in persisted


def test_sequence_checks_flag_hydrophobic_windows_and_cysteine_review(
    tmp_path: Path,
) -> None:
    result = screen_developability(
        candidate_id="ATLAS-SEQUENCE",
        sequence="VVVVVVVCCDEFGHIKLMNP",
        reference_sequence="AAAAAAAACDEFGHIKLMNP",
        output_dir=tmp_path,
    )

    codes = {flag.code for flag in result.flags}
    assert "hydrophobic_sequence_patch" in codes
    assert "cysteine_requires_review" in codes
    assert result.sequence_metrics["length"] == 20
    assert result.sequence_metrics["max_hydrophobic_residues_in_7mer"] == 7
    assert result.sequence_metrics["added_chemical_liability_motifs"] == {}


def test_structure_checks_detect_an_exposed_hydrophobic_patch(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "candidate.pdb"
    _write_peptide_pdb(structure_path, ("VAL", "VAL", "VAL", "VAL"))

    result = screen_developability(
        candidate_id="ATLAS-STRUCTURE",
        sequence="VVVV",
        reference_sequence="AAAA",
        structure_path=structure_path,
        output_dir=tmp_path / "result",
    )

    assert result.status == "completed"
    assert result.outcome == "concerns_identified"
    assert result.uncertainty["level"] == "moderate"
    assert result.structure_metrics is not None
    assert result.structure_metrics["sequence_coverage_fraction"] == 1.0
    assert result.structure_metrics["largest_exposed_hydrophobic_patch"] == 4
    structure_flag = next(
        flag for flag in result.flags if flag.code == "exposed_hydrophobic_surface_patch"
    )
    assert structure_flag.evidence["residues"] == ["A:1", "A:2", "A:3", "A:4"]
    assert result.provenance["structure_sha256"]


def test_structure_sequence_mismatch_is_indeterminate_not_favorable(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "mismatch.pdb"
    _write_peptide_pdb(structure_path, ("ALA", "ALA", "ALA"))

    result = screen_developability(
        candidate_id="ATLAS-MISMATCH",
        sequence="VVV",
        structure_path=structure_path,
        output_dir=tmp_path / "result",
    )

    assert result.status == "indeterminate"
    assert result.outcome == "review_required"
    assert result.uncertainty["level"] == "high"
    assert "structure_sequence_mismatch" in {flag.code for flag in result.flags}
    assert result.structure_metrics is None


@pytest.mark.parametrize("sequence", ["", "ACDX", "acde"])
def test_noncanonical_or_unnormalized_sequence_is_rejected(
    tmp_path: Path, sequence: str
) -> None:
    with pytest.raises(ValueError, match="standard uppercase amino acids"):
        screen_developability(
            candidate_id="ATLAS-BAD",
            sequence=sequence,
            output_dir=tmp_path,
        )


def test_artifact_is_deterministic_for_identical_inputs(tmp_path: Path) -> None:
    first = screen_developability(
        candidate_id="ATLAS-REPEAT",
        sequence="ACDEFGHIKLMNPQRSTVWY",
        output_dir=tmp_path / "one",
    )
    second = screen_developability(
        candidate_id="ATLAS-REPEAT",
        sequence="ACDEFGHIKLMNPQRSTVWY",
        output_dir=tmp_path / "two",
    )

    assert json.loads(first.artifact_path.read_text()) == json.loads(
        second.artifact_path.read_text()
    )
