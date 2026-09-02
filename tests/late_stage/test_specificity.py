from pathlib import Path

import pytest

from atlas.late_stage.specificity import (
    assess_sidechain_specificity,
    assess_specificity,
    build_abeta42_s2_panel,
    specificity_contact_construction_audit,
)
from atlas.structure.reconstruct import reconstruct_active_like


SOURCE = Path(__file__).parents[2] / "data" / "23WN.cif"


@pytest.fixture()
def active_like(tmp_path: Path) -> Path:
    output = tmp_path / "active_like.pdb"
    reconstruct_active_like(SOURCE, output, tmp_path / "numbering.csv")
    return output


def test_panel_uses_only_deposited_abeta42_windows_and_marks_s2_target() -> None:
    """A wrong sequence/window mapping must not become specificity evidence."""
    panel = build_abeta42_s2_panel(SOURCE)

    assert panel.source_structure == "PDB 23WN chain B"
    assert panel.source_accession == "UniProt P05067"
    assert panel.canonical_sequence == "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"
    assert [
        (member.start, member.end, member.sequence, member.is_intended)
        for member in panel.members
    ] == [
        (31, 38, "IIGLMVGG", False),
        (32, 39, "IGLMVGGV", False),
        (33, 40, "GLMVGGVV", False),
        (34, 41, "LMVGGVVI", True),
        (35, 42, "MVGGVVIA", False),
    ]


def test_specificity_scores_candidate_contacts_without_affinity_claims(
    active_like: Path,
) -> None:
    """Dropping real contact scoring or overstating it must fail this contract."""
    result = assess_specificity(active_like, SOURCE, variant_id="DP622-reference")

    assert result.variant_id == "DP622-reference"
    assert result.classification in {
        "intended_preferred",
        "off_target_warning",
        "indeterminate",
    }
    assert result.contact_count > 0
    assert len(result.contact_fingerprint) == result.contact_count
    assert [score.panel_member_id for score in result.scores] == [
        "abeta31_38",
        "abeta32_39",
        "abeta33_40",
        "s2_abeta34_41",
        "abeta35_42",
    ]
    assert sum(score.is_intended for score in result.scores) == 1
    assert result.intended_score is not None
    assert result.best_off_target_score is not None
    assert result.preference_margin == pytest.approx(
        result.intended_score - result.best_off_target_score
    )
    assert "contact-compatibility" in result.method
    assert "does not estimate binding affinity, cleavage kinetics, or proteome-wide specificity" in (
        result.interpretation_limit
    )


def test_missing_resolved_substrate_is_indeterminate_not_favorable(
    active_like: Path, tmp_path: Path
) -> None:
    """Missing chain-B evidence must never be interpreted as specificity support."""
    without_substrate = tmp_path / "without_substrate.pdb"
    without_substrate.write_text(
        "".join(
            line
            for line in active_like.read_text().splitlines(keepends=True)
            if not (line.startswith(("ATOM  ", "HETATM")) and line[21:22] == "B")
        )
    )

    result = assess_specificity(without_substrate, SOURCE, variant_id="missing-B")

    assert result.classification == "indeterminate"
    assert result.contact_count == 0
    assert result.intended_score is None
    assert result.best_off_target_score is None
    assert result.preference_margin is None
    assert any("resolved chain B" in warning for warning in result.warnings)


def _pdb_atom(
    serial: int,
    atom: str,
    residue: str,
    chain: str,
    number: int,
    x: float,
    y: float,
    element: str,
) -> str:
    return (
        f"ATOM  {serial:5d} {atom:>4s} {residue:>3s} {chain}{number:4d}    "
        f"{x:8.3f}{y:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}          "
        f"{element:>2s}\n"
    )


def test_sidechain_specificity_uses_candidate_chemistry_not_backbone_identity(
    tmp_path: Path,
) -> None:
    """Backbone-only proximity must not dilute a side-chain specificity hypothesis."""
    lines = [
        _pdb_atom(1, "CA", "GLU", "A", 10, -1.0, 0.0, "C"),
        _pdb_atom(2, "OE1", "GLU", "A", 10, 0.0, 0.0, "O"),
        _pdb_atom(3, "CA", "LYS", "A", 11, 7.0, 0.0, "C"),
        _pdb_atom(4, "NZ", "LYS", "A", 11, 6.0, 0.0, "N"),
        # This backbone-only residue is close but has no side-chain atom in the fixture.
        _pdb_atom(5, "CA", "LEU", "A", 12, 12.0, 0.0, "C"),
    ]
    substrate = ("LEU", "MET", "VAL", "GLY", "GLY", "VAL", "VAL", "ILE")
    coordinates = (
        (30.0, 30.0),
        (32.0, 30.0),
        (12.0, 3.0),
        (0.0, 3.0),
        (6.0, 3.0),
        (34.0, 30.0),
        (36.0, 30.0),
        (38.0, 30.0),
    )
    for offset, (residue, (x, y)) in enumerate(zip(substrate, coordinates)):
        lines.append(
            _pdb_atom(6 + offset, "CA", residue, "B", 34 + offset, x, y, "C")
        )
    path = tmp_path / "sidechain_contacts.pdb"
    path.write_text("".join(lines) + "TER\nEND\n")

    result = assess_sidechain_specificity(path, SOURCE, variant_id="sidechain-test")

    assert result.contact_count == 2
    assert result.classification == "intended_preferred"
    assert result.preference_margin is not None
    assert result.preference_margin > 0.05
    assert "side-chain" in result.method


def test_contact_construction_audit_counts_backbone_only_assignments(
    tmp_path: Path,
) -> None:
    """Method correction requires measured construction dominance, not intuition."""
    lines = [
        _pdb_atom(1, "CA", "LEU", "A", 10, 0.0, 0.0, "C"),
        _pdb_atom(2, "CA", "GLU", "A", 11, 8.0, 0.0, "C"),
        _pdb_atom(3, "OE1", "GLU", "A", 11, 9.0, 0.0, "O"),
    ]
    for offset, residue in enumerate(
        ("LEU", "MET", "VAL", "GLY", "GLY", "VAL", "VAL", "ILE")
    ):
        x = 0.0 if offset == 0 else 9.0 if offset == 4 else 30.0 + offset
        lines.append(_pdb_atom(4 + offset, "CA", residue, "B", 34 + offset, x, 3.0, "C"))
    path = tmp_path / "construction.pdb"
    path.write_text("".join(lines) + "TER\nEND\n")

    audit = specificity_contact_construction_audit(path)

    assert audit.total_residue_contacts == 2
    assert audit.chemistry_bearing_contacts == 1
    assert audit.backbone_only_contacts == 1
    assert audit.backbone_only_fraction == pytest.approx(0.5)
