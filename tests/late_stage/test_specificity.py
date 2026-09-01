from pathlib import Path

import pytest

from atlas.late_stage.specificity import (
    assess_specificity,
    build_abeta42_s2_panel,
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
