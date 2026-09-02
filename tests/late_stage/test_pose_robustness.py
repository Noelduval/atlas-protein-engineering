from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import PDBParser

from atlas.late_stage.pose_robustness import (
    assess_candidate_pose_robustness,
    assess_local_pose_robustness,
    generate_local_pose_perturbations,
)
from atlas.structure.reconstruct import reconstruct_active_like


SOURCE = Path(__file__).parents[2] / "data" / "23WN.cif"


@pytest.fixture()
def active_like(tmp_path: Path) -> Path:
    output = tmp_path / "active_like.pdb"
    reconstruct_active_like(SOURCE, output, tmp_path / "numbering.csv")
    return output


def _coordinates(path: Path, chain: str) -> np.ndarray:
    structure = PDBParser(QUIET=True).get_structure(path.stem, path)
    return np.asarray(
        [atom.coord for atom in structure[0][chain].get_atoms()], dtype=float
    )


def test_local_perturbations_are_deterministic_and_move_only_resolved_abeta(
    active_like: Path, tmp_path: Path
) -> None:
    """Moving DP622/Zn or using a non-fixed pose set must fail this test."""
    first = generate_local_pose_perturbations(
        active_like, tmp_path / "first", variant_id="reference"
    )
    second = generate_local_pose_perturbations(
        active_like, tmp_path / "second", variant_id="reference"
    )

    assert [pose.pose_id for pose in first] == [
        "reference",
        "translate_x_plus",
        "translate_x_minus",
        "translate_y_plus",
        "translate_y_minus",
        "translate_z_plus",
        "translate_z_minus",
        "rotate_x_plus",
        "rotate_x_minus",
        "rotate_y_plus",
        "rotate_y_minus",
        "rotate_z_plus",
        "rotate_z_minus",
    ]
    assert len(first) == 13
    assert first[1].path.read_bytes() == second[1].path.read_bytes()

    source_abeta = _coordinates(active_like, "B")
    translated_abeta = _coordinates(first[1].path, "B")
    displacement = translated_abeta.mean(axis=0) - source_abeta.mean(axis=0)
    assert displacement == pytest.approx([0.35, 0.0, 0.0], abs=2e-3)
    assert _coordinates(first[1].path, "A") == pytest.approx(
        _coordinates(active_like, "A"), abs=2e-3
    )
    assert _coordinates(first[1].path, "C") == pytest.approx(
        _coordinates(active_like, "C"), abs=2e-3
    )


def test_deposited_pose_is_robust_to_bounded_local_perturbations(
    active_like: Path, tmp_path: Path
) -> None:
    """A complete, coherent local pose neighborhood should classify as robust."""
    result = assess_local_pose_robustness(
        active_like,
        reference_pdb=active_like,
        output_dir=tmp_path / "poses",
        variant_id="DP622-reference",
    )

    assert result.classification == "robust"
    assert len(result.samples) == 13
    assert result.max_substrate_rmsd_a is not None
    assert result.max_substrate_rmsd_a <= result.thresholds["substrate_rmsd_a"]
    assert result.max_substrate_pose_drift_a is not None
    assert result.max_substrate_pose_drift_a <= result.thresholds[
        "substrate_pose_drift_a"
    ]
    assert result.warnings == ()
    assert "resolved 23WN Aβ34–41 pose" in result.method
    assert "not a conformational ensemble of full-length Aβ42" in result.interpretation_limit


def test_large_deterministic_displacement_is_pose_sensitive(
    active_like: Path, tmp_path: Path
) -> None:
    """Threshold violations must not be mislabeled robust."""
    result = assess_local_pose_robustness(
        active_like,
        reference_pdb=active_like,
        output_dir=tmp_path / "large_poses",
        variant_id="large-displacement",
        translation_distance_a=1.5,
    )

    assert result.classification == "pose-sensitive"
    assert result.max_substrate_pose_drift_a is not None
    assert result.max_substrate_pose_drift_a > result.thresholds[
        "substrate_pose_drift_a"
    ]


def test_missing_zinc_geometry_is_indeterminate(active_like: Path, tmp_path: Path) -> None:
    """Incomplete catalytic geometry must not become a favorable pose call."""
    without_zinc = tmp_path / "without_zinc.pdb"
    without_zinc.write_text(
        "".join(
            line
            for line in active_like.read_text().splitlines(keepends=True)
            if not (line.startswith("HETATM") and line[76:78].strip() == "ZN")
        )
    )

    result = assess_local_pose_robustness(
        without_zinc,
        reference_pdb=active_like,
        output_dir=tmp_path / "incomplete",
        variant_id="missing-zinc",
    )

    assert result.classification == "indeterminate"
    assert any("incomplete catalytic geometry" in warning for warning in result.warnings)


def test_candidate_incremental_robustness_does_not_double_count_baseline_pose_offset(
    active_like: Path, tmp_path: Path
) -> None:
    """Baseline structural drift is gated elsewhere and must not be counted twice."""
    shifted = generate_local_pose_perturbations(
        active_like,
        tmp_path / "shifted_source",
        variant_id="shifted",
        translation_distance_a=0.30,
        rotation_degrees=0.0,
    )[1].path

    absolute = assess_local_pose_robustness(
        shifted,
        reference_pdb=active_like,
        output_dir=tmp_path / "absolute",
        variant_id="absolute",
    )
    incremental = assess_candidate_pose_robustness(
        shifted,
        active_reference_pdb=active_like,
        output_dir=tmp_path / "incremental",
        variant_id="incremental",
    )

    assert absolute.classification == "pose-sensitive"
    assert incremental.classification == "robust"
    assert incremental.minimum_contact_retention_fraction is not None
    assert incremental.minimum_contact_retention_fraction >= incremental.thresholds[
        "minimum_contact_retention_fraction"
    ]
    assert incremental.max_substrate_pose_drift_a is not None
    assert incremental.max_substrate_pose_drift_a <= incremental.thresholds[
        "substrate_pose_drift_a"
    ]
    assert "candidate baseline" in incremental.method


def test_candidate_incremental_robustness_rejects_large_local_perturbation(
    active_like: Path, tmp_path: Path
) -> None:
    """The candidate-relative correction must not weaken the existing displacement gate."""
    result = assess_candidate_pose_robustness(
        active_like,
        active_reference_pdb=active_like,
        output_dir=tmp_path / "large_incremental",
        variant_id="large-incremental",
        translation_distance_a=1.5,
    )

    assert result.classification == "pose-sensitive"
    assert result.max_substrate_pose_drift_a is not None
    assert result.max_substrate_pose_drift_a > result.thresholds[
        "substrate_pose_drift_a"
    ]
