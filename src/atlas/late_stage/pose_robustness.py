"""Deterministic local robustness checks for the resolved 23WN Aβ pose."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.PDB import PDBIO, PDBParser

from atlas.geometry.catalytic_metrics import GeometryRecord, measure_geometry


@dataclass(frozen=True)
class PosePerturbation:
    pose_id: str
    path: Path
    translation_a: tuple[float, float, float]
    rotation_axis: str | None
    rotation_degrees: float


@dataclass(frozen=True)
class PoseGeometrySample:
    pose_id: str
    path: Path
    geometry: GeometryRecord


@dataclass(frozen=True)
class PoseRobustnessAssessment:
    variant_id: str
    classification: str
    samples: tuple[PoseGeometrySample, ...]
    thresholds: dict[str, float]
    max_substrate_rmsd_a: float | None
    max_substrate_pose_drift_a: float | None
    max_zn_scissile_distance_delta_a: float | None
    max_e96_scissile_distance_delta_a: float | None
    max_h172_scissile_distance_delta_a: float | None
    method: str
    interpretation_limit: str
    warnings: tuple[str, ...]


_THRESHOLDS = {
    "substrate_rmsd_a": 0.60,
    "substrate_pose_drift_a": 0.50,
    "zn_scissile_distance_delta_a": 0.50,
    "e96_scissile_distance_delta_a": 0.75,
    "h172_scissile_distance_delta_a": 0.75,
}


def _rotation_matrix(axis: str, degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    if axis == "x":
        return np.asarray(((1, 0, 0), (0, cosine, -sine), (0, sine, cosine)))
    if axis == "y":
        return np.asarray(((cosine, 0, sine), (0, 1, 0), (-sine, 0, cosine)))
    if axis == "z":
        return np.asarray(((cosine, -sine, 0), (sine, cosine, 0), (0, 0, 1)))
    raise ValueError(f"Unsupported rotation axis: {axis}")


def _specifications(
    translation_distance_a: float, rotation_degrees: float
) -> tuple[tuple[str, tuple[float, float, float], str | None, float], ...]:
    specifications: list[tuple[str, tuple[float, float, float], str | None, float]] = [
        ("reference", (0.0, 0.0, 0.0), None, 0.0)
    ]
    for axis_index, axis in enumerate("xyz"):
        for label, sign in (("plus", 1.0), ("minus", -1.0)):
            translation = [0.0, 0.0, 0.0]
            translation[axis_index] = sign * translation_distance_a
            specifications.append(
                (f"translate_{axis}_{label}", tuple(translation), None, 0.0)
            )
    for axis in "xyz":
        for label, sign in (("plus", 1.0), ("minus", -1.0)):
            specifications.append(
                (f"rotate_{axis}_{label}", (0.0, 0.0, 0.0), axis, sign * rotation_degrees)
            )
    return tuple(specifications)


def generate_local_pose_perturbations(
    candidate_pdb: str | Path,
    output_dir: str | Path,
    *,
    variant_id: str | None = None,
    translation_distance_a: float = 0.35,
    rotation_degrees: float = 2.5,
) -> tuple[PosePerturbation, ...]:
    """Write the fixed local chain-B rigid-body perturbation set."""
    if translation_distance_a < 0 or rotation_degrees < 0:
        raise ValueError("Perturbation magnitudes must be non-negative")
    source = Path(candidate_pdb)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    identifier = variant_id or source.stem
    results: list[PosePerturbation] = []

    for pose_id, translation, rotation_axis, angle in _specifications(
        translation_distance_a, rotation_degrees
    ):
        structure = PDBParser(QUIET=True).get_structure(identifier, source)
        model = next(structure.get_models())
        if "B" not in model:
            raise ValueError("Candidate does not contain resolved 23WN chain B")
        atoms = list(model["B"].get_atoms())
        if not atoms:
            raise ValueError("Candidate 23WN chain B contains no atoms")
        coordinates = np.asarray([atom.coord for atom in atoms], dtype=float)
        centroid = coordinates.mean(axis=0)
        if rotation_axis is not None:
            rotation = _rotation_matrix(rotation_axis, angle)
            coordinates = (coordinates - centroid) @ rotation.T + centroid
        coordinates += np.asarray(translation, dtype=float)
        for atom, coordinate in zip(atoms, coordinates):
            atom.coord = coordinate

        path = destination / f"{identifier}__{pose_id}.pdb"
        writer = PDBIO()
        writer.set_structure(structure)
        writer.save(str(path))
        results.append(
            PosePerturbation(
                pose_id=pose_id,
                path=path,
                translation_a=translation,
                rotation_axis=rotation_axis,
                rotation_degrees=angle,
            )
        )
    return tuple(results)


def _maximum(values: list[float | None]) -> float | None:
    return max(values) if values and all(value is not None for value in values) else None


def _maximum_delta(values: list[float | None], reference: float | None) -> float | None:
    if reference is None or not values or any(value is None for value in values):
        return None
    return max(abs(float(value) - reference) for value in values)


def assess_local_pose_robustness(
    candidate_pdb: str | Path,
    *,
    reference_pdb: str | Path,
    output_dir: str | Path,
    variant_id: str | None = None,
    translation_distance_a: float = 0.35,
    rotation_degrees: float = 2.5,
) -> PoseRobustnessAssessment:
    """Classify sensitivity of catalytic geometry to the fixed local pose set."""
    path = Path(candidate_pdb)
    identifier = variant_id or path.stem
    method = (
        "Fixed rigid-body translations and rotations of the resolved 23WN Aβ34–41 pose "
        "with DP622 and Zn held unchanged."
    )
    limit = (
        "This is a deterministic local sensitivity analysis, not a conformational "
        "ensemble of full-length Aβ42."
    )
    try:
        perturbations = generate_local_pose_perturbations(
            path,
            output_dir,
            variant_id=identifier,
            translation_distance_a=translation_distance_a,
            rotation_degrees=rotation_degrees,
        )
    except (ValueError, OSError) as exc:
        return PoseRobustnessAssessment(
            variant_id=identifier,
            classification="indeterminate",
            samples=(),
            thresholds=dict(_THRESHOLDS),
            max_substrate_rmsd_a=None,
            max_substrate_pose_drift_a=None,
            max_zn_scissile_distance_delta_a=None,
            max_e96_scissile_distance_delta_a=None,
            max_h172_scissile_distance_delta_a=None,
            method=method,
            interpretation_limit=limit,
            warnings=(str(exc),),
        )

    samples = tuple(
        PoseGeometrySample(
            pose_id=perturbation.pose_id,
            path=perturbation.path,
            geometry=measure_geometry(
                perturbation.path,
                reference_pdb=reference_pdb,
                variant_id=f"{identifier}:{perturbation.pose_id}",
            ),
        )
        for perturbation in perturbations
    )
    reference_geometry = measure_geometry(
        reference_pdb, reference_pdb=reference_pdb, variant_id="pose-reference"
    )
    geometries = [sample.geometry for sample in samples]
    max_substrate_rmsd = _maximum([item.substrate_rmsd_a for item in geometries])
    max_pose_drift = _maximum([item.substrate_pose_drift_a for item in geometries])
    max_zn_delta = _maximum_delta(
        [item.zn_scissile_oxygen_distance_a for item in geometries],
        reference_geometry.zn_scissile_oxygen_distance_a,
    )
    max_e96_delta = _maximum_delta(
        [item.e96_to_scissile_carbonyl_distance_a for item in geometries],
        reference_geometry.e96_to_scissile_carbonyl_distance_a,
    )
    max_h172_delta = _maximum_delta(
        [item.h172_to_scissile_oxygen_distance_a for item in geometries],
        reference_geometry.h172_to_scissile_oxygen_distance_a,
    )
    observed = {
        "substrate_rmsd_a": max_substrate_rmsd,
        "substrate_pose_drift_a": max_pose_drift,
        "zn_scissile_distance_delta_a": max_zn_delta,
        "e96_scissile_distance_delta_a": max_e96_delta,
        "h172_scissile_distance_delta_a": max_h172_delta,
    }

    incomplete = (
        not reference_geometry.geometry_complete
        or any(not geometry.geometry_complete for geometry in geometries)
        or any(value is None or not np.isfinite(value) for value in observed.values())
    )
    if incomplete:
        classification = "indeterminate"
        warnings = (
            "One or more poses have incomplete catalytic geometry; local robustness "
            "cannot be adjudicated.",
        )
    else:
        exceeded = [
            name
            for name, value in observed.items()
            if float(value) > _THRESHOLDS[name]
        ]
        if exceeded:
            classification = "pose-sensitive"
            warnings = (
                "Local perturbations exceed the engineering sensitivity threshold(s): "
                + ", ".join(exceeded)
                + ".",
            )
        else:
            classification = "robust"
            warnings = ()

    return PoseRobustnessAssessment(
        variant_id=identifier,
        classification=classification,
        samples=samples,
        thresholds=dict(_THRESHOLDS),
        max_substrate_rmsd_a=max_substrate_rmsd,
        max_substrate_pose_drift_a=max_pose_drift,
        max_zn_scissile_distance_delta_a=max_zn_delta,
        max_e96_scissile_distance_delta_a=max_e96_delta,
        max_h172_scissile_distance_delta_a=max_h172_delta,
        method=method,
        interpretation_limit=limit,
        warnings=warnings,
    )
