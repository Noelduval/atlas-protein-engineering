"""Deterministic sequence/structure developability-risk heuristics.

The screen reports transparent, inexpensive warning signals.  It is not a
predictor of safety, exposure, efficacy, or clinical suitability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import Bio
import numpy as np
from Bio.PDB import PDBParser, ShrakeRupley
from Bio.SeqUtils import seq1
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from atlas.adaptive.models import STANDARD_AMINO_ACIDS
from atlas.design.design_space import MAXIMUM_RESIDUE_SASA_A2


_METHOD_VERSION = "atlas-developability-heuristics-v1"
_HYDROPHOBIC = frozenset("AILMFWVY")
_CHARGED = frozenset("DEKRH")
_WINDOW_SIZE = 7
_HYDROPHOBIC_WINDOW_THRESHOLD = 5
_EXPOSED_RELATIVE_SASA = 0.5
_BURIED_RELATIVE_SASA = 0.2
_SURFACE_CONTACT_CUTOFF_A = 4.5
_SURFACE_PATCH_SIZE_THRESHOLD = 3
_CHEMICAL_MOTIFS = {
    "deamidation_prone": re.compile(r"N[GS]"),
    "aspartate_isomerization_prone": re.compile(r"D[GS]"),
}
_EXCLUDED_CLAIMS = [
    "safety",
    "blood-brain barrier penetration",
    "pharmacokinetics",
    "immunogenicity",
    "clinical developability",
]


@dataclass(frozen=True)
class DevelopabilityFlag:
    """One transparent warning signal emitted by the screen."""

    code: str
    source: str
    severity: str
    detail: str
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DevelopabilityResult:
    """Persisted result of the bounded heuristic screen."""

    candidate_id: str
    status: str
    outcome: str
    uncertainty: Mapping[str, Any]
    sequence_metrics: Mapping[str, Any]
    structure_metrics: Mapping[str, Any] | None
    flags: tuple[DevelopabilityFlag, ...]
    provenance: Mapping[str, Any]
    artifact_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status,
            "outcome": self.outcome,
            "uncertainty": dict(self.uncertainty),
            "sequence_metrics": dict(self.sequence_metrics),
            "structure_metrics": (
                None if self.structure_metrics is None else dict(self.structure_metrics)
            ),
            "flags": [flag.to_dict() for flag in self.flags],
            "provenance": dict(self.provenance),
        }


class _StructureSequenceMismatch(ValueError):
    pass


def _validate_sequence(sequence: str, *, label: str) -> None:
    if not sequence or any(residue not in STANDARD_AMINO_ACIDS for residue in sequence):
        raise ValueError(f"{label} must contain only standard uppercase amino acids")


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _motif_positions(sequence: str) -> dict[str, list[int]]:
    motifs: dict[str, list[int]] = {}
    for name, pattern in _CHEMICAL_MOTIFS.items():
        positions = [match.start() + 1 for match in pattern.finditer(sequence)]
        if positions:
            motifs[name] = positions
    return motifs


def _added_motifs(
    candidate_motifs: Mapping[str, list[int]],
    reference_motifs: Mapping[str, list[int]] | None,
) -> dict[str, list[int]]:
    if reference_motifs is None:
        return dict(candidate_motifs)
    added: dict[str, list[int]] = {}
    for name, positions in candidate_motifs.items():
        new_positions = sorted(set(positions) - set(reference_motifs.get(name, ())))
        if new_positions:
            added[name] = new_positions
    return added


def _sequence_metrics(
    sequence: str, reference_sequence: str | None
) -> tuple[dict[str, Any], list[DevelopabilityFlag]]:
    analysis = ProteinAnalysis(sequence)
    window_size = min(_WINDOW_SIZE, len(sequence))
    hydrophobic_counts = [
        sum(residue in _HYDROPHOBIC for residue in sequence[start : start + window_size])
        for start in range(len(sequence) - window_size + 1)
    ]
    max_hydrophobic = max(hydrophobic_counts)
    hydrophobic_windows = [
        {"start": start + 1, "end": start + window_size}
        for start, count in enumerate(hydrophobic_counts)
        if window_size == _WINDOW_SIZE and count >= _HYDROPHOBIC_WINDOW_THRESHOLD
    ]
    motifs = _motif_positions(sequence)
    reference_motifs = (
        None if reference_sequence is None else _motif_positions(reference_sequence)
    )
    added_motifs = _added_motifs(motifs, reference_motifs)
    introduced_cysteines = (
        [index + 1 for index, residue in enumerate(sequence) if residue == "C"]
        if reference_sequence is None
        else [
            index + 1
            for index, (residue, reference) in enumerate(
                zip(sequence, reference_sequence, strict=True)
            )
            if residue == "C" and reference != "C"
        ]
    )
    metrics: dict[str, Any] = {
        "length": len(sequence),
        "molecular_weight_da": _rounded(analysis.molecular_weight()),
        "isoelectric_point": _rounded(analysis.isoelectric_point()),
        "net_charge_at_ph_7": _rounded(analysis.charge_at_pH(7.0)),
        "gravy": _rounded(analysis.gravy()),
        "aromaticity": _rounded(analysis.aromaticity()),
        "instability_index": _rounded(analysis.instability_index()),
        "max_hydrophobic_residues_in_7mer": max_hydrophobic,
        "hydrophobic_7mer_windows": hydrophobic_windows,
        "chemical_liability_motifs": motifs,
        "added_chemical_liability_motifs": added_motifs,
        "cysteine_count": sequence.count("C"),
        "introduced_cysteine_positions": introduced_cysteines,
    }
    flags: list[DevelopabilityFlag] = []
    if hydrophobic_windows:
        flags.append(
            DevelopabilityFlag(
                code="hydrophobic_sequence_patch",
                source="sequence",
                severity="moderate",
                detail=(
                    "At least five hydrophobic residues occur in a seven-residue "
                    "window; this is a nonspecific aggregation/solubility warning."
                ),
                evidence={"windows": hydrophobic_windows},
            )
        )
    if introduced_cysteines:
        flags.append(
            DevelopabilityFlag(
                code="cysteine_requires_review",
                source="sequence",
                severity="moderate",
                detail=(
                    "Introduced cysteine chemistry is unresolved without experimental "
                    "oxidation-state and disulfide-context evidence."
                ),
                evidence={"positions": introduced_cysteines},
            )
        )
    for motif, positions in sorted(added_motifs.items()):
        flags.append(
            DevelopabilityFlag(
                code=f"added_{motif}",
                source="sequence",
                severity="low",
                detail=(
                    "Candidate introduces a sequence motif associated with a possible "
                    "chemical-liability mechanism; experimental confirmation is required."
                ),
                evidence={"positions": positions},
            )
        )
    return metrics, flags


def _heavy_coordinates(residue: Any) -> np.ndarray:
    return np.asarray(
        [atom.coord for atom in residue if (atom.element or "").upper() != "H"],
        dtype=float,
    )


def _connected_surface_patches(residues: list[Any]) -> list[list[Any]]:
    adjacency = {index: set() for index in range(len(residues))}
    for left_index, left in enumerate(residues):
        left_coordinates = _heavy_coordinates(left)
        for right_index in range(left_index + 1, len(residues)):
            right_coordinates = _heavy_coordinates(residues[right_index])
            distance = float(
                np.linalg.norm(
                    left_coordinates[:, None, :] - right_coordinates[None, :, :], axis=2
                ).min()
            )
            if distance <= _SURFACE_CONTACT_CUTOFF_A:
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)
    patches: list[list[Any]] = []
    unseen = set(adjacency)
    while unseen:
        root = min(unseen)
        stack = [root]
        component: list[int] = []
        unseen.remove(root)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        patches.append([residues[index] for index in sorted(component)])
    return patches


def _residue_label(residue: Any) -> str:
    return f"{residue.get_parent().id}:{int(residue.id[1])}"


def _structure_screen(
    structure_path: Path,
    sequence: str,
    reference_sequence: str | None,
    chain_id: str,
) -> tuple[dict[str, Any], list[DevelopabilityFlag]]:
    structure = PDBParser(QUIET=True).get_structure("developability", structure_path)
    model = next(structure.get_models())
    if chain_id not in model:
        raise ValueError(f"Structure does not contain requested chain {chain_id!r}")
    residues = [
        residue
        for residue in model[chain_id]
        if residue.id[0] == " " and residue.resname != "HOH"
    ]
    if not residues:
        raise ValueError(f"Structure chain {chain_id!r} has no standard residues")
    structure_sequence = "".join(
        seq1(residue.resname, custom_map={"MSE": "M"}) for residue in residues
    )
    if "X" in structure_sequence or structure_sequence != sequence:
        raise _StructureSequenceMismatch(
            f"Chain {chain_id} sequence does not exactly match candidate sequence"
        )

    ShrakeRupley(probe_radius=1.4, n_points=100).compute(structure, level="R")
    relative_sasa: list[float] = []
    for residue, amino_acid in zip(residues, sequence, strict=True):
        sasa = float(getattr(residue, "sasa"))
        relative_sasa.append(min(1.0, sasa / MAXIMUM_RESIDUE_SASA_A2[amino_acid]))
    exposed_hydrophobic = [
        residue
        for residue, amino_acid, exposure in zip(
            residues, sequence, relative_sasa, strict=True
        )
        if amino_acid in _HYDROPHOBIC and exposure >= _EXPOSED_RELATIVE_SASA
    ]
    patches = _connected_surface_patches(exposed_hydrophobic)
    largest_patch = max(patches, key=len, default=[])
    buried_charged = [
        _residue_label(residue)
        for residue, amino_acid, exposure in zip(
            residues, sequence, relative_sasa, strict=True
        )
        if amino_acid in _CHARGED and exposure < _BURIED_RELATIVE_SASA
    ]
    introduced_exposed_hydrophobic: list[str] = []
    introduced_buried_charge: list[str] = []
    if reference_sequence is not None:
        for residue, amino_acid, reference, exposure in zip(
            residues, sequence, reference_sequence, relative_sasa, strict=True
        ):
            if (
                amino_acid in _HYDROPHOBIC
                and reference not in _HYDROPHOBIC
                and exposure >= _EXPOSED_RELATIVE_SASA
            ):
                introduced_exposed_hydrophobic.append(_residue_label(residue))
            if (
                amino_acid in _CHARGED
                and reference not in _CHARGED
                and exposure < _BURIED_RELATIVE_SASA
            ):
                introduced_buried_charge.append(_residue_label(residue))

    metrics: dict[str, Any] = {
        "chain_id": chain_id,
        "resolved_residue_count": len(residues),
        "sequence_coverage_fraction": 1.0,
        "mean_relative_sasa": _rounded(float(np.mean(relative_sasa))),
        "exposed_hydrophobic_residue_count": len(exposed_hydrophobic),
        "largest_exposed_hydrophobic_patch": len(largest_patch),
        "buried_charged_residues": buried_charged,
        "introduced_exposed_hydrophobic_residues": introduced_exposed_hydrophobic,
        "introduced_buried_charge_residues": introduced_buried_charge,
    }
    flags: list[DevelopabilityFlag] = []
    if len(largest_patch) >= _SURFACE_PATCH_SIZE_THRESHOLD:
        flags.append(
            DevelopabilityFlag(
                code="exposed_hydrophobic_surface_patch",
                source="structure",
                severity="moderate",
                detail=(
                    "A connected solvent-exposed hydrophobic patch is present in this "
                    "static structure model."
                ),
                evidence={
                    "residues": [_residue_label(residue) for residue in largest_patch],
                    "contact_cutoff_a": _SURFACE_CONTACT_CUTOFF_A,
                },
            )
        )
    if introduced_exposed_hydrophobic:
        flags.append(
            DevelopabilityFlag(
                code="introduced_exposed_hydrophobe",
                source="structure",
                severity="moderate",
                detail="Candidate introduces hydrophobic residues at exposed model positions.",
                evidence={"residues": introduced_exposed_hydrophobic},
            )
        )
    if introduced_buried_charge:
        flags.append(
            DevelopabilityFlag(
                code="introduced_buried_charge",
                source="structure",
                severity="moderate",
                detail=(
                    "Candidate introduces charged residues at buried model positions; "
                    "local compensation was not inferred by this screen."
                ),
                evidence={"residues": introduced_buried_charge},
            )
        )
    return metrics, flags


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def screen_developability(
    *,
    candidate_id: str,
    sequence: str,
    output_dir: str | Path,
    structure_path: str | Path | None = None,
    reference_sequence: str | None = None,
    chain_id: str = "A",
) -> DevelopabilityResult:
    """Run and persist a bounded developability-risk screen.

    Missing or invalid structure evidence is explicitly indeterminate and can
    never yield a favorable screening outcome.
    """

    _validate_sequence(sequence, label="sequence")
    if reference_sequence is not None:
        _validate_sequence(reference_sequence, label="reference_sequence")
        if len(reference_sequence) != len(sequence):
            raise ValueError("reference_sequence must have the candidate sequence length")
    if not candidate_id.strip():
        raise ValueError("candidate_id is required")

    sequence_metrics, flags = _sequence_metrics(sequence, reference_sequence)
    structure_metrics: dict[str, Any] | None = None
    resolved_structure_path = None if structure_path is None else Path(structure_path)
    structure_sha256: str | None = None
    structure_state = "unavailable"
    if resolved_structure_path is None or not resolved_structure_path.is_file():
        flags.append(
            DevelopabilityFlag(
                code="structure_evidence_unavailable",
                source="structure",
                severity="high",
                detail=(
                    "Candidate structure evidence is unavailable; burial and surface-patch "
                    "risks remain unresolved."
                ),
                evidence={
                    "requested_path": (
                        None
                        if resolved_structure_path is None
                        else str(resolved_structure_path)
                    )
                },
            )
        )
    else:
        structure_sha256 = _sha256_file(resolved_structure_path)
        try:
            structure_metrics, structure_flags = _structure_screen(
                resolved_structure_path, sequence, reference_sequence, chain_id
            )
        except _StructureSequenceMismatch as exc:
            structure_state = "invalid"
            flags.append(
                DevelopabilityFlag(
                    code="structure_sequence_mismatch",
                    source="structure",
                    severity="high",
                    detail=str(exc),
                    evidence={"chain_id": chain_id},
                )
            )
        except Exception as exc:
            structure_state = "invalid"
            flags.append(
                DevelopabilityFlag(
                    code="structure_evidence_invalid",
                    source="structure",
                    severity="high",
                    detail=f"Structure screening failed: {type(exc).__name__}: {exc}",
                    evidence={"chain_id": chain_id},
                )
            )
        else:
            structure_state = "available"
            flags.extend(structure_flags)

    if structure_state != "available":
        status = "indeterminate"
        outcome = "review_required"
        uncertainty: dict[str, Any] = {
            "level": "high",
            "reasons": [
                "Candidate-specific burial and surface context are unavailable or invalid.",
                "Sequence heuristics alone do not resolve structure-dependent risks.",
            ],
        }
    else:
        status = "completed"
        outcome = "concerns_identified" if flags else "no_major_flags_in_screen"
        uncertainty = {
            "level": "moderate",
            "reasons": [
                "Rules use a single static structure model and deterministic thresholds.",
                "No experimental formulation or biochemical measurements are included.",
            ],
        }

    provenance: dict[str, Any] = {
        "method": "deterministic_sequence_structure_heuristics",
        "method_version": _METHOD_VERSION,
        "biopython_version": Bio.__version__,
        "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        "reference_sequence_sha256": (
            None
            if reference_sequence is None
            else hashlib.sha256(reference_sequence.encode("ascii")).hexdigest()
        ),
        "structure_path": (
            None if resolved_structure_path is None else str(resolved_structure_path)
        ),
        "structure_sha256": structure_sha256,
        "structure_state": structure_state,
        "chain_id": chain_id,
        "thresholds": {
            "hydrophobic_window_size": _WINDOW_SIZE,
            "hydrophobic_residues_per_window": _HYDROPHOBIC_WINDOW_THRESHOLD,
            "exposed_relative_sasa": _EXPOSED_RELATIVE_SASA,
            "buried_relative_sasa": _BURIED_RELATIVE_SASA,
            "surface_contact_cutoff_a": _SURFACE_CONTACT_CUTOFF_A,
            "surface_patch_residue_count": _SURFACE_PATCH_SIZE_THRESHOLD,
        },
        "excluded_claims": _EXCLUDED_CLAIMS,
        "claim_boundary": (
            "Heuristic sequence/static-structure risk flags only; absence of a flag is not "
            "evidence of clinical suitability."
        ),
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifact_path = destination / "developability.json"
    result = DevelopabilityResult(
        candidate_id=candidate_id,
        status=status,
        outcome=outcome,
        uncertainty=uncertainty,
        sequence_metrics=sequence_metrics,
        structure_metrics=structure_metrics,
        flags=tuple(flags),
        provenance=provenance,
        artifact_path=artifact_path,
    )
    artifact_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return result
