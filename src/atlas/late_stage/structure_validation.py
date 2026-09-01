"""Orthogonal sequence-to-structure validation through a real external predictor.

The adapter measures structural consistency with the active-like reference.  It does
not model Zn, Aβ, catalysis, catalytic rates, or experimental activity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib.metadata import PackageNotFoundError, version as package_version
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
from Bio.PDB import PDBParser
from Bio.SVDSuperimposer import SVDSuperimposer


_STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_CATALYTIC_SITE_RESIDUES = frozenset({91, 95, 96, 99, 122, 126, 172})
_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT"})
_RECOVERY_RMSD_THRESHOLD_A = 2.0
_CLAIM_LIMITS = {
    "activity_prediction": False,
    "metal_or_substrate_modeled": False,
    "interpretation": (
        "Comparative fold and catalytic-residue constellation consistency only; "
        "not evidence of Zn coordination, Aβ binding, catalytic turnover, or efficacy."
    ),
}


class StructureValidationStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


@dataclass(frozen=True)
class StructureValidationResult:
    candidate_id: str
    status: StructureValidationStatus
    output_dir: Path
    predicted_structure: Path | None
    tool_version: str | None
    confidence: Mapping[str, float]
    alignment: Mapping[str, Any] | None
    catalytic_site_recovery: Mapping[str, Any] | None
    reason: str | None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _command_name(command: Sequence[str]) -> str:
    for part in reversed(command):
        name = Path(part).name
        if name and not name.startswith("-"):
            return name
    return Path(command[0]).name


def _resolve_command(command: Sequence[str] | None) -> tuple[str, ...] | None:
    if command is None:
        executable = shutil.which("colabfold_batch")
        return None if executable is None else (executable,)
    resolved = tuple(str(part) for part in command)
    if not resolved:
        return None
    executable = resolved[0]
    located = shutil.which(executable)
    if located is None and not (Path(executable).is_file() and Path(executable).stat().st_mode):
        return None
    return resolved


def _tool_version(command: Sequence[str], timeout_s: float) -> str | None:
    try:
        completed = subprocess.run(
            [*command, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=min(timeout_s, 30.0),
        )
    except (OSError, subprocess.SubprocessError):
        completed = None
    if completed is not None:
        text = completed.stdout.strip() or completed.stderr.strip()
        if completed.returncode == 0 and text:
            return text.splitlines()[0]
    try:
        return package_version("colabfold")
    except PackageNotFoundError:
        return None


def _prediction_path(prediction_dir: Path) -> Path | None:
    patterns = (
        "*rank_001*.pdb",
        "*rank_1*.pdb",
        "*ranked_0*.pdb",
        "*.pdb",
    )
    for pattern in patterns:
        matches = sorted(prediction_dir.rglob(pattern))
        if matches:
            return matches[0]
    return None


def _confidence(prediction_dir: Path, structure_path: Path) -> dict[str, float]:
    score_files = sorted(prediction_dir.rglob("*scores*.json"))
    if not score_files:
        score_files = sorted(prediction_dir.rglob("*.json"))
    for score_file in score_files:
        try:
            scores = json.loads(score_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        confidence: dict[str, float] = {}
        plddt = scores.get("plddt")
        if isinstance(plddt, list) and plddt:
            confidence["mean_plddt"] = float(np.mean(np.asarray(plddt, dtype=float)))
        for name in ("ptm", "iptm", "ranking_confidence"):
            value = scores.get(name)
            if isinstance(value, (int, float)):
                confidence[name] = float(value)
        if confidence:
            return confidence

    structure = PDBParser(QUIET=True).get_structure("prediction", structure_path)
    ca_b_factors = [float(atom.bfactor) for atom in structure.get_atoms() if atom.id == "CA"]
    return {"mean_plddt": float(np.mean(ca_b_factors))} if ca_b_factors else {}


def _chain(structure, preferred: str):
    model = next(structure.get_models())
    if preferred in model:
        return model[preferred]
    chains = list(model.get_chains())
    if not chains:
        raise ValueError("predicted structure has no chains")
    return chains[0]


def _ca_atoms(chain) -> dict[int, Any]:
    return {
        residue.id[1]: residue["CA"]
        for residue in chain
        if residue.id[0] == " " and "CA" in residue
    }


def _rmsd(fixed: np.ndarray, moving: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((moving - fixed) ** 2, axis=1))))


def _structural_metrics(
    reference_pdb: Path, predicted_pdb: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    parser = PDBParser(QUIET=True)
    reference = parser.get_structure("reference", reference_pdb)
    predicted = parser.get_structure("prediction", predicted_pdb)
    reference_chain = _chain(reference, "A")
    predicted_chain = _chain(predicted, "A")
    reference_ca = _ca_atoms(reference_chain)
    predicted_ca = _ca_atoms(predicted_chain)
    common = sorted(set(reference_ca) & set(predicted_ca))
    if len(common) < 3:
        raise ValueError("fewer than three residue-number-matched CA atoms are available")

    fixed = np.asarray([reference_ca[number].coord for number in common], dtype=float)
    moving = np.asarray([predicted_ca[number].coord for number in common], dtype=float)
    superimposer = SVDSuperimposer()
    superimposer.set(fixed, moving)
    superimposer.run()
    rotation, translation = superimposer.get_rotran()
    transformed = np.dot(moving, rotation) + translation
    alignment = {
        "reference_chain": reference_chain.id,
        "predicted_chain": predicted_chain.id,
        "aligned_ca_count": len(common),
        "ca_rmsd_a": _rmsd(fixed, transformed),
    }

    fixed_sidechain: list[np.ndarray] = []
    moving_sidechain: list[np.ndarray] = []
    matched_residues: list[int] = []
    for number in sorted(_CATALYTIC_SITE_RESIDUES):
        if not reference_chain.has_id(number) or not predicted_chain.has_id(number):
            continue
        reference_residue = reference_chain[number]
        predicted_residue = predicted_chain[number]
        names = sorted(
            {
                atom.id
                for atom in reference_residue
                if atom.element != "H" and atom.id not in _BACKBONE_ATOMS
            }
            & {
                atom.id
                for atom in predicted_residue
                if atom.element != "H" and atom.id not in _BACKBONE_ATOMS
            }
        )
        if names:
            matched_residues.append(number)
        for name in names:
            fixed_sidechain.append(np.asarray(reference_residue[name].coord, dtype=float))
            moving_sidechain.append(
                np.dot(np.asarray(predicted_residue[name].coord, dtype=float), rotation)
                + translation
            )

    sidechain_rmsd = (
        _rmsd(np.asarray(fixed_sidechain), np.asarray(moving_sidechain))
        if fixed_sidechain
        else None
    )
    recovery = {
        "residue_numbers": sorted(_CATALYTIC_SITE_RESIDUES),
        "matched_residue_numbers": matched_residues,
        "matched_sidechain_heavy_atoms": len(fixed_sidechain),
        "sidechain_heavy_atom_rmsd_a": sidechain_rmsd,
        "engineering_threshold_a": _RECOVERY_RMSD_THRESHOLD_A,
        "recovered_by_engineering_threshold": (
            sidechain_rmsd is not None
            and len(matched_residues) == len(_CATALYTIC_SITE_RESIDUES)
            and sidechain_rmsd <= _RECOVERY_RMSD_THRESHOLD_A
        ),
        "interpretation": (
            "Recovery is a reference-relative structural engineering criterion only; "
            "the predictor does not establish catalytic activity."
        ),
    }
    return alignment, recovery


def _report_payload(
    *,
    candidate_id: str,
    sequence: str,
    fasta_path: Path,
    command: Sequence[str] | None,
    version: str | None,
    status: StructureValidationStatus,
    predicted_structure: Path | None,
    confidence: Mapping[str, float],
    alignment: Mapping[str, Any] | None,
    recovery: Mapping[str, Any] | None,
    reason: str | None,
    process: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "status": status.value,
        "reason": reason,
        "input": {
            "candidate_id": candidate_id,
            "sequence": sequence,
            "sequence_length": len(sequence),
            "fasta": str(fasta_path),
        },
        "tool": {
            "name": None if command is None else _command_name(command),
            "version": version,
            "command": None if command is None else list(command),
        },
        "process": None if process is None else dict(process),
        "predicted_structure": (
            None if predicted_structure is None else str(predicted_structure)
        ),
        "confidence": dict(confidence),
        "alignment": None if alignment is None else dict(alignment),
        "catalytic_site_recovery": None if recovery is None else dict(recovery),
        "claim_limits": dict(_CLAIM_LIMITS),
    }


def validate_structure_prediction(
    *,
    candidate_id: str,
    sequence: str,
    reference_pdb: str | Path,
    output_dir: str | Path,
    predictor_command: Sequence[str] | None = None,
    timeout_s: float = 7200.0,
) -> StructureValidationResult:
    """Run a ColabFold-class command and persist conservative validation evidence.

    The command contract is ``COMMAND input.fasta prediction_directory``.  With no
    configured command, ``colabfold_batch`` is discovered on ``PATH``.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    normalized_sequence = sequence.strip().upper()
    fasta_path = output / f"{candidate_id}.fasta"
    fasta_path.write_text(f">{candidate_id}\n{normalized_sequence}\n", encoding="ascii")
    requested_command = (
        None
        if predictor_command is None
        else tuple(str(part) for part in predictor_command)
    )
    command = _resolve_command(predictor_command)
    report_command = command or requested_command or ("colabfold_batch",)

    def finish(
        status: StructureValidationStatus,
        *,
        version: str | None = None,
        predicted_structure: Path | None = None,
        confidence: Mapping[str, float] | None = None,
        alignment: Mapping[str, Any] | None = None,
        recovery: Mapping[str, Any] | None = None,
        reason: str | None = None,
        process: Mapping[str, Any] | None = None,
    ) -> StructureValidationResult:
        normalized_confidence = {} if confidence is None else dict(confidence)
        report = _report_payload(
            candidate_id=candidate_id,
            sequence=normalized_sequence,
            fasta_path=fasta_path,
            command=report_command,
            version=version,
            status=status,
            predicted_structure=predicted_structure,
            confidence=normalized_confidence,
            alignment=alignment,
            recovery=recovery,
            reason=reason,
            process=process,
        )
        _write_json(output / "structure_validation.json", report)
        return StructureValidationResult(
            candidate_id=candidate_id,
            status=status,
            output_dir=output,
            predicted_structure=predicted_structure,
            tool_version=version,
            confidence=normalized_confidence,
            alignment=alignment,
            catalytic_site_recovery=recovery,
            reason=reason,
        )

    if not normalized_sequence or any(
        residue not in _STANDARD_AMINO_ACIDS for residue in normalized_sequence
    ):
        return finish(
            StructureValidationStatus.INVALID,
            reason="input sequence must contain only standard amino-acid letters",
        )
    if command is None:
        return finish(
            StructureValidationStatus.UNAVAILABLE,
            reason="configured predictor is not executable and colabfold_batch is unavailable",
        )

    version = _tool_version(command, timeout_s)
    prediction_dir = output / "prediction"
    prediction_dir.mkdir(exist_ok=True)
    try:
        completed = subprocess.run(
            [*command, str(fasta_path), str(prediction_dir)],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except FileNotFoundError as exc:
        return finish(
            StructureValidationStatus.UNAVAILABLE,
            version=version,
            reason=str(exc),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return finish(
            StructureValidationStatus.INVALID,
            version=version,
            reason=f"predictor execution failed: {exc}",
        )

    (output / "predictor.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output / "predictor.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    process = {"returncode": completed.returncode}
    if completed.returncode != 0:
        return finish(
            StructureValidationStatus.INVALID,
            version=version,
            reason=f"predictor exited with status {completed.returncode}",
            process=process,
        )

    predicted = _prediction_path(prediction_dir)
    if predicted is None:
        return finish(
            StructureValidationStatus.INVALID,
            version=version,
            reason="predictor completed without producing a PDB structure",
            process=process,
        )
    try:
        confidence = _confidence(prediction_dir, predicted)
        alignment, recovery = _structural_metrics(Path(reference_pdb), predicted)
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        return finish(
            StructureValidationStatus.INVALID,
            version=version,
            predicted_structure=predicted,
            reason=f"prediction could not be validated: {exc}",
            process=process,
        )

    _write_json(output / "confidence.json", confidence)
    _write_json(output / "alignment.json", alignment)
    _write_json(output / "catalytic_site_recovery.json", recovery)
    return finish(
        StructureValidationStatus.AVAILABLE,
        version=version,
        predicted_structure=predicted,
        confidence=confidence,
        alignment=alignment,
        recovery=recovery,
        process=process,
    )
