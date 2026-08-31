"""Open, reproducible construction of arbitrary legal DP622 mutant complexes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

from atlas.adaptive.models import CandidateRecord, HardViolation
from atlas.structure.chemistry import (
    detect_catastrophic_clashes,
    validate_unintended_zinc_coordination,
    validate_required_zinc_coordination,
)


PDBFIXER_COMMIT = "5e658a4fb8d2b90d65ca8408c845cbf104ce7099"
HARD_PROTECTED_POSITIONS = frozenset({95, 96, 99, 122, 172})
ONE_TO_THREE = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}


class MutantConstructionError(RuntimeError):
    """The candidate cannot be represented without violating hard constraints."""


@dataclass(frozen=True)
class MutantComplexResult:
    candidate_id: str
    raw_pdb: Path
    relaxed_pdb: Path | None
    atom_provenance_csv: Path
    provenance_json: Path
    retained_atom_count: int
    modeled_atom_count: int
    removed_atom_count: int
    hard_violations: tuple[HardViolation, ...]
    warnings: tuple[str, ...]


def _load_atoms(pdb_path: str | Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    structure = PDBParser(QUIET=True).get_structure("atoms", pdb_path)
    atoms: dict[tuple[str, int, str], dict[str, Any]] = {}
    for atom in structure.get_atoms():
        residue = atom.get_parent()
        chain = residue.get_parent()
        key = (chain.id, int(residue.id[1]), atom.id)
        atoms[key] = {
            "chain": chain.id,
            "residue": int(residue.id[1]),
            "resname": residue.resname,
            "atom": atom.id,
            "element": atom.element,
            "coordinate": np.asarray(atom.coord, dtype=float),
        }
    return atoms


def _validate_components(pdb_path: Path, candidate: CandidateRecord) -> None:
    structure = PDBParser(QUIET=True).get_structure("mutant", pdb_path)
    model = structure[0]
    if not {"A", "B", "C"} <= {chain.id for chain in model}:
        raise MutantConstructionError("Mutant complex lost protein, substrate, or metal chain")
    if [residue.id[1] for residue in model["A"]] != list(range(1, 216)):
        raise MutantConstructionError("Mutant DP622 chain is not contiguous 1..215")
    if [residue.id[1] for residue in model["B"]] != list(range(34, 42)):
        raise MutantConstructionError("Mutant complex did not retain Aβ residues 34..41")
    zinc_atoms = [atom for atom in model["C"].get_atoms() if atom.element == "ZN"]
    if len(zinc_atoms) != 1:
        raise MutantConstructionError("Mutant complex must retain exactly one Zn atom")
    for mutation in candidate.mutations:
        residue = model["A"][mutation.position]
        if residue.resname != ONE_TO_THREE[mutation.mutant]:
            raise MutantConstructionError(
                f"Mutation {mutation.label} produced residue {residue.resname}"
            )


def build_mutant_complex(
    active_like_pdb: str | Path,
    candidate: CandidateRecord,
    output_dir: str | Path,
    *,
    seed: int,
    relax: bool = False,
    dynamics_config: Any | None = None,
) -> MutantComplexResult:
    """Build missing mutant side-chain atoms while retaining Aβ and Zn.

    PDBFixer is used only for the requested substitutions and their missing
    heavy atoms. Missing residues and fragment-terminal atoms are deliberately
    not built.
    """
    protected = sorted(
        mutation.position
        for mutation in candidate.mutations
        if mutation.position in HARD_PROTECTED_POSITIONS
    )
    if protected:
        raise MutantConstructionError(
            "Candidate mutates protected catalytic position(s): "
            + ", ".join(map(str, protected))
        )
    try:
        from openmm.app import PDBFile
        from pdbfixer import PDBFixer
    except (ImportError, ModuleNotFoundError) as exc:
        raise MutantConstructionError(f"PDBFixer/OpenMM unavailable: {exc}") from exc

    source = Path(active_like_pdb)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    raw_pdb = destination / f"{candidate.candidate_id}_mutant_complex.pdb"
    mutations = [
        f"{ONE_TO_THREE[mutation.wildtype]}-{mutation.position}-"
        f"{ONE_TO_THREE[mutation.mutant]}"
        for mutation in candidate.mutations
    ]
    fixer = PDBFixer(filename=str(source))
    try:
        fixer.applyMutations(mutations, "A")
        fixer.findMissingResidues()
        fixer.missingResidues = {}
        fixer.findMissingAtoms()
        fixer.missingTerminals = {}
        fixer.addMissingAtoms(seed=int(seed))
    except Exception as exc:
        raise MutantConstructionError(
            f"Side-chain construction failed: {type(exc).__name__}: {exc}"
        ) from exc
    with raw_pdb.open("w") as handle:
        PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
    _validate_components(raw_pdb, candidate)

    before, after = _load_atoms(source), _load_atoms(raw_pdb)
    rows: list[dict[str, Any]] = []
    for key, atom in sorted(after.items()):
        if key in before:
            displacement = float(
                np.linalg.norm(atom["coordinate"] - before[key]["coordinate"])
            )
            provenance = "retained_deposited_coordinate"
        else:
            displacement = float("nan")
            provenance = "modeled_sidechain"
        rows.append(
            {
                "chain": atom["chain"],
                "residue": atom["residue"],
                "resname": atom["resname"],
                "atom": atom["atom"],
                "element": atom["element"],
                "source": provenance,
                "coordinate_displacement_a": displacement,
            }
        )
    provenance_csv = destination / "atom_provenance.csv"
    pd.DataFrame(rows).to_csv(provenance_csv, index=False)
    removed = sorted(set(before) - set(after))
    modeled = sum(row["source"] == "modeled_sidechain" for row in rows)
    retained = len(rows) - modeled

    violations = list(validate_required_zinc_coordination(raw_pdb, candidate.candidate_id))
    if candidate.requires_candidate_geometry:
        violations.extend(validate_unintended_zinc_coordination(raw_pdb, candidate))
    clashes = detect_catastrophic_clashes(raw_pdb)
    violations.extend(
        HardViolation(
            candidate.candidate_id,
            "catastrophic_structural_clash",
            clash,
        )
        for clash in clashes
    )
    warnings: list[str] = []
    relaxed_pdb: Path | None = None
    if relax and not violations:
        from atlas.dynamics.models import DynamicsConfig
        from atlas.dynamics.openmm_minimize import minimize_variant

        relaxation = minimize_variant(
            raw_pdb,
            destination / "restrained_relaxation",
            dynamics_config or DynamicsConfig(),
        )
        if relaxation.status == "completed":
            relaxed_pdb = relaxation.output_pdb
        else:
            warnings.append(relaxation.warning)

    provenance_json = destination / "model_provenance.json"
    provenance_payload = {
        "candidate_id": candidate.candidate_id,
        "mutation_set": candidate.mutation_set,
        "model_label": "active_like_inferred",
        "source_pdb": str(source),
        "raw_mutant_pdb": str(raw_pdb),
        "relaxed_pdb": None if relaxed_pdb is None else str(relaxed_pdb),
        "pdbfixer_commit": PDBFIXER_COMMIT,
        "seed": int(seed),
        "retained_input_atoms": retained,
        "modeled_sidechain_atoms": modeled,
        "removed_input_atoms": len(removed),
        "removed_atom_keys": [f"{chain}:{residue}:{atom}" for chain, residue, atom in removed],
        "hard_violations": [
            {"code": violation.code, "detail": violation.detail}
            for violation in violations
        ],
        "warnings": warnings,
        "claim_boundary": (
            "Open, reproducible side-chain model; not an experimentally observed mutant complex."
        ),
    }
    provenance_json.write_text(
        json.dumps(provenance_payload, indent=2, sort_keys=True) + "\n"
    )
    return MutantComplexResult(
        candidate_id=candidate.candidate_id,
        raw_pdb=raw_pdb,
        relaxed_pdb=relaxed_pdb,
        atom_provenance_csv=provenance_csv,
        provenance_json=provenance_json,
        retained_atom_count=retained,
        modeled_atom_count=modeled,
        removed_atom_count=len(removed),
        hard_violations=tuple(violations),
        warnings=tuple(warnings),
    )
