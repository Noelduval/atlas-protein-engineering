"""Hard chemistry and metal-site checks for modeled DP622/Aβ complexes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from Bio.PDB import NeighborSearch, PDBParser

from atlas.adaptive.models import HardViolation
from atlas.geometry.selectors import (
    E122_OE1,
    E122_OE2,
    H95_NE2,
    H99_NE2,
    SCISSILE_OXYGEN,
    ZINC,
    select_atom,
)


def _atom_label(atom) -> str:
    residue = atom.get_parent()
    chain = residue.get_parent()
    return f"{chain.id}:{residue.id[1]}:{atom.id}"


def detect_catastrophic_clashes(
    pdb_path: str | Path, *, cutoff_a: float = 0.65
) -> tuple[str, ...]:
    """Return impossible heavy-atom overlaps between distinct residues."""
    structure = PDBParser(QUIET=True).get_structure("chemistry", pdb_path)
    atoms = [
        atom
        for atom in structure.get_atoms()
        if (atom.element or "").upper() != "H"
    ]
    pairs = NeighborSearch(atoms).search_all(cutoff_a, level="A")
    clashes: set[str] = set()
    for left, right in pairs:
        if left.get_parent() is right.get_parent():
            continue
        left_label, right_label = sorted((_atom_label(left), _atom_label(right)))
        distance = float(np.linalg.norm(left.coord - right.coord))
        clashes.add(f"{left_label}--{right_label}:{distance:.4f}A")
    return tuple(sorted(clashes))


def validate_required_zinc_coordination(
    pdb_path: str | Path, candidate_id: str
) -> tuple[HardViolation, ...]:
    """Validate presence and a permissive physical range for required Zn contacts."""
    structure = PDBParser(QUIET=True).get_structure("zinc", pdb_path)
    try:
        zinc = select_atom(structure, ZINC)
        distances = {
            "H95_NE2": float(zinc - select_atom(structure, H95_NE2)),
            "H99_NE2": float(zinc - select_atom(structure, H99_NE2)),
            "E122_O": min(
                float(zinc - select_atom(structure, E122_OE1)),
                float(zinc - select_atom(structure, E122_OE2)),
            ),
            "B38_scissile_O": float(zinc - select_atom(structure, SCISSILE_OXYGEN)),
        }
    except Exception as exc:
        return (
            HardViolation(
                candidate_id,
                "required_zinc_atom_missing",
                f"Required Zn-site atom selection failed: {type(exc).__name__}: {exc}",
            ),
        )
    violations = []
    for label, distance in distances.items():
        if not 1.5 <= distance <= 3.2:
            violations.append(
                HardViolation(
                    candidate_id,
                    "required_zinc_coordination_lost",
                    f"{label} distance {distance:.3f} Å is outside 1.5–3.2 Å.",
                )
            )
    return tuple(violations)

