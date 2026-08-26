"""Reproducible, evidence-based classification of the DP622 mutation space."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, ShrakeRupley
from Bio.SeqUtils import seq1

from atlas.design.residue_evidence import (
    CATALYTIC_ROLES,
    DIRECT_ZINC_LIGANDS,
    HARD_PROTECTION_REASONS,
    KNOWN_EXPERIMENTAL_EVIDENCE,
    secondary_structure_by_dp622_position,
)


MAXIMUM_RESIDUE_SASA_A2 = {
    "A": 129.0,
    "C": 167.0,
    "D": 193.0,
    "E": 223.0,
    "F": 240.0,
    "G": 104.0,
    "H": 224.0,
    "I": 197.0,
    "K": 236.0,
    "L": 201.0,
    "M": 224.0,
    "N": 195.0,
    "P": 159.0,
    "Q": 225.0,
    "R": 274.0,
    "S": 155.0,
    "T": 172.0,
    "V": 174.0,
    "W": 285.0,
    "Y": 263.0,
}


class ResidueClass(str, Enum):
    HARD_PROTECTED = "HARD_PROTECTED"
    CONTEXT_SENSITIVE = "CONTEXT_SENSITIVE"
    DESIGNABLE = "DESIGNABLE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


@dataclass(frozen=True)
class ResidueDesignRecord:
    position: int
    wildtype: str
    residue_name: str
    residue_class: ResidueClass
    structural_region: str
    model_label: str
    catalytic_role: str
    zinc_coordination: bool
    min_zinc_distance_a: float
    min_substrate_distance_a: float
    sasa_a2: float
    relative_sasa: float
    burial_class: str
    packing_neighbors: int
    secondary_structure: str
    mean_b_factor: float
    structural_uncertainty: str
    experimental_evidence: str
    protection_reason: str
    allowed_substitution_classes: tuple[str, ...]
    classification_rule: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["residue_class"] = self.residue_class.value
        data["allowed_substitution_classes"] = list(self.allowed_substitution_classes)
        return data


def _heavy_coordinates(residue) -> np.ndarray:
    return np.asarray(
        [atom.coord for atom in residue if (atom.element or "").upper() != "H"],
        dtype=float,
    )


def _minimum_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2).min())


def _packing_neighbors(residues: list, index: int, cutoff_a: float = 4.5) -> int:
    focus = _heavy_coordinates(residues[index])
    neighbors = 0
    for other_index, other in enumerate(residues):
        if other_index == index:
            continue
        if _minimum_distance(focus, _heavy_coordinates(other)) <= cutoff_a:
            neighbors += 1
    return neighbors


def _region(position: int, min_substrate_a: float, min_zinc_a: float, relative_sasa: float) -> str:
    if position in HARD_PROTECTION_REASONS:
        return "catalytic_center"
    if min_substrate_a <= 5.0:
        return "substrate_interface"
    if min_substrate_a <= 8.0 or min_zinc_a <= 10.0 or position in {91, 126}:
        return "second_shell"
    if relative_sasa <= 0.35:
        return "distal_stability"
    return "scaffold_surface"


def _allowed_classes(
    residue_class: ResidueClass, structural_region: str
) -> tuple[str, ...]:
    if residue_class in {ResidueClass.HARD_PROTECTED, ResidueClass.OUT_OF_SCOPE}:
        return ()
    if residue_class is ResidueClass.CONTEXT_SENSITIVE:
        return ("conservative", "experimentally_informed")
    if structural_region == "substrate_interface":
        return ("conservative", "hydrogen_bond", "hydrophobic_tuning")
    if structural_region == "second_shell":
        return ("conservative", "preorganization", "packing")
    if structural_region == "distal_stability":
        return ("conservative", "packing", "helix_propensity", "charge_balance")
    return ("conservative", "surface_charge", "solubility")


def classify_design_space(
    active_like_pdb: str | Path, source_cif: str | Path
) -> tuple[ResidueDesignRecord, ...]:
    """Classify all 215 positions using deposited and coordinate evidence."""
    structure = PDBParser(QUIET=True).get_structure("active_like_inferred", active_like_pdb)
    model = structure[0]
    if not {"A", "B", "C"} <= {chain.id for chain in model}:
        raise ValueError("Design-space classification requires DP622, Aβ, and Zn chains")
    protein = list(model["A"])
    if [residue.id[1] for residue in protein] != list(range(1, 216)):
        raise ValueError("DP622 protein chain must contain contiguous residues 1..215")
    substrate_coordinates = np.concatenate(
        [_heavy_coordinates(residue) for residue in model["B"]], axis=0
    )
    zinc_coordinates = np.asarray(
        [atom.coord for residue in model["C"] for atom in residue if atom.element == "ZN"],
        dtype=float,
    )
    if zinc_coordinates.shape != (1, 3):
        raise ValueError("Design-space classification requires exactly one Zn atom")

    ShrakeRupley(probe_radius=1.4, n_points=100).compute(structure, level="R")
    secondary = secondary_structure_by_dp622_position(source_cif)
    records: list[ResidueDesignRecord] = []
    for index, residue in enumerate(protein):
        position = residue.id[1]
        wildtype = seq1(residue.resname, custom_map={"MSE": "M"})
        residue_coordinates = _heavy_coordinates(residue)
        min_zinc = _minimum_distance(residue_coordinates, zinc_coordinates)
        min_substrate = _minimum_distance(residue_coordinates, substrate_coordinates)
        sasa = float(getattr(residue, "sasa", 0.0))
        relative_sasa = min(1.0, sasa / MAXIMUM_RESIDUE_SASA_A2[wildtype])
        if relative_sasa < 0.2:
            burial = "buried"
        elif relative_sasa < 0.5:
            burial = "partially_buried"
        else:
            burial = "exposed"
        b_factors = [float(atom.bfactor) for atom in residue]
        mean_b = float(np.mean(b_factors))
        if position in HARD_PROTECTION_REASONS:
            residue_class = ResidueClass.HARD_PROTECTED
            rule = "Protected by deposited coordination or indispensable catalytic role."
        elif position in {91, 126}:
            residue_class = ResidueClass.CONTEXT_SENSITIVE
            rule = "Known single/double mutation evidence requires context-aware exploration."
        elif position in {1, 2, 214, 215}:
            residue_class = ResidueClass.OUT_OF_SCOPE
            rule = "Coordinate-fragment boundary excluded from prospective mutation."
        else:
            residue_class = ResidueClass.DESIGNABLE
            rule = (
                "Resolved mapped scaffold position with region-specific hypotheses; class uses "
                "contacts, burial, packing, secondary structure, uncertainty, and known evidence."
            )
        structural_region = _region(position, min_substrate, min_zinc, relative_sasa)
        if position in {1, 2, 214, 215}:
            structural_region = "fragment_boundary"
        uncertainty = (
            "fragment_boundary_high"
            if position in {1, 2, 214, 215}
            else "deposited_coordinate_supported"
        )
        records.append(
            ResidueDesignRecord(
                position=position,
                wildtype=wildtype,
                residue_name=residue.resname,
                residue_class=residue_class,
                structural_region=structural_region,
                model_label="active_like_inferred",
                catalytic_role=CATALYTIC_ROLES.get(position, "none_assigned"),
                zinc_coordination=position in DIRECT_ZINC_LIGANDS,
                min_zinc_distance_a=round(min_zinc, 4),
                min_substrate_distance_a=round(min_substrate, 4),
                sasa_a2=round(sasa, 4),
                relative_sasa=round(relative_sasa, 6),
                burial_class=burial,
                packing_neighbors=_packing_neighbors(protein, index),
                secondary_structure=secondary[position],
                mean_b_factor=round(mean_b, 4),
                structural_uncertainty=uncertainty,
                experimental_evidence=KNOWN_EXPERIMENTAL_EVIDENCE.get(
                    position, "No position-specific published mutation result assigned."
                ),
                protection_reason=HARD_PROTECTION_REASONS.get(position, ""),
                allowed_substitution_classes=_allowed_classes(
                    residue_class, structural_region
                ),
                classification_rule=rule,
            )
        )
    return tuple(records)


def write_design_space(
    records: tuple[ResidueDesignRecord, ...],
    csv_path: str | Path,
    json_path: str | Path,
) -> None:
    rows = [record.to_dict() for record in records]
    csv_rows = []
    for row in rows:
        csv_row = dict(row)
        csv_row["allowed_substitution_classes"] = ";".join(
            row["allowed_substitution_classes"]
        )
        csv_rows.append(csv_row)
    csv_destination, json_destination = Path(csv_path), Path(json_path)
    csv_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(csv_destination, index=False)
    json_destination.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

