from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from Bio.PDB import PDBParser

from atlas.adaptive.models import CandidateRecord, DesignStrategy
from atlas.structure.mutant_complex import (
    MutantConstructionError,
    build_mutant_complex,
    detect_catastrophic_clashes,
    validate_required_zinc_coordination,
)
from atlas.structure.chemistry import validate_unintended_zinc_coordination
from atlas.structure.reconstruct import reconstruct_active_like


SOURCE = Path(__file__).parents[2] / "data" / "23WN.cif"
REFERENCE = "RNLELARAADVTVTVADTPEEMYEAAKVAVETVRELAAGDPRRDEYVALAERLFRTGIERGGIAGIAIYADGRRRVFVVAPSDASDEALIYALAHELAHLIIAEDLERRGLPLSAVPPGVVEGLADVFGATAYAAYLELKGEKVTLEKWREMQLRLAEETERIGREAGLEAHVEGGRIAAEIARRTNEEEAQKLIEEVKPLVEFILGLLRVARTA"


def _candidate(mutation: str) -> CandidateRecord:
    return CandidateRecord.create(
        reference_sequence=REFERENCE,
        mutations=[mutation],
        parents=(),
        strategy=DesignStrategy.SUBSTRATE_INTERFACE,
        structural_region="substrate_interface",
        round_index=1,
        hypothesis="Test an arbitrary legal side-chain substitution.",
        intended_upside="Tune the interface.",
        expected_risk="Could clash.",
    )


def _starting_pdb(tmp_path: Path) -> Path:
    pdb = tmp_path / "active_like.pdb"
    reconstruct_active_like(SOURCE, pdb, tmp_path / "map.csv")
    return pdb


def test_arbitrary_mutant_builds_complete_sidechain_and_retains_complex(tmp_path: Path) -> None:
    pytest.importorskip("pdbfixer")
    source = _starting_pdb(tmp_path)
    candidate = _candidate("A37W")
    result = build_mutant_complex(source, candidate, tmp_path / "mutant", seed=622)

    structure = PDBParser(QUIET=True).get_structure("mutant", result.raw_pdb)
    residue = structure[0]["A"][37]
    assert residue.resname == "TRP"
    assert {"CG", "CD1", "NE1", "CE2", "CZ2", "CH2", "CZ3", "CE3", "CD2"} <= {
        atom.id for atom in residue
    }
    assert [residue.id[1] for residue in structure[0]["B"]] == list(range(34, 42))
    assert sum(1 for atom in structure.get_atoms() if atom.element == "ZN") == 1
    assert result.modeled_atom_count >= 9
    assert result.hard_violations == ()


def test_atom_provenance_distinguishes_retained_modeled_and_removed_atoms(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pdbfixer")
    source = _starting_pdb(tmp_path)
    result = build_mutant_complex(source, _candidate("A37W"), tmp_path / "mutant", seed=622)
    table = pd.read_csv(result.atom_provenance_csv)
    summary = json.loads(result.provenance_json.read_text())

    assert {"retained_deposited_coordinate", "modeled_sidechain"} <= set(table.source)
    assert summary["model_label"] == "active_like_inferred"
    assert summary["candidate_id"] == result.candidate_id
    assert summary["pdbfixer_commit"] == "5e658a4fb8d2b90d65ca8408c845cbf104ce7099"
    assert summary["removed_input_atoms"] >= 0
    substrate = table.loc[(table.chain == "B") & (table.residue == 38)]
    assert set(substrate.source) == {"retained_deposited_coordinate"}
    assert substrate.coordinate_displacement_a.max() < 1e-5


def test_protected_catalytic_mutation_is_a_hard_construction_error(tmp_path: Path) -> None:
    pytest.importorskip("pdbfixer")
    with pytest.raises(MutantConstructionError, match="protected"):
        build_mutant_complex(
            _starting_pdb(tmp_path), _candidate("E96Q"), tmp_path / "illegal", seed=622
        )


def test_catastrophic_inter_residue_clash_is_detected(tmp_path: Path) -> None:
    pdb = _starting_pdb(tmp_path)
    structure = PDBParser(QUIET=True).get_structure("clash", pdb)
    first = structure[0]["A"][10]["CA"]
    second = structure[0]["A"][20]["CA"]
    second.coord = first.coord.copy()
    from Bio.PDB import PDBIO

    writer = PDBIO()
    writer.set_structure(structure)
    clashing = tmp_path / "clashing.pdb"
    writer.save(str(clashing))
    clashes = detect_catastrophic_clashes(clashing, cutoff_a=0.65)
    assert any("A:10:CA" in clash and "A:20:CA" in clash for clash in clashes)


def test_starting_and_legal_mutant_retain_required_zinc_coordination(tmp_path: Path) -> None:
    pytest.importorskip("pdbfixer")
    source = _starting_pdb(tmp_path)
    candidate = _candidate("A37W")
    result = build_mutant_complex(source, candidate, tmp_path / "mutant", seed=622)
    assert validate_required_zinc_coordination(source, "WT") == ()
    assert validate_required_zinc_coordination(result.raw_pdb, candidate.candidate_id) == ()


def test_unintended_new_metal_ligand_is_a_hard_candidate_specific_failure(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pdbfixer")
    source = _starting_pdb(tmp_path)
    candidate = CandidateRecord.create(
        reference_sequence=REFERENCE,
        mutations=["A125H"],
        parents=(),
        strategy=DesignStrategy.SECOND_SHELL_PREORGANIZATION,
        structural_region="second_shell",
        round_index=1,
        hypothesis="Test candidate-specific metal liability.",
        intended_upside="Test fixture.",
        expected_risk="Unintended Zn coordination.",
        metal_liability="HIGH_RISK_METAL_SITE_HYPOTHESIS",
        requires_candidate_geometry=True,
    )
    result = build_mutant_complex(source, candidate, tmp_path / "mutant", seed=622)
    structure = PDBParser(QUIET=True).get_structure("mutant", result.raw_pdb)
    zinc = next(atom for atom in structure.get_atoms() if atom.element == "ZN")
    introduced = structure[0]["A"][125]["NE2"]
    introduced.coord = zinc.coord + [2.0, 0.0, 0.0]
    from Bio.PDB import PDBIO

    writer = PDBIO()
    writer.set_structure(structure)
    coordinating = tmp_path / "coordinating.pdb"
    writer.save(str(coordinating))

    violations = validate_unintended_zinc_coordination(coordinating, candidate)

    assert violations[0].code == "unintended_zinc_coordination"
