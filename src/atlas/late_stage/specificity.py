"""Bounded negative-design evidence from the deposited 23WN Aβ42 pose."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.SeqUtils import seq1


@dataclass(frozen=True)
class SubstratePanelMember:
    panel_member_id: str
    start: int
    end: int
    sequence: str
    is_intended: bool


@dataclass(frozen=True)
class SubstratePanel:
    source_structure: str
    source_accession: str
    canonical_sequence: str
    members: tuple[SubstratePanelMember, ...]


@dataclass(frozen=True)
class ContactFingerprintEntry:
    enzyme_residue: int
    enzyme_resname: str
    substrate_position: int
    deposited_substrate_resname: str
    minimum_heavy_atom_distance_a: float


@dataclass(frozen=True)
class PanelCompatibilityScore:
    panel_member_id: str
    sequence: str
    is_intended: bool
    contact_compatibility_score: float


@dataclass(frozen=True)
class SpecificityAssessment:
    variant_id: str
    classification: str
    contact_count: int
    contact_fingerprint: tuple[ContactFingerprintEntry, ...]
    scores: tuple[PanelCompatibilityScore, ...]
    intended_score: float | None
    best_off_target_score: float | None
    preference_margin: float | None
    method: str
    interpretation_limit: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class SpecificityContactConstructionAudit:
    total_residue_contacts: int
    chemistry_bearing_contacts: int
    backbone_only_contacts: int
    backbone_only_fraction: float


_CONTACT_CUTOFF_A = 4.5
_INTENDED_START = 34
_INTENDED_END = 41
_PANEL_WINDOWS = ((31, 38), (32, 39), (33, 40), (34, 41), (35, 42))
_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT"})


def _values(value: str | list[str]) -> list[str]:
    return value if isinstance(value, list) else [value]


def _clean_sequence(value: str) -> str:
    return "".join(value.split()).upper()


def build_abeta42_s2_panel(source_cif: str | Path) -> SubstratePanel:
    """Build the local S2 comparison panel from 23WN's deposited chain-B sequence."""
    path = Path(source_cif)
    if not path.is_file():
        raise FileNotFoundError(f"23WN input does not exist: {path}")

    cif = MMCIF2Dict(str(path))
    entity_ids = _values(cif["_entity_poly.entity_id"])
    strand_ids = _values(cif["_entity_poly.pdbx_strand_id"])
    sequences = _values(cif["_entity_poly.pdbx_seq_one_letter_code_can"])
    chain_b = [
        (entity_id, _clean_sequence(sequence))
        for entity_id, strands, sequence in zip(entity_ids, strand_ids, sequences)
        if "B" in {strand.strip() for strand in strands.split(",")}
    ]
    if len(chain_b) != 1:
        raise ValueError(f"Expected one polymer entity for 23WN chain B, found {len(chain_b)}")
    entity_id, canonical = chain_b[0]
    if len(canonical) != 42:
        raise ValueError(f"Expected deposited Aβ42 sequence, found length {len(canonical)}")

    ref_entities = _values(cif["_struct_ref.entity_id"])
    accessions = _values(cif["_struct_ref.pdbx_db_accession"])
    matching_accessions = [
        accession for ref_entity, accession in zip(ref_entities, accessions) if ref_entity == entity_id
    ]
    if matching_accessions != ["P05067"]:
        raise ValueError(
            "23WN chain B is not uniquely provenance-linked to UniProt P05067"
        )

    model = MMCIFParser(QUIET=True, auth_chains=True, auth_residues=True).get_structure(
        "23WN", path
    )[0]
    if "B" not in model:
        raise ValueError("23WN does not contain deposited chain B")
    resolved = [residue for residue in model["B"] if _INTENDED_START <= residue.id[1] <= _INTENDED_END]
    resolved_sequence = "".join(seq1(residue.resname) for residue in resolved)
    expected_resolved = canonical[_INTENDED_START - 1 : _INTENDED_END]
    if [residue.id[1] for residue in resolved] != list(range(34, 42)):
        raise ValueError("23WN chain B does not resolve the complete Aβ34–41 S2 segment")
    if resolved_sequence != expected_resolved:
        raise ValueError("Resolved 23WN chain-B residues disagree with deposited Aβ42 sequence")

    members = tuple(
        SubstratePanelMember(
            panel_member_id=(
                "s2_abeta34_41" if (start, end) == (34, 41) else f"abeta{start}_{end}"
            ),
            start=start,
            end=end,
            sequence=canonical[start - 1 : end],
            is_intended=(start, end) == (_INTENDED_START, _INTENDED_END),
        )
        for start, end in _PANEL_WINDOWS
    )
    return SubstratePanel(
        source_structure="PDB 23WN chain B",
        source_accession="UniProt P05067",
        canonical_sequence=canonical,
        members=members,
    )


def _heavy_atoms(residue) -> list:
    return [atom for atom in residue if atom.element.upper() != "H"]


def _contact_fingerprint(structure) -> tuple[ContactFingerprintEntry, ...]:
    model = next(structure.get_models())
    if "A" not in model or "B" not in model:
        return ()
    substrate = {
        residue.id[1]: residue
        for residue in model["B"]
        if _INTENDED_START <= residue.id[1] <= _INTENDED_END
    }
    if set(substrate) != set(range(_INTENDED_START, _INTENDED_END + 1)):
        return ()

    contacts: list[ContactFingerprintEntry] = []
    for enzyme_residue in model["A"]:
        enzyme_atoms = _heavy_atoms(enzyme_residue)
        if not enzyme_atoms:
            continue
        enzyme_coordinates = np.asarray([atom.coord for atom in enzyme_atoms], dtype=float)
        for position, substrate_residue in substrate.items():
            substrate_atoms = _heavy_atoms(substrate_residue)
            if not substrate_atoms:
                continue
            substrate_coordinates = np.asarray(
                [atom.coord for atom in substrate_atoms], dtype=float
            )
            distances = np.linalg.norm(
                enzyme_coordinates[:, None, :] - substrate_coordinates[None, :, :], axis=2
            )
            minimum = float(distances.min())
            if minimum <= _CONTACT_CUTOFF_A:
                contacts.append(
                    ContactFingerprintEntry(
                        enzyme_residue=enzyme_residue.id[1],
                        enzyme_resname=enzyme_residue.resname,
                        substrate_position=position,
                        deposited_substrate_resname=substrate_residue.resname,
                        minimum_heavy_atom_distance_a=minimum,
                    )
                )
    return tuple(
        sorted(contacts, key=lambda item: (item.substrate_position, item.enzyme_residue))
    )


def _specificity_sidechain_atoms(residue) -> list:
    atoms = [
        atom
        for atom in residue
        if atom.element.upper() != "H" and atom.id not in _BACKBONE_ATOMS
    ]
    if atoms:
        return atoms
    return [residue["CA"]] if residue.resname == "GLY" and "CA" in residue else []


def _sidechain_contact_fingerprint(structure) -> tuple[ContactFingerprintEntry, ...]:
    """Return chemistry-bearing contacts, excluding backbone-only proximity."""
    model = next(structure.get_models())
    if "A" not in model or "B" not in model:
        return ()
    substrate = {
        residue.id[1]: residue
        for residue in model["B"]
        if _INTENDED_START <= residue.id[1] <= _INTENDED_END
    }
    if set(substrate) != set(range(_INTENDED_START, _INTENDED_END + 1)):
        return ()

    contacts: list[ContactFingerprintEntry] = []
    for enzyme_residue in model["A"]:
        enzyme_atoms = _specificity_sidechain_atoms(enzyme_residue)
        if not enzyme_atoms:
            continue
        enzyme_coordinates = np.asarray([atom.coord for atom in enzyme_atoms], dtype=float)
        for position, substrate_residue in substrate.items():
            substrate_atoms = _specificity_sidechain_atoms(substrate_residue)
            if not substrate_atoms:
                continue
            substrate_coordinates = np.asarray(
                [atom.coord for atom in substrate_atoms], dtype=float
            )
            minimum = float(
                np.linalg.norm(
                    enzyme_coordinates[:, None, :]
                    - substrate_coordinates[None, :, :],
                    axis=2,
                ).min()
            )
            if minimum <= _CONTACT_CUTOFF_A:
                contacts.append(
                    ContactFingerprintEntry(
                        enzyme_residue=enzyme_residue.id[1],
                        enzyme_resname=enzyme_residue.resname,
                        substrate_position=position,
                        deposited_substrate_resname=substrate_residue.resname,
                        minimum_heavy_atom_distance_a=minimum,
                    )
                )
    return tuple(
        sorted(contacts, key=lambda item: (item.substrate_position, item.enzyme_residue))
    )


def specificity_contact_construction_audit(
    candidate_pdb: str | Path,
) -> SpecificityContactConstructionAudit:
    """Report how many resolved contacts depend on chemistry-bearing atoms."""
    path = Path(candidate_pdb)
    structure = PDBParser(QUIET=True).get_structure(path.stem, path)
    model = next(structure.get_models())
    if "A" not in model or "B" not in model:
        return SpecificityContactConstructionAudit(0, 0, 0, 0.0)
    substrate = {
        residue.id[1]: residue
        for residue in model["B"]
        if _INTENDED_START <= residue.id[1] <= _INTENDED_END
    }
    if set(substrate) != set(range(_INTENDED_START, _INTENDED_END + 1)):
        return SpecificityContactConstructionAudit(0, 0, 0, 0.0)

    total = 0
    chemistry_bearing = 0
    for enzyme_residue in model["A"]:
        enzyme_atoms = _heavy_atoms(enzyme_residue)
        enzyme_sidechain_atoms = _specificity_sidechain_atoms(enzyme_residue)
        if not enzyme_atoms:
            continue
        enzyme_coordinates = np.asarray([atom.coord for atom in enzyme_atoms], dtype=float)
        for substrate_residue in substrate.values():
            substrate_atoms = _heavy_atoms(substrate_residue)
            if not substrate_atoms:
                continue
            substrate_coordinates = np.asarray([atom.coord for atom in substrate_atoms], dtype=float)
            if float(
                np.linalg.norm(
                    enzyme_coordinates[:, None, :] - substrate_coordinates[None, :, :], axis=2
                ).min()
            ) > _CONTACT_CUTOFF_A:
                continue
            total += 1
            substrate_sidechain_atoms = _specificity_sidechain_atoms(substrate_residue)
            if enzyme_sidechain_atoms and substrate_sidechain_atoms:
                chemistry_bearing += 1
    backbone_only = total - chemistry_bearing
    return SpecificityContactConstructionAudit(
        total_residue_contacts=total,
        chemistry_bearing_contacts=chemistry_bearing,
        backbone_only_contacts=backbone_only,
        backbone_only_fraction=backbone_only / total if total else 0.0,
    )


_HYDROPHOBIC = frozenset("AVILMFWY")
_POSITIVE = frozenset("KRH")
_NEGATIVE = frozenset("DE")
_POLAR = frozenset("STNQC")


def _residue_class(one_letter: str) -> str:
    if one_letter in _HYDROPHOBIC:
        return "hydrophobic"
    if one_letter in _POSITIVE:
        return "positive"
    if one_letter in _NEGATIVE:
        return "negative"
    if one_letter in _POLAR:
        return "polar"
    if one_letter == "G":
        return "glycine"
    if one_letter == "P":
        return "proline"
    return "other"


def _contact_fit(enzyme_resname: str, substrate_one_letter: str) -> float:
    enzyme = _residue_class(seq1(enzyme_resname))
    substrate = _residue_class(substrate_one_letter)
    if substrate == "glycine":
        return 0.35
    if substrate == "proline":
        return -0.25
    if enzyme == substrate == "hydrophobic":
        return 1.0
    if (enzyme, substrate) in {("positive", "negative"), ("negative", "positive")}:
        return 1.0
    if enzyme == substrate == "polar":
        return 0.75
    if enzyme in {"positive", "negative"} and substrate == enzyme:
        return -0.75
    if {enzyme, substrate} == {"hydrophobic", "positive"} or {
        enzyme,
        substrate,
    } == {"hydrophobic", "negative"}:
        return -0.50
    if enzyme == "polar" or substrate == "polar":
        return 0.25
    return 0.0


def _score_member(
    member: SubstratePanelMember,
    contacts: tuple[ContactFingerprintEntry, ...],
) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for contact in contacts:
        sequence_index = contact.substrate_position - _INTENDED_START
        weight = _CONTACT_CUTOFF_A + 0.5 - contact.minimum_heavy_atom_distance_a
        weighted_sum += weight * _contact_fit(
            contact.enzyme_resname, member.sequence[sequence_index]
        )
        total_weight += weight
    return weighted_sum / total_weight


def assess_specificity(
    candidate_pdb: str | Path,
    source_cif: str | Path,
    *,
    variant_id: str | None = None,
) -> SpecificityAssessment:
    """Compare local Aβ42 windows in the candidate's resolved 23WN contact geometry."""
    path = Path(candidate_pdb)
    panel = build_abeta42_s2_panel(source_cif)
    structure = PDBParser(QUIET=True).get_structure(variant_id or path.stem, path)
    contacts = _contact_fingerprint(structure)
    method = (
        "Candidate-specific 23WN heavy-atom contact-compatibility heuristic over a "
        "source-backed local Aβ42 window panel."
    )
    limit = (
        "This bounded structural heuristic does not estimate binding affinity, cleavage "
        "kinetics, or proteome-wide specificity."
    )
    if not contacts:
        return SpecificityAssessment(
            variant_id=variant_id or path.stem,
            classification="indeterminate",
            contact_count=0,
            contact_fingerprint=(),
            scores=(),
            intended_score=None,
            best_off_target_score=None,
            preference_margin=None,
            method=method,
            interpretation_limit=limit,
            warnings=("Candidate lacks a complete resolved chain B Aβ34–41 contact pose.",),
        )

    scores = tuple(
        PanelCompatibilityScore(
            panel_member_id=member.panel_member_id,
            sequence=member.sequence,
            is_intended=member.is_intended,
            contact_compatibility_score=_score_member(member, contacts),
        )
        for member in panel.members
    )
    intended = next(score.contact_compatibility_score for score in scores if score.is_intended)
    best_off_target = max(
        score.contact_compatibility_score for score in scores if not score.is_intended
    )
    margin = intended - best_off_target
    warnings: tuple[str, ...] = ()
    if margin > 0.05:
        classification = "intended_preferred"
    elif margin < -0.05:
        classification = "off_target_warning"
        warnings = (
            "At least one local Aβ42 window has higher contact-compatibility than the "
            "intended deposited S2 window.",
        )
    else:
        classification = "indeterminate"
        warnings = (
            "The bounded contact heuristic does not separate the intended S2 window from "
            "the best local alternative.",
        )
    return SpecificityAssessment(
        variant_id=variant_id or path.stem,
        classification=classification,
        contact_count=len(contacts),
        contact_fingerprint=contacts,
        scores=scores,
        intended_score=intended,
        best_off_target_score=best_off_target,
        preference_margin=margin,
        method=method,
        interpretation_limit=limit,
        warnings=warnings,
    )


def assess_sidechain_specificity(
    candidate_pdb: str | Path,
    source_cif: str | Path,
    *,
    variant_id: str | None = None,
) -> SpecificityAssessment:
    """Score only chemistry-bearing side-chain contacts on the unchanged panel/gate."""
    path = Path(candidate_pdb)
    panel = build_abeta42_s2_panel(source_cif)
    structure = PDBParser(QUIET=True).get_structure(variant_id or path.stem, path)
    contacts = _sidechain_contact_fingerprint(structure)
    method = (
        "Candidate-specific side-chain heavy-atom contact-compatibility heuristic over "
        "the source-backed local Aβ42 window panel (atlas-specificity-v2)."
    )
    limit = (
        "This bounded structural heuristic does not estimate binding affinity, cleavage "
        "kinetics, or proteome-wide specificity."
    )
    if not contacts:
        return SpecificityAssessment(
            variant_id=variant_id or path.stem,
            classification="indeterminate",
            contact_count=0,
            contact_fingerprint=(),
            scores=(),
            intended_score=None,
            best_off_target_score=None,
            preference_margin=None,
            method=method,
            interpretation_limit=limit,
            warnings=(
                "Candidate lacks a complete chemistry-bearing side-chain contact "
                "fingerprint for resolved chain B Aβ34–41.",
            ),
        )

    scores = tuple(
        PanelCompatibilityScore(
            panel_member_id=member.panel_member_id,
            sequence=member.sequence,
            is_intended=member.is_intended,
            contact_compatibility_score=_score_member(member, contacts),
        )
        for member in panel.members
    )
    intended = next(score.contact_compatibility_score for score in scores if score.is_intended)
    best_off_target = max(
        score.contact_compatibility_score for score in scores if not score.is_intended
    )
    margin = intended - best_off_target
    if margin > 0.05:
        classification = "intended_preferred"
        warnings: tuple[str, ...] = ()
    elif margin < -0.05:
        classification = "off_target_warning"
        warnings = (
            "At least one local Aβ42 window has higher side-chain contact-compatibility "
            "than the intended deposited S2 window.",
        )
    else:
        classification = "indeterminate"
        warnings = (
            "The bounded side-chain contact heuristic does not separate the intended "
            "S2 window from the best local alternative.",
        )
    return SpecificityAssessment(
        variant_id=variant_id or path.stem,
        classification=classification,
        contact_count=len(contacts),
        contact_fingerprint=contacts,
        scores=scores,
        intended_score=intended,
        best_off_target_score=best_off_target,
        preference_margin=margin,
        method=method,
        interpretation_limit=limit,
        warnings=warnings,
    )
