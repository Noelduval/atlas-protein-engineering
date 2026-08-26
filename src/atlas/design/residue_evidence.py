"""Deposition-backed residue evidence used to define the DP622 design space."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from atlas.structure.numbering import deposited_to_dp622


DIRECT_ZINC_LIGANDS = frozenset({95, 99, 122})
CATALYTIC_ROLES = {
    95: "direct_zinc_ligand",
    96: "general_base_glutamate",
    99: "direct_zinc_ligand",
    122: "direct_zinc_ligand",
    172: "designed_oxyanion_hole_histidine",
}
HARD_PROTECTION_REASONS = {
    95: "Deposited direct Zn ligand H95.",
    96: "Intended general-base glutamate restored from deposited E96Q.",
    99: "Deposited direct Zn ligand H99.",
    122: "Deposited direct Zn ligand E122.",
    172: "Designed oxyanion-hole histidine; H172A is a published negative control.",
}
KNOWN_EXPERIMENTAL_EVIDENCE = {
    91: (
        "Published Y91F single-mutant catalytic efficiency improved, while the "
        "published Y91F/D126A combination regressed; context dependent."
    ),
    126: (
        "Published D126A single-mutant catalytic efficiency improved, while the "
        "published Y91F/D126A combination regressed; context dependent."
    ),
    172: "Published H172A reduced catalytic efficiency; functional negative control.",
}


def _as_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def secondary_structure_by_dp622_position(source_cif: str | Path) -> dict[int, str]:
    """Return deposition-provided helix/sheet assignments in DP622 numbering."""
    data = MMCIF2Dict(str(source_cif))
    assignments = {position: "coil" for position in range(1, 216)}

    helix_chains = _as_list(data.get("_struct_conf.beg_auth_asym_id"))
    helix_starts = _as_list(data.get("_struct_conf.beg_auth_seq_id"))
    helix_ends = _as_list(data.get("_struct_conf.end_auth_seq_id"))
    for chain, start_text, end_text in zip(helix_chains, helix_starts, helix_ends):
        if chain != "A":
            continue
        start, end = int(start_text), int(end_text)
        for deposited in range(max(25, start), min(239, end) + 1):
            assignments[deposited_to_dp622(deposited)] = "helix"

    sheet_chains = _as_list(data.get("_struct_sheet_range.beg_auth_asym_id"))
    sheet_starts = _as_list(data.get("_struct_sheet_range.beg_auth_seq_id"))
    sheet_ends = _as_list(data.get("_struct_sheet_range.end_auth_seq_id"))
    for chain, start_text, end_text in zip(sheet_chains, sheet_starts, sheet_ends):
        if chain != "A":
            continue
        start, end = int(start_text), int(end_text)
        for deposited in range(max(25, start), min(239, end) + 1):
            assignments[deposited_to_dp622(deposited)] = "sheet"
    return assignments

