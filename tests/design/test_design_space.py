from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from atlas.design.design_space import (
    ResidueClass,
    classify_design_space,
    write_design_space,
)
from atlas.structure.reconstruct import reconstruct_active_like


SOURCE = Path(__file__).parents[2] / "data" / "23WN.cif"


def _space(tmp_path: Path):
    pdb = tmp_path / "active_like.pdb"
    reconstruct_active_like(SOURCE, pdb, tmp_path / "numbering.csv")
    return classify_design_space(pdb, SOURCE)


def test_design_space_classifies_every_dp622_position_once(tmp_path: Path) -> None:
    records = _space(tmp_path)
    assert len(records) == 215
    assert [record.position for record in records] == list(range(1, 216))
    assert {record.residue_class for record in records} <= set(ResidueClass)
    assert {record.model_label for record in records} == {"active_like_inferred"}


def test_catalytic_machinery_is_hard_protected(tmp_path: Path) -> None:
    by_position = {record.position: record for record in _space(tmp_path)}
    for position in (95, 96, 99, 122, 172):
        assert by_position[position].residue_class is ResidueClass.HARD_PROTECTED
        assert by_position[position].protection_reason
    assert by_position[96].catalytic_role == "general_base_glutamate"
    assert by_position[95].zinc_coordination is True
    assert by_position[99].zinc_coordination is True
    assert by_position[122].zinc_coordination is True


def test_known_context_dependent_positions_are_not_blacklisted(tmp_path: Path) -> None:
    by_position = {record.position: record for record in _space(tmp_path)}
    for position in (91, 126):
        record = by_position[position]
        assert record.residue_class is ResidueClass.CONTEXT_SENSITIVE
        assert "published" in record.experimental_evidence.lower()
        assert record.allowed_substitution_classes


def test_classification_uses_multiple_structural_evidence_types(tmp_path: Path) -> None:
    records = _space(tmp_path)
    interface = [record for record in records if record.structural_region == "substrate_interface"]
    distal = [record for record in records if record.structural_region == "distal_stability"]
    assert interface
    assert distal
    assert all(record.min_substrate_distance_a is not None for record in records)
    assert all(record.relative_sasa is not None for record in records)
    assert all(record.secondary_structure in {"helix", "sheet", "coil"} for record in records)
    assert all(record.packing_neighbors >= 0 for record in records)
    assert all(record.mean_b_factor >= 0 for record in records)


def test_design_space_exports_reproducible_csv_and_json(tmp_path: Path) -> None:
    records = _space(tmp_path)
    csv_path = tmp_path / "design_space.csv"
    json_path = tmp_path / "design_space.json"
    write_design_space(records, csv_path, json_path)

    table = pd.read_csv(csv_path)
    payload = json.loads(json_path.read_text())
    assert len(table) == len(payload) == 215
    assert table.loc[table.position == 96, "residue_class"].item() == "HARD_PROTECTED"
    assert payload[95]["position"] == 96
    assert payload[95]["model_label"] == "active_like_inferred"
    assert "classification_rule" in payload[95]

