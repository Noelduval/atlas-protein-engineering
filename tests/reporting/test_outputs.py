from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from atlas.reporting.csv_outputs import write_provenance
from atlas.reporting.plots import (
    plot_candidate_ranking,
    plot_catalytic_geometry,
    plot_validation_dashboard,
)


def test_provenance_records_current_atlas_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    versions = {
        "atlas-protein-engineering": "1.0.0",
        "biopython": "1.85",
        "numpy": "2.0.0",
        "pandas": "2.2.0",
        "matplotlib": "3.9.0",
        "openmm": "8.2.0",
        "torch": "2.5.1",
    }
    monkeypatch.setattr("atlas.reporting.csv_outputs.version", versions.__getitem__)

    input_path = tmp_path / "input.cif"
    input_path.write_text("data")
    output = tmp_path / "provenance.json"
    write_provenance(input_path, output)

    packages = __import__("json").loads(output.read_text())["packages"]
    assert packages["atlas-protein-engineering"] == "1.0.0"
    assert "atlas-therapeutic-optimization" not in packages


def test_required_plots_are_nonempty_pngs(tmp_path: Path) -> None:
    validation = pd.DataFrame(
        {
            "variant_id": ["WT", "Y91F"],
            "predicted_ddg_or_score": [0.0, -0.2],
            "gate_outcome": ["reference", "pass"],
        }
    )
    geometry = pd.DataFrame(
        {
            "variant_id": ["WT", "Y91F"],
            "zn_scissile_oxygen_distance_a": [2.3, 2.4],
            "e96_to_scissile_carbonyl_distance_a": [3.0, 3.1],
        }
    )
    ranking = pd.DataFrame(
        {"variant_id": ["L10A", "V11A"], "ranking_score": [-0.1, 0.2]}
    )
    paths = [
        plot_validation_dashboard(validation, tmp_path / "validation.png"),
        plot_catalytic_geometry(geometry, tmp_path / "geometry.png"),
        plot_candidate_ranking(ranking, tmp_path / "ranking.png"),
    ]
    assert all(path.stat().st_size > 1_000 for path in paths)
