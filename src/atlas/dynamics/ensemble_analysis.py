"""Cross-replica summaries that preserve axes and expose disagreement."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from atlas.dynamics.models import EnsembleSummary, ReplicaResult


METRIC_COLUMNS = (
    "protein_ca_rmsd_a",
    "active_site_rmsd_a",
    "substrate_rmsd_a",
    "substrate_centroid_drift_a",
    "contact_fraction",
    "zn_h95_distance_a",
    "zn_h99_distance_a",
    "zn_e122_distance_a",
    "zn_scissile_o_distance_a",
    "e96_scissile_carbon_distance_a",
    "h172_scissile_o_distance_a",
)


def summarize_replicated_md(
    replicas: tuple[ReplicaResult, ...], output_dir: str | Path
) -> EnsembleSummary:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    rmsf_by_replica: dict[int, float] = {}
    for replica in replicas:
        if replica.status != "completed" or replica.metrics_csv is None:
            continue
        frame = pd.read_csv(replica.metrics_csv)
        row: dict[str, float | int] = {
            "replica_id": replica.replica_id,
            "seed": replica.seed,
            "snapshot_count": len(frame),
        }
        for column in METRIC_COLUMNS:
            if column in frame:
                values = frame[column].astype(float)
                row[f"{column}_mean"] = float(values.mean())
                row[f"{column}_std"] = float(values.std(ddof=0))
                row[f"{column}_q05"] = float(values.quantile(0.05))
                row[f"{column}_q95"] = float(values.quantile(0.95))
        coordinates = replica.output_dir / "analysis_coordinates.npz"
        if coordinates.is_file():
            array = np.load(coordinates)["aligned_selected_positions_a"]
            if len(array):
                mean_positions = array.mean(axis=0)
                per_atom = np.sqrt(np.mean(np.sum((array - mean_positions) ** 2, axis=2), axis=0))
                rmsf_by_replica[replica.replica_id] = float(per_atom.mean())
                row["selected_atom_mean_rmsf_a"] = rmsf_by_replica[replica.replica_id]
        rows.append(row)
    table = pd.DataFrame(rows)
    table_path = destination / "replica_summaries.csv"
    table.to_csv(table_path, index=False)
    disagreement: dict[str, dict[str, float]] = {}
    ensemble_axes: dict[str, dict[str, float]] = {}
    if not table.empty:
        for metric in METRIC_COLUMNS:
            column = f"{metric}_mean"
            if column not in table:
                continue
            values = table[column].astype(float)
            disagreement[metric] = {
                "standard_deviation": float(values.std(ddof=0)),
                "range": float(values.max() - values.min()),
            }
            ensemble_axes[metric] = {
                "replica_mean": float(values.mean()),
                "replica_min": float(values.min()),
                "replica_max": float(values.max()),
            }
    completed = len(rows)
    invalid = len(replicas) - completed
    summary_path = destination / "ensemble_summary.json"
    payload = {
        "completed_replicas": completed,
        "invalid_replicas": invalid,
        "evidence_status": "available" if completed >= 2 else "unavailable",
        "ensemble_axes": ensemble_axes,
        "replica_disagreement": disagreement,
        "selected_atom_mean_rmsf_a_by_replica": rmsf_by_replica,
        "claim_boundary": (
            "Replicated structural/dynamic evidence; no catalytic turnover or kcat/Km claim."
        ),
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return EnsembleSummary(
        completed_replicas=completed,
        invalid_replicas=invalid,
        replica_summary_csv=table_path,
        summary_json=summary_path,
    )

