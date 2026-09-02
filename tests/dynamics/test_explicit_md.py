from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from atlas.dynamics.ensemble_analysis import summarize_replicated_md
from atlas.dynamics.explicit_md import (
    build_equilibration_plan,
    build_replica_plan,
    prepare_explicit_system,
    run_explicit_md_replica,
    run_replicated_explicit_md,
)
from atlas.dynamics.models import ExplicitMDConfig, ReplicaResult
from atlas.structure.reconstruct import reconstruct_active_like


SOURCE = Path(__file__).parents[2] / "data" / "23WN.cif"


def _pdb(tmp_path: Path) -> Path:
    path = tmp_path / "active_like.pdb"
    reconstruct_active_like(SOURCE, path, tmp_path / "map.csv")
    return path


def test_replica_plan_has_independent_seeds_and_documented_protocol() -> None:
    config = ExplicitMDConfig(replica_count=3, base_seed=622)
    plan = build_replica_plan(config)
    assert [replica.seed for replica in plan] == [622, 8541, 16460]
    assert len({replica.seed for replica in plan}) == 3
    assert all(replica.production_steps == 250_000 for replica in plan)
    assert all(replica.production_time_ps == 500.0 for replica in plan)
    assert config.position_restraint_schedule_kj_mol_nm2[-1] == 0.0
    assert config.zinc_restraint_k_kj_mol_nm2 > 0


def test_equilibration_plan_uses_stable_timestep_without_shortening_duration() -> None:
    config = ExplicitMDConfig(equilibration_steps=50_000, timestep_fs=2.0)
    plan = build_equilibration_plan(config)
    assert [force_constant for force_constant, _ in plan] == [1_000.0, 100.0, 10.0, 0.0]
    assert sum(steps for _, steps in plan) == 200_000
    assert sum(steps for _, steps in plan) * 0.5 / 1_000.0 == 100.0


def test_explicit_preparation_adds_solvent_and_tracks_fragment_termination(
    tmp_path: Path,
) -> None:
    pytest.importorskip("openmm")
    pytest.importorskip("pdbfixer")
    config = ExplicitMDConfig(
        replica_count=1,
        solvent_padding_nm=0.4,
        nonbonded_cutoff_nm=0.8,
        equilibration_steps=4,
        production_steps=4,
        report_interval_steps=2,
        checkpoint_interval_steps=2,
        platform_name="CPU",
    )
    prepared = prepare_explicit_system(_pdb(tmp_path), tmp_path / "prepared", config, seed=622)
    metadata = json.loads(prepared.metadata_json.read_text())
    assert prepared.periodic is True
    assert prepared.atom_count > prepared.solute_atom_count
    assert prepared.water_residue_count > 0
    assert metadata["modeled_fragment_terminal_atoms"] == ["A:215:OXT", "B:41:OXT"]
    assert metadata["force_fields"] == ["amber14-all.xml", "amber14/tip3pfb.xml"]
    assert metadata["zinc_coordination_model"] == "bonded harmonic restraint"
    assert metadata["zinc_ligand_nonbonded_exclusions"] == [
        "A:95:NE2",
        "A:99:NE2",
        "A:122:OE1",
        "A:122:OE2",
        "B:38:O",
    ]
    assert metadata["claim_boundary"].startswith("Structural/dynamic simulation")


def test_tiny_real_replica_persists_checkpoint_trajectory_and_metrics(tmp_path: Path) -> None:
    pytest.importorskip("openmm")
    pytest.importorskip("pdbfixer")
    config = ExplicitMDConfig(
        replica_count=1,
        solvent_padding_nm=0.4,
        nonbonded_cutoff_nm=0.8,
        equilibration_steps=4,
        production_steps=4,
        report_interval_steps=2,
        checkpoint_interval_steps=2,
        minimization_max_iterations=2,
        platform_name="CPU",
    )
    result = run_explicit_md_replica(
        _pdb(tmp_path), tmp_path / "replica-01", config, replica_id=1, seed=622
    )
    assert result.status == "completed", result.error
    assert result.trajectory_path and result.trajectory_path.is_file()
    assert result.checkpoint_path and result.checkpoint_path.is_file()
    assert result.metrics_csv and result.metrics_csv.is_file()
    metrics = pd.read_csv(result.metrics_csv)
    assert len(metrics) == 2
    assert {
        "zn_scissile_o_distance_a",
        "substrate_rmsd_a",
        "contact_fraction",
        "active_site_rmsd_a",
    } <= set(metrics.columns)
    resumed = run_explicit_md_replica(
        _pdb(tmp_path), tmp_path / "replica-01", config, replica_id=1, seed=622
    )
    assert resumed.resumed is True
    assert len(pd.read_csv(resumed.metrics_csv)) == 2


def test_replicated_runner_includes_independent_results_and_reference_flag(
    tmp_path: Path,
) -> None:
    calls = []

    def fake_runner(pdb_path, output_dir, config, *, replica_id, seed):
        calls.append((Path(pdb_path).name, replica_id, seed))
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        metrics = directory / "metrics.csv"
        pd.DataFrame([{"step": 1, "substrate_rmsd_a": replica_id}]).to_csv(
            metrics, index=False
        )
        return ReplicaResult(
            replica_id=replica_id,
            seed=seed,
            status="completed",
            output_dir=directory,
            trajectory_path=directory / "trajectory.dcd",
            checkpoint_path=directory / "checkpoint.chk",
            metrics_csv=metrics,
            final_pdb=directory / "final.pdb",
            summary_json=directory / "summary.json",
            resumed=False,
            error="",
        )

    config = ExplicitMDConfig(replica_count=3, equilibration_steps=4, production_steps=4)
    ensemble = run_replicated_explicit_md(
        tmp_path / "candidate.pdb",
        tmp_path / "ensemble",
        config,
        system_label="candidate",
        replica_runner=fake_runner,
    )
    assert [call[2] for call in calls] == [622, 8541, 16460]
    assert len(ensemble.replicas) == 3
    assert ensemble.system_label == "candidate"


def test_ensemble_summary_reports_replica_disagreement_without_hiding_axes(
    tmp_path: Path,
) -> None:
    replicas = []
    for replica_id, values in enumerate(((0.4, 0.5), (0.5, 0.6), (1.1, 1.2)), start=1):
        directory = tmp_path / f"replica-{replica_id}"
        directory.mkdir()
        metrics = directory / "metrics.csv"
        pd.DataFrame(
            {
                "substrate_rmsd_a": values,
                "zn_scissile_o_distance_a": (2.2, 2.3),
                "contact_fraction": (0.9, 0.8),
                "active_site_rmsd_a": (0.3, 0.4),
            }
        ).to_csv(metrics, index=False)
        replicas.append(
            ReplicaResult(
                replica_id=replica_id,
                seed=replica_id,
                status="completed",
                output_dir=directory,
                trajectory_path=None,
                checkpoint_path=None,
                metrics_csv=metrics,
                final_pdb=None,
                summary_json=None,
                resumed=False,
                error="",
            )
        )
    summary = summarize_replicated_md(tuple(replicas), tmp_path / "summary")
    payload = json.loads(summary.summary_json.read_text())
    assert summary.completed_replicas == 3
    assert payload["replica_disagreement"]["substrate_rmsd_a"]["range"] > 0.5
    assert "universal_score" not in payload
    assert summary.replica_summary_csv.is_file()
