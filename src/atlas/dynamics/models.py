"""Typed OpenMM configuration and results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DynamicsConfig:
    temperature_k: float = 300.0
    timestep_fs: float = 2.0
    md_steps: int = 5_000
    snapshot_count: int = 10
    restraint_k_kj_mol_nm2: float = 1_000.0
    zinc_restraint_k_kj_mol_nm2: float = 2_000.0
    minimization_tolerance_kj_mol_nm: float = 10.0
    minimization_max_iterations: int = 2_000
    random_seed: int = 622


@dataclass(frozen=True)
class DynamicsResult:
    status: str
    output_pdb: Path | None
    snapshot_records: tuple[dict[str, float | int | str], ...]
    warning: str = ""


@dataclass(frozen=True)
class ExplicitMDConfig:
    temperature_k: float = 300.0
    pressure_bar: float = 1.0
    ph: float = 7.4
    ionic_strength_molar: float = 0.15
    solvent_padding_nm: float = 1.0
    nonbonded_cutoff_nm: float = 1.0
    timestep_fs: float = 2.0
    friction_per_ps: float = 1.0
    equilibration_steps: int = 50_000
    production_steps: int = 250_000
    report_interval_steps: int = 2_500
    checkpoint_interval_steps: int = 2_500
    replica_count: int = 3
    base_seed: int = 622
    position_restraint_schedule_kj_mol_nm2: tuple[float, ...] = (
        1_000.0,
        100.0,
        10.0,
        0.0,
    )
    zinc_restraint_k_kj_mol_nm2: float = 2_000.0
    minimization_tolerance_kj_mol_nm: float = 10.0
    minimization_max_iterations: int = 2_000
    platform_name: str = "CUDA"

    def __post_init__(self) -> None:
        for name in (
            "equilibration_steps",
            "production_steps",
            "report_interval_steps",
            "checkpoint_interval_steps",
            "replica_count",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not self.position_restraint_schedule_kj_mol_nm2:
            raise ValueError("A documented position-restraint schedule is required")
        if self.position_restraint_schedule_kj_mol_nm2[-1] != 0.0:
            raise ValueError("Production MD requires positional restraints to be released")


@dataclass(frozen=True)
class ReplicaPlan:
    replica_id: int
    seed: int
    equilibration_steps: int
    production_steps: int
    production_time_ps: float


@dataclass(frozen=True)
class ReplicaResult:
    replica_id: int
    seed: int
    status: str
    output_dir: Path
    trajectory_path: Path | None
    checkpoint_path: Path | None
    metrics_csv: Path | None
    final_pdb: Path | None
    summary_json: Path | None
    resumed: bool
    error: str


@dataclass(frozen=True)
class ReplicatedMDResult:
    system_label: str
    replicas: tuple[ReplicaResult, ...]
    manifest_json: Path


@dataclass(frozen=True)
class EnsembleSummary:
    completed_replicas: int
    invalid_replicas: int
    replica_summary_csv: Path
    summary_json: Path
