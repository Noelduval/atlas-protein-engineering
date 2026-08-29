"""Checkpointed, replicated explicit-solvent OpenMM structural dynamics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from atlas.dynamics.models import (
    ExplicitMDConfig,
    ReplicaPlan,
    ReplicaResult,
    ReplicatedMDResult,
)


FORCE_FIELDS = ("amber14-all.xml", "amber14/tip3pfb.xml")
DYNAMICS_PROTOCOL = "atlas-explicit-md-v2-staged-timestep"
EQUILIBRATION_TIMESTEP_FS = 0.5
CLAIM_BOUNDARY = (
    "Structural/dynamic simulation with explicit solvent and documented Zn restraints; "
    "does not simulate catalytic turnover or predict kcat/Km."
)


@dataclass
class PreparedExplicitSystem:
    topology: Any
    positions: Any
    system: Any
    prepared_pdb: Path
    metadata_json: Path
    atom_count: int
    solute_atom_count: int
    water_residue_count: int
    periodic: bool
    context_hash: str


@dataclass(frozen=True)
class _MetricContext:
    reference_positions_a: np.ndarray
    protein_ca: np.ndarray
    substrate_heavy: np.ndarray
    active_site_heavy: np.ndarray
    selected_for_rmsf: np.ndarray
    contact_pairs: np.ndarray
    atom_indices: dict[str, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _context_hash(source: Path, config: ExplicitMDConfig, seed: int) -> str:
    payload = {
        "source_sha256": _sha256(source),
        "config": asdict(config),
        "seed": seed,
        "force_fields": FORCE_FIELDS,
        "protocol": "atlas-explicit-md-v1",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_replica_plan(config: ExplicitMDConfig) -> tuple[ReplicaPlan, ...]:
    return tuple(
        ReplicaPlan(
            replica_id=index + 1,
            seed=config.base_seed + 7_919 * index,
            equilibration_steps=config.equilibration_steps,
            production_steps=config.production_steps,
            production_time_ps=config.production_steps * config.timestep_fs / 1_000.0,
        )
        for index in range(config.replica_count)
    )


def _equilibration_timestep_fs(config: ExplicitMDConfig) -> float:
    return min(EQUILIBRATION_TIMESTEP_FS, config.timestep_fs)


def build_equilibration_plan(
    config: ExplicitMDConfig,
) -> tuple[tuple[float, int], ...]:
    """Preserve the requested equilibration duration at a stable metal-site timestep."""
    schedule = config.position_restraint_schedule_kj_mol_nm2
    base_steps, remainder = divmod(config.equilibration_steps, len(schedule))
    scale = config.timestep_fs / _equilibration_timestep_fs(config)
    plan: list[tuple[float, int]] = []
    for segment, force_constant in enumerate(schedule):
        nominal_steps = base_steps + (1 if segment < remainder else 0)
        executed_steps = round(nominal_steps * scale)
        if abs(executed_steps - nominal_steps * scale) > 1e-9:
            raise ValueError("Equilibration timestep does not produce an integral step plan")
        plan.append((force_constant, executed_steps))
    return tuple(plan)


def _replica_context_hash(preparation_context_hash: str, config: ExplicitMDConfig) -> str:
    payload = {
        "preparation_context_hash": preparation_context_hash,
        "config": asdict(config),
        "dynamics_protocol": DYNAMICS_PROTOCOL,
        "equilibration_timestep_fs": _equilibration_timestep_fs(config),
        "equilibration_plan": build_equilibration_plan(config),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_openmm():
    import openmm
    from openmm import app, unit

    return openmm, app, unit


def _topology_index(topology, chain: str, residue: int, atom_name: str) -> int:
    matches = [
        atom.index
        for atom in topology.atoms()
        if atom.residue.chain.id == chain
        and int(atom.residue.id) == residue
        and atom.name == atom_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one topology atom {chain}:{residue}:{atom_name}, found {len(matches)}"
        )
    return matches[0]


def _add_position_restraints(openmm, unit, system, topology, positions, solute_atoms: int):
    force = openmm.CustomExternalForce(
        "0.5*k_position*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
    )
    force.addGlobalParameter(
        "k_position",
        0.0 * unit.kilojoule_per_mole / unit.nanometer**2,
    )
    for parameter in ("x0", "y0", "z0"):
        force.addPerParticleParameter(parameter)
    for atom in topology.atoms():
        if atom.index >= solute_atoms:
            break
        if atom.element is None or atom.element.symbol == "H":
            continue
        coordinates = positions[atom.index].value_in_unit(unit.nanometer)
        force.addParticle(atom.index, [float(value) for value in coordinates])
    force.setForceGroup(20)
    system.addForce(force)


def _add_zinc_restraints(openmm, unit, system, topology, positions, force_constant: float):
    zinc = _topology_index(topology, "C", 1601, "ZN")
    target_keys = (
        ("A", 95, "NE2"),
        ("A", 99, "NE2"),
        ("A", 122, "OE1"),
        ("A", 122, "OE2"),
        ("B", 38, "O"),
    )
    force = openmm.HarmonicBondForce()
    targets: dict[str, float] = {}
    zinc_position = positions[zinc]
    for chain, residue, atom_name in target_keys:
        index = _topology_index(topology, chain, residue, atom_name)
        delta = (positions[index] - zinc_position).value_in_unit(unit.nanometer)
        distance_nm = float(sum(float(value) ** 2 for value in delta) ** 0.5)
        force.addBond(
            zinc,
            index,
            distance_nm * unit.nanometer,
            force_constant
            * unit.kilojoule_per_mole
            / unit.nanometer**2,
        )
        targets[f"{chain}:{residue}:{atom_name}"] = distance_nm * 10.0
    force.setForceGroup(21)
    system.addForce(force)
    return targets


def prepare_explicit_system(
    pdb_path: str | Path,
    output_dir: str | Path,
    config: ExplicitMDConfig,
    *,
    seed: int,
) -> PreparedExplicitSystem:
    """Prepare or safely reopen one explicit-solvent system."""
    source, destination = Path(pdb_path), Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    prepared_pdb = destination / "prepared_solvated.pdb"
    metadata_json = destination / "preparation.json"
    context_hash = _context_hash(source, config, seed)
    openmm, app, unit = _load_openmm()
    forcefield = app.ForceField(*FORCE_FIELDS)

    if prepared_pdb.is_file() and metadata_json.is_file():
        metadata = json.loads(metadata_json.read_text())
        if metadata.get("context_hash") != context_hash:
            raise RuntimeError("Refusing explicit-MD checkpoint with mismatched context")
        pdb = app.PDBFile(str(prepared_pdb))
        topology, positions = pdb.topology, pdb.positions
        solute_atom_count = int(metadata["solute_atom_count"])
        water_count = int(metadata["water_residue_count"])
    else:
        from pdbfixer import PDBFixer

        fixer = PDBFixer(filename=str(source))
        fixer.findMissingResidues()
        fixer.missingResidues = {}
        fixer.findMissingAtoms()
        modeled_terminals = sorted(
            f"{residue.chain.id}:{residue.id}:{atom_name}"
            for residue, atom_names in fixer.missingTerminals.items()
            for atom_name in atom_names
        )
        fixer.addMissingAtoms(seed=seed)
        fixer.addMissingHydrogens(config.ph)
        modeller = app.Modeller(fixer.topology, fixer.positions)
        solute_atom_count = modeller.topology.getNumAtoms()
        modeller.addSolvent(
            forcefield,
            model="tip3p",
            padding=config.solvent_padding_nm * unit.nanometer,
            ionicStrength=config.ionic_strength_molar * unit.molar,
            neutralize=True,
        )
        topology, positions = modeller.topology, modeller.positions
        water_count = sum(
            1 for residue in topology.residues() if residue.name in {"HOH", "WAT"}
        )
        with prepared_pdb.open("w") as handle:
            app.PDBFile.writeFile(topology, positions, handle, keepIds=True)
        metadata = {
            "context_hash": context_hash,
            "source_pdb": str(source),
            "source_sha256": _sha256(source),
            "prepared_pdb": str(prepared_pdb),
            "force_fields": list(FORCE_FIELDS),
            "water_model": "TIP3P-FB force-field template with Modeller TIP3P geometry",
            "ph": config.ph,
            "ionic_strength_molar": config.ionic_strength_molar,
            "solvent_padding_nm": config.solvent_padding_nm,
            "solute_atom_count": solute_atom_count,
            "total_atom_count": topology.getNumAtoms(),
            "water_residue_count": water_count,
            "modeled_fragment_terminal_atoms": modeled_terminals,
            "fragment_termination_limitation": (
                "OXT and terminal hydrogens make the deposited chain fragments chemically "
                "parameterizable; these atoms are modeled simulation preparation, not deposited."
            ),
            "seed": seed,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    system = forcefield.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=config.nonbonded_cutoff_nm * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
    )
    barostat = openmm.MonteCarloBarostat(
        config.pressure_bar * unit.bar,
        config.temperature_k * unit.kelvin,
        25,
    )
    barostat.setRandomNumberSeed(seed)
    system.addForce(barostat)
    _add_position_restraints(
        openmm, unit, system, topology, positions, solute_atom_count
    )
    zinc_targets = _add_zinc_restraints(
        openmm,
        unit,
        system,
        topology,
        positions,
        config.zinc_restraint_k_kj_mol_nm2,
    )
    metadata = json.loads(metadata_json.read_text())
    if metadata.get("zinc_restraint_targets_a") != zinc_targets:
        metadata["zinc_restraint_targets_a"] = zinc_targets
        metadata["zinc_restraint_k_kj_mol_nm2"] = config.zinc_restraint_k_kj_mol_nm2
        metadata["position_restraint_schedule_kj_mol_nm2"] = list(
            config.position_restraint_schedule_kj_mol_nm2
        )
        metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return PreparedExplicitSystem(
        topology=topology,
        positions=positions,
        system=system,
        prepared_pdb=prepared_pdb,
        metadata_json=metadata_json,
        atom_count=topology.getNumAtoms(),
        solute_atom_count=solute_atom_count,
        water_residue_count=water_count,
        periodic=topology.getPeriodicBoxVectors() is not None,
        context_hash=context_hash,
    )


def _metric_context(topology, reference_positions, unit) -> _MetricContext:
    reference = np.asarray(reference_positions.value_in_unit(unit.angstrom), dtype=float)
    protein_ca, substrate, active, selected = [], [], [], []
    atom_indices: dict[str, int] = {}
    for atom in topology.atoms():
        chain, residue = atom.residue.chain.id, int(atom.residue.id)
        element = None if atom.element is None else atom.element.symbol
        if chain == "A" and atom.name == "CA":
            protein_ca.append(atom.index)
        if chain == "B" and element != "H":
            substrate.append(atom.index)
        if chain == "A" and residue in {91, 95, 96, 99, 122, 126, 172} and element != "H":
            active.append(atom.index)
        key = f"{chain}:{residue}:{atom.name}"
        if key in {
            "C:1601:ZN",
            "A:95:NE2",
            "A:99:NE2",
            "A:122:OE1",
            "A:122:OE2",
            "A:96:OE1",
            "A:96:OE2",
            "A:172:NE2",
            "B:38:C",
            "B:38:O",
        }:
            atom_indices[key] = atom.index
    selected = sorted(set(active + substrate))
    protein_heavy = [
        atom.index
        for atom in topology.atoms()
        if atom.residue.chain.id == "A"
        and atom.element is not None
        and atom.element.symbol != "H"
    ]
    protein_array, substrate_array = np.asarray(protein_heavy), np.asarray(substrate)
    distances = np.linalg.norm(
        reference[protein_array, None, :] - reference[substrate_array, :][None, :, :],
        axis=2,
    )
    contacts = np.argwhere(distances <= 4.0)
    contact_pairs = np.column_stack(
        (protein_array[contacts[:, 0]], substrate_array[contacts[:, 1]])
    )
    return _MetricContext(
        reference_positions_a=reference,
        protein_ca=np.asarray(protein_ca),
        substrate_heavy=np.asarray(substrate),
        active_site_heavy=np.asarray(active),
        selected_for_rmsf=np.asarray(selected),
        contact_pairs=contact_pairs,
        atom_indices=atom_indices,
    )


def _align(current: np.ndarray, reference: np.ndarray, indices: np.ndarray) -> np.ndarray:
    mobile = current[indices]
    target = reference[indices]
    mobile_center, target_center = mobile.mean(axis=0), target.mean(axis=0)
    covariance = (mobile - mobile_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    return (current - mobile_center) @ rotation + target_center


def _distance(positions: np.ndarray, left: int, right: int) -> float:
    return float(np.linalg.norm(positions[left] - positions[right]))


def _snapshot_metrics(
    positions_a: np.ndarray,
    potential_kj_mol: float,
    context: _MetricContext,
    production_step: int,
    timestep_fs: float,
) -> tuple[dict[str, float | int], np.ndarray]:
    aligned = _align(positions_a, context.reference_positions_a, context.protein_ca)
    reference = context.reference_positions_a

    def rmsd(indices: np.ndarray) -> float:
        delta = aligned[indices] - reference[indices]
        return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))

    index = context.atom_indices
    zinc = index["C:1601:ZN"]
    e122_distance = min(
        _distance(positions_a, zinc, index["A:122:OE1"]),
        _distance(positions_a, zinc, index["A:122:OE2"]),
    )
    e96_distance = min(
        _distance(positions_a, index["A:96:OE1"], index["B:38:C"]),
        _distance(positions_a, index["A:96:OE2"], index["B:38:C"]),
    )
    contact_distances = np.linalg.norm(
        positions_a[context.contact_pairs[:, 0]]
        - positions_a[context.contact_pairs[:, 1]],
        axis=1,
    )
    substrate_centroid_drift = float(
        np.linalg.norm(
            aligned[context.substrate_heavy].mean(axis=0)
            - reference[context.substrate_heavy].mean(axis=0)
        )
    )
    metrics: dict[str, float | int] = {
        "production_step": production_step,
        "time_ps": production_step * timestep_fs / 1_000.0,
        "potential_energy_kj_mol": potential_kj_mol,
        "protein_ca_rmsd_a": rmsd(context.protein_ca),
        "active_site_rmsd_a": rmsd(context.active_site_heavy),
        "substrate_rmsd_a": rmsd(context.substrate_heavy),
        "substrate_centroid_drift_a": substrate_centroid_drift,
        "contact_fraction": float(np.mean(contact_distances <= 4.5)),
        "zn_h95_distance_a": _distance(positions_a, zinc, index["A:95:NE2"]),
        "zn_h99_distance_a": _distance(positions_a, zinc, index["A:99:NE2"]),
        "zn_e122_distance_a": e122_distance,
        "zn_scissile_o_distance_a": _distance(
            positions_a, zinc, index["B:38:O"]
        ),
        "e96_scissile_carbon_distance_a": e96_distance,
        "h172_scissile_o_distance_a": _distance(
            positions_a, index["A:172:NE2"], index["B:38:O"]
        ),
    }
    return metrics, aligned[context.selected_for_rmsf]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _completed_result(directory: Path, replica_id: int, seed: int) -> ReplicaResult | None:
    summary = directory / "replica_summary.json"
    if not summary.is_file():
        return None
    payload = json.loads(summary.read_text())
    if (
        payload.get("status") != "completed"
        or payload.get("dynamics_protocol") != DYNAMICS_PROTOCOL
    ):
        return None
    return ReplicaResult(
        replica_id=replica_id,
        seed=seed,
        status="completed",
        output_dir=directory,
        trajectory_path=directory / "trajectory.dcd",
        checkpoint_path=directory / "checkpoint.chk",
        metrics_csv=directory / "metrics.csv",
        final_pdb=directory / "final.pdb",
        summary_json=summary,
        resumed=True,
        error="",
    )


def run_explicit_md_replica(
    pdb_path: str | Path,
    output_dir: str | Path,
    config: ExplicitMDConfig,
    *,
    replica_id: int,
    seed: int,
) -> ReplicaResult:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    completed = _completed_result(directory, replica_id, seed)
    if completed is not None:
        return completed
    checkpoint = directory / "checkpoint.chk"
    progress_path = directory / "progress.json"
    trajectory = directory / "trajectory.dcd"
    metrics_path = directory / "metrics.csv"
    coordinates_path = directory / "analysis_coordinates.npz"
    final_pdb = directory / "final.pdb"
    summary_path = directory / "replica_summary.json"
    resumed = checkpoint.is_file() and progress_path.is_file()
    existing_progress = json.loads(progress_path.read_text()) if resumed else None
    try:
        openmm, app, unit = _load_openmm()
        prepared = prepare_explicit_system(
            pdb_path, directory / "prepared", config, seed=seed
        )
        starting_timestep_fs = (
            config.timestep_fs
            if existing_progress is not None
            and existing_progress.get("phase") == "production"
            else _equilibration_timestep_fs(config)
        )
        integrator = openmm.LangevinMiddleIntegrator(
            config.temperature_k * unit.kelvin,
            config.friction_per_ps / unit.picosecond,
            starting_timestep_fs * unit.femtoseconds,
        )
        integrator.setRandomNumberSeed(seed)
        platform = openmm.Platform.getPlatformByName(config.platform_name)
        properties = {"Precision": "mixed"} if config.platform_name == "CUDA" else {}
        simulation = app.Simulation(
            prepared.topology,
            prepared.system,
            integrator,
            platform,
            properties,
        )
        replica_context_hash = _replica_context_hash(prepared.context_hash, config)
        if resumed:
            progress = existing_progress
            if progress.get("context_hash") != replica_context_hash:
                raise RuntimeError("Replica checkpoint context does not match preparation")
            simulation.loadCheckpoint(str(checkpoint))
        else:
            progress = {
                "context_hash": replica_context_hash,
                "phase": "equilibration",
                "equilibration_segment": 0,
                "equilibration_steps_completed": 0,
                "production_steps_completed": 0,
            }
            simulation.context.setPositions(prepared.positions)
            simulation.minimizeEnergy(
                tolerance=config.minimization_tolerance_kj_mol_nm
                * unit.kilojoule_per_mole
                / unit.nanometer,
                maxIterations=config.minimization_max_iterations,
            )
            simulation.context.setVelocitiesToTemperature(
                config.temperature_k * unit.kelvin, seed
            )

        equilibration_plan = build_equilibration_plan(config)
        if progress["phase"] == "equilibration":
            for segment in range(
                int(progress["equilibration_segment"]), len(equilibration_plan)
            ):
                force_constant, steps = equilibration_plan[segment]
                simulation.context.setParameter(
                    "k_position",
                    force_constant
                    * unit.kilojoule_per_mole
                    / unit.nanometer**2,
                )
                simulation.step(steps)
                progress["equilibration_segment"] = segment + 1
                progress["equilibration_steps_completed"] = int(
                    progress["equilibration_steps_completed"]
                ) + steps
                simulation.saveCheckpoint(str(checkpoint))
                _write_json(progress_path, progress)
            progress["phase"] = "production"
            progress["production_steps_completed"] = 0
            integrator.setStepSize(config.timestep_fs * unit.femtoseconds)
            simulation.context.setParameter(
                "k_position", 0.0 * unit.kilojoule_per_mole / unit.nanometer**2
            )
            simulation.saveCheckpoint(str(checkpoint))
            _write_json(progress_path, progress)

        metric_context = _metric_context(
            prepared.topology, prepared.positions, unit
        )
        existing_metrics = (
            pd.read_csv(metrics_path).to_dict("records") if metrics_path.is_file() else []
        )
        if coordinates_path.is_file():
            coordinate_snapshots = list(
                np.load(coordinates_path)["aligned_selected_positions_a"]
            )
        else:
            coordinate_snapshots = []
        simulation.reporters.append(
            app.DCDReporter(
                str(trajectory),
                config.report_interval_steps,
                append=trajectory.is_file(),
                enforcePeriodicBox=False,
            )
        )
        state_log = directory / "state.csv"
        simulation.reporters.append(
            app.StateDataReporter(
                str(state_log),
                config.report_interval_steps,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                temperature=True,
                volume=True,
                density=True,
                speed=True,
                separator=",",
                append=state_log.is_file(),
            )
        )
        completed_steps = int(progress["production_steps_completed"])
        while completed_steps < config.production_steps:
            steps = min(
                config.report_interval_steps,
                config.production_steps - completed_steps,
            )
            simulation.step(steps)
            completed_steps += steps
            state = simulation.context.getState(getPositions=True, getEnergy=True)
            positions_a = np.asarray(
                state.getPositions(asNumpy=True).value_in_unit(unit.angstrom),
                dtype=float,
            )
            potential = float(
                state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            )
            metrics, aligned_selected = _snapshot_metrics(
                positions_a,
                potential,
                metric_context,
                completed_steps,
                config.timestep_fs,
            )
            existing_metrics.append(metrics)
            coordinate_snapshots.append(aligned_selected)
            pd.DataFrame(existing_metrics).to_csv(metrics_path, index=False)
            np.savez_compressed(
                coordinates_path,
                aligned_selected_positions_a=np.asarray(coordinate_snapshots),
                selected_atom_indices=metric_context.selected_for_rmsf,
            )
            progress["production_steps_completed"] = completed_steps
            simulation.saveCheckpoint(str(checkpoint))
            _write_json(progress_path, progress)

        final_state = simulation.context.getState(getPositions=True, getEnergy=True)
        with final_pdb.open("w") as handle:
            app.PDBFile.writeFile(
                prepared.topology,
                final_state.getPositions(),
                handle,
                keepIds=True,
            )
        progress["phase"] = "completed"
        _write_json(progress_path, progress)
        summary = {
            "status": "completed",
            "dynamics_protocol": DYNAMICS_PROTOCOL,
            "replica_id": replica_id,
            "seed": seed,
            "context_hash": replica_context_hash,
            "preparation_context_hash": prepared.context_hash,
            "equilibration_nominal_steps": config.equilibration_steps,
            "equilibration_executed_steps": sum(
                steps for _, steps in equilibration_plan
            ),
            "equilibration_timestep_fs": _equilibration_timestep_fs(config),
            "equilibration_time_ps": (
                config.equilibration_steps * config.timestep_fs / 1_000.0
            ),
            "production_steps": config.production_steps,
            "production_timestep_fs": config.timestep_fs,
            "production_time_ps": config.production_steps
            * config.timestep_fs
            / 1_000.0,
            "metric_snapshots": len(existing_metrics),
            "platform": config.platform_name,
            "trajectory": str(trajectory),
            "checkpoint": str(checkpoint),
            "metrics": str(metrics_path),
            "preparation": str(prepared.metadata_json),
            "claim_boundary": CLAIM_BOUNDARY,
        }
        _write_json(summary_path, summary)
        del simulation, integrator
        return ReplicaResult(
            replica_id=replica_id,
            seed=seed,
            status="completed",
            output_dir=directory,
            trajectory_path=trajectory,
            checkpoint_path=checkpoint,
            metrics_csv=metrics_path,
            final_pdb=final_pdb,
            summary_json=summary_path,
            resumed=resumed,
            error="",
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        _write_json(
            summary_path,
            {
                "status": "invalid_simulation",
                "dynamics_protocol": DYNAMICS_PROTOCOL,
                "replica_id": replica_id,
                "seed": seed,
                "error": error,
                "claim_boundary": "Unavailable evidence; no favorable value was imputed.",
            },
        )
        return ReplicaResult(
            replica_id=replica_id,
            seed=seed,
            status="invalid_simulation",
            output_dir=directory,
            trajectory_path=trajectory if trajectory.is_file() else None,
            checkpoint_path=checkpoint if checkpoint.is_file() else None,
            metrics_csv=metrics_path if metrics_path.is_file() else None,
            final_pdb=None,
            summary_json=summary_path,
            resumed=resumed,
            error=error,
        )


def run_replicated_explicit_md(
    pdb_path: str | Path,
    output_dir: str | Path,
    config: ExplicitMDConfig,
    *,
    system_label: str,
    replica_runner: Callable[..., ReplicaResult] = run_explicit_md_replica,
) -> ReplicatedMDResult:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    results = tuple(
        replica_runner(
            pdb_path,
            destination / f"replica-{plan.replica_id:02d}",
            config,
            replica_id=plan.replica_id,
            seed=plan.seed,
        )
        for plan in build_replica_plan(config)
    )
    manifest = destination / "replicated_md_manifest.json"
    _write_json(
        manifest,
        {
            "system_label": system_label,
            "input_pdb": str(pdb_path),
            "protocol": asdict(config),
            "effective_integration": {
                "dynamics_protocol": DYNAMICS_PROTOCOL,
                "equilibration_timestep_fs": _equilibration_timestep_fs(config),
                "equilibration_executed_steps": sum(
                    steps for _, steps in build_equilibration_plan(config)
                ),
                "production_timestep_fs": config.timestep_fs,
            },
            "replicas": [
                {
                    "replica_id": result.replica_id,
                    "seed": result.seed,
                    "status": result.status,
                    "output_dir": str(result.output_dir),
                    "error": result.error,
                }
                for result in results
            ],
            "claim_boundary": CLAIM_BOUNDARY,
        },
    )
    return ReplicatedMDResult(system_label, results, manifest)
