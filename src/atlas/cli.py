"""Atlas v1 command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from atlas.adaptive_pipeline import AdaptivePipelineConfig, run_adaptive_pipeline
from atlas.pipeline import PipelineConfig, run_pipeline
from atlas.preflight import run_preflight as run_environment_preflight
from atlas.structure.reconstruct import reconstruct_active_like
from atlas.validation.validation_gate import (
    evaluate_validation,
    require_validation_pass,
    write_validation_outputs,
)


app = typer.Typer(
    no_args_is_help=True,
    help="Validation-gated stability and catalytic-geometry screening for DP622.",
)


@app.command()
def preflight(
    input_path: Annotated[
        Path, typer.Option("--input", help="Committed 23WN mmCIF input.")
    ] = Path("data/23WN.cif"),
    atlas_repo: Annotated[
        Path, typer.Option(help="Atlas Git checkout to verify.")
    ] = Path("."),
    thermompnn_repo: Annotated[
        Path, typer.Option(help="Pinned official ThermoMPNN repository.")
    ] = Path(".external/ThermoMPNN"),
    thermompnn_d_repo: Annotated[
        Path, typer.Option(help="Pinned official ThermoMPNN-D repository.")
    ] = Path(".external/ThermoMPNN-D"),
    output_json: Annotated[
        Path, typer.Option(help="Machine-readable preflight report.")
    ] = Path("outputs/preflight.json"),
) -> None:
    """Fail fast on GPU, CUDA, checkout, import, or 23WN setup errors."""
    try:
        report = run_environment_preflight(
            input_path,
            atlas_repo,
            thermompnn_repo,
            thermompnn_d_repo,
        )
        report.write_json(output_json)
    except Exception as exc:
        typer.echo(f"Preflight failed: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Preflight passed on {report.gpu_name}")
    typer.echo(f"Free GPU memory: {report.gpu_memory_free_mib} MiB")
    typer.echo(f"Atlas commit: {report.atlas_commit}")
    typer.echo(f"ThermoMPNN commit: {report.thermompnn_commit}")
    typer.echo(f"ThermoMPNN-D commit: {report.thermompnn_d_commit}")
    typer.echo(f"Report: {output_json}")


@app.command()
def reconstruct(
    input_path: Annotated[
        Path, typer.Option("--input", help="Committed 23WN mmCIF/PDB input.")
    ] = Path("data/23WN.cif"),
    output_dir: Annotated[
        Path, typer.Option(help="Directory for structural-only outputs.")
    ] = Path("outputs/reconstruction"),
) -> None:
    """Build the active-like DP622–Aβ reconstruction without model dependencies."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result = reconstruct_active_like(
        input_path,
        output_dir / "DP622_active_like_reconstruction.pdb",
        output_dir / "residue_numbering_map.csv",
    )
    typer.echo(f"Active-like reconstruction: {result.structure_path}")
    typer.echo(f"Residue map: {result.numbering_map_path}")


@app.command()
def validate(
    stability_csv: Annotated[Path, typer.Option(help="Normalized stability CSV.")],
    geometry_csv: Annotated[Path, typer.Option(help="Catalytic geometry CSV.")],
    output_dir: Annotated[
        Path, typer.Option(help="Directory for gate CSV and Markdown report.")
    ] = Path("outputs/validation"),
    dynamics_csv: Annotated[
        Path | None, typer.Option(help="Optional OpenMM summary CSV.")
    ] = None,
) -> None:
    """Evaluate precomputed real scientific outputs through the hard gate."""
    stability = pd.read_csv(stability_csv)
    geometry = pd.read_csv(geometry_csv)
    dynamics = pd.read_csv(dynamics_csv) if dynamics_csv else None
    result = evaluate_validation(stability, geometry, dynamics)
    write_validation_outputs(result, output_dir)
    try:
        require_validation_pass(result)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(3) from exc
    typer.echo(f"Validation passed: {output_dir / 'validation_report.md'}")


@app.command()
def run(
    input_path: Annotated[
        Path, typer.Option("--input", help="Committed 23WN mmCIF/PDB input.")
    ] = Path("data/23WN.cif"),
    output_root: Annotated[
        Path, typer.Option(help="Parent for fresh, non-overwriting run directories.")
    ] = Path("outputs"),
    atlas_repo: Annotated[
        Path, typer.Option(help="Atlas Git checkout used for commit provenance.")
    ] = Path("."),
    thermompnn_repo: Annotated[
        Path, typer.Option(help="Pinned official ThermoMPNN repository.")
    ] = Path(".external/ThermoMPNN"),
    thermompnn_d_repo: Annotated[
        Path, typer.Option(help="Pinned official ThermoMPNN-D repository.")
    ] = Path(".external/ThermoMPNN-D"),
    dynamics_mode: Annotated[
        str,
        typer.Option(help="OpenMM stage: minimize, short-md, or skip."),
    ] = "minimize",
    run_id: Annotated[
        str | None,
        typer.Option(help="Stable run identifier required for resumable Colab stages."),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(help="Reuse only provenance-matched completed stage outputs."),
    ] = False,
    stop_after: Annotated[
        str | None,
        typer.Option(
            help="Stop after structure, thermompnn, thermompnn-d, geometry, dynamics, or validation."
        ),
    ] = None,
) -> None:
    """Run reconstruction, benchmarks, hard validation, then conditional design."""
    try:
        result = run_pipeline(
            PipelineConfig(
                input_structure=input_path,
                output_root=output_root,
                atlas_repo=atlas_repo,
                thermompnn_repo=thermompnn_repo,
                thermompnn_d_repo=thermompnn_d_repo,
                dynamics_mode=dynamics_mode,
                run_id=run_id,
                resume=resume,
                stop_after=stop_after,
            )
        )
    except Exception as exc:
        typer.echo(f"Atlas stopped: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Atlas status: {result.status}")
    typer.echo(f"Run directory: {result.run_dir}")
    if result.status == "completed":
        typer.echo(
            "Novel candidates are computational predictions requiring experimental validation."
        )


@app.command("adaptive-run")
def adaptive_run(
    input_path: Annotated[
        Path, typer.Option("--input", help="Committed 23WN mmCIF/PDB input.")
    ] = Path("data/23WN.cif"),
    output_root: Annotated[
        Path, typer.Option(help="Parent for provenance-bound adaptive run directories.")
    ] = Path("outputs"),
    atlas_repo: Annotated[
        Path, typer.Option(help="Atlas Git checkout used for commit provenance.")
    ] = Path("."),
    thermompnn_repo: Annotated[
        Path, typer.Option(help="Pinned official ThermoMPNN repository.")
    ] = Path(".external/ThermoMPNN"),
    thermompnn_d_repo: Annotated[
        Path, typer.Option(help="Pinned official ThermoMPNN-D repository.")
    ] = Path(".external/ThermoMPNN-D"),
    run_id: Annotated[
        str | None,
        typer.Option(help="Stable run identifier required for checkpoint/resume."),
    ] = None,
    candidate_budget: Annotated[
        int, typer.Option(help="Unique legal seed-search budget; minimum 5,000.")
    ] = 5_000,
    broad_target: Annotated[
        int, typer.Option(help="Maximum broad-screen survivors.")
    ] = 500,
    structure_target: Annotated[
        int, typer.Option(help="Maximum mutant-complex analyses.")
    ] = 100,
    md_target: Annotated[
        int, typer.Option(help="Maximum replicated explicit-solvent MD candidates.")
    ] = 20,
    adversarial_target: Annotated[
        int, typer.Option(help="Maximum blinded adversarial reviews.")
    ] = 10,
    portfolio_target: Annotated[
        int, typer.Option(help="Maximum experimentally untested finalists.")
    ] = 5,
    seed: Annotated[int, typer.Option(help="Deterministic adaptive search seed.")] = 622,
    resume: Annotated[
        bool, typer.Option(help="Reuse only exact-context completed checkpoints.")
    ] = False,
    stop_after: Annotated[
        str | None,
        typer.Option(
            help=(
                "Stop after setup, round1, round2, round3, broad, structure, repair, "
                "md, adversarial, or reports."
            )
        ),
    ] = None,
) -> None:
    """Run the prospective 5,000+ adaptive design and evidence funnel."""
    try:
        result = run_adaptive_pipeline(
            AdaptivePipelineConfig(
                input_structure=input_path,
                output_root=output_root,
                atlas_repo=atlas_repo,
                thermompnn_repo=thermompnn_repo,
                thermompnn_d_repo=thermompnn_d_repo,
                run_id=run_id,
                candidate_budget=candidate_budget,
                broad_target=broad_target,
                structure_target=structure_target,
                md_target=md_target,
                adversarial_target=adversarial_target,
                portfolio_target=portfolio_target,
                seed=seed,
                resume=resume,
                stop_after=stop_after,
            )
        )
    except Exception as exc:
        typer.echo(f"Adaptive Atlas stopped: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Adaptive Atlas status: {result.status}")
    typer.echo(f"Run directory: {result.run_dir}")
    typer.echo(
        f"Unique legal candidates evaluated: {result.completion_audit.unique_legal_evaluated}"
    )
    if result.finalist_ids:
        typer.echo("Finalists are EXPERIMENTALLY UNTESTED: " + ", ".join(result.finalist_ids))
