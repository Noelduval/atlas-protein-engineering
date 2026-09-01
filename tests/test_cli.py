from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from atlas.cli import app


runner = CliRunner()


def test_reconstruct_command_runs_without_external_models(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "reconstruct",
            "--input",
            "data/23WN.cif",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "DP622_active_like_reconstruction.pdb").is_file()
    assert (tmp_path / "residue_numbering_map.csv").is_file()


def test_run_missing_models_exits_nonzero_with_action(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--input",
            "data/23WN.cif",
            "--output-root",
            str(tmp_path),
            "--thermompnn-repo",
            str(tmp_path / "missing-single"),
            "--thermompnn-d-repo",
            str(tmp_path / "missing-double"),
            "--dynamics-mode",
            "skip",
        ],
    )
    assert result.exit_code != 0
    assert "repository" in result.output.lower()


def test_run_can_stop_after_structure_and_resume_without_models(tmp_path: Path) -> None:
    args = [
        "run",
        "--input",
        "data/23WN.cif",
        "--output-root",
        str(tmp_path),
        "--atlas-repo",
        str(Path.cwd()),
        "--run-id",
        "colab-checkpoint",
        "--dynamics-mode",
        "skip",
        "--stop-after",
        "structure",
    ]
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    assert "stopped_after_structure" in first.output
    resumed = runner.invoke(app, [*args, "--resume"])
    assert resumed.exit_code == 0, resumed.output
    assert (tmp_path / "colab-checkpoint" / "run_context.json").is_file()


def test_adaptive_run_can_checkpoint_setup_without_model_repositories(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "adaptive-run",
            "--input",
            "data/23WN.cif",
            "--output-root",
            str(tmp_path),
            "--atlas-repo",
            str(Path.cwd()),
            "--run-id",
            "adaptive-setup",
            "--stop-after",
            "setup",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "stopped_after_setup" in result.output
    run_dir = tmp_path / "adaptive-setup"
    assert (run_dir / "design_space.csv").is_file()
    assert (run_dir / "design_memory" / "atlas_science.sqlite").is_file()


def test_late_stage_command_consumes_existing_run(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run_late_stage(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            status="completed",
            finalist_ids=("ATLAS-ONE",),
            summary_path=tmp_path / "late_stage" / "completion.json",
        )

    monkeypatch.setattr("atlas.cli.run_late_stage", fake_run_late_stage)
    result = runner.invoke(
        app,
        [
            "late-stage",
            "--run-dir",
            str(tmp_path),
            "--input",
            "data/23WN.cif",
            "--seed",
            "622",
            "--resume",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "run_dir": tmp_path,
        "input_structure": Path("data/23WN.cif"),
        "seed": 622,
        "resume": True,
    }
    assert "ATLAS-ONE" in result.output
