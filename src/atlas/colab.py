"""Fail-fast setup, readiness, and stage-execution helpers for Colab."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
from typing import Mapping, Sequence

from atlas.stability.upstream_execution import UpstreamPythonExecution


class StageExecutionError(RuntimeError):
    """A Colab production stage failed after its full evidence was printed."""


class ColabReadinessError(RuntimeError):
    """Cheap Colab boundaries failed before scientific stages could start."""


_SAFE_ENVIRONMENT_KEYS = (
    "COLAB_RELEASE_TAG",
    "CONDA_PREFIX",
    "CUDA_VISIBLE_DEVICES",
    "PATH",
    "PYTHONPATH",
    "VIRTUAL_ENV",
)

_DEFAULT_REQUIRED_MODULES = (
    "atlas",
    "atlas.cli",
    "atlas.colab",
    "atlas.pipeline",
    "atlas.structure.reconstruct",
    "atlas.stability.thermompnn_runner",
    "atlas.stability.thermompnn_d_runner",
    "Bio",
    "joblib",
    "matplotlib",
    "numpy",
    "omegaconf",
    "openmm",
    "pandas",
    "pytorch_lightning",
    "scipy",
    "sklearn",
    "torch",
    "torchmetrics",
    "tqdm",
    "typer",
    "wandb",
)

_ATLAS_LAYOUT = (
    "pyproject.toml",
    "src/atlas/__init__.py",
    "src/atlas/__main__.py",
    "src/atlas/cli.py",
    "src/atlas/pipeline.py",
    "data/23WN.cif",
)

_THERMOMPNN_LAYOUT = (
    ".git",
    "analysis/custom_inference.py",
    "local.yaml",
    "models/thermoMPNN_default.pt",
    "vanilla_model_weights/v_48_020.pt",
)

_THERMOMPNN_D_LAYOUT = (
    ".git",
    "v2_ssm.py",
    "examples/configs/local.yaml",
    "examples/configs/epistatic.yaml",
    "model_weights/ThermoMPNN-D-ens1.ckpt",
    "vanilla_model_weights/v_48_020.pt",
)

_THERMOMPNN_DIR_LINE = re.compile(
    r"^(?P<prefix>\s*thermompnn_dir\s*:\s*).*$", re.MULTILINE
)


def configure_upstream_runtime_paths(
    thermompnn_repo: str | Path,
    thermompnn_d_repo: str | Path,
) -> list[Path]:
    """Replace only documented local checkout paths in pinned upstream YAML."""

    configurations = [
        (Path(thermompnn_repo).resolve(), Path(thermompnn_repo) / "local.yaml"),
        (
            Path(thermompnn_d_repo).resolve(),
            Path(thermompnn_d_repo) / "examples/configs/local.yaml",
        ),
    ]
    changed: list[Path] = []
    for repository, config_path in configurations:
        config_path = config_path.resolve()
        if not config_path.is_file():
            raise ColabReadinessError(
                f"Upstream runtime configuration does not exist: {config_path}"
            )
        content = config_path.read_text()
        updated, count = _THERMOMPNN_DIR_LINE.subn(
            lambda match: f'{match.group("prefix")}"{repository}"',
            content,
        )
        if count != 1:
            raise ColabReadinessError(
                f"Expected exactly one platform.thermompnn_dir in {config_path}; "
                f"found {count}"
            )
        config_path.write_text(updated)
        changed.append(config_path)
    return changed


def _configured_upstream_root(config_path: Path) -> Path | None:
    if not config_path.is_file():
        return None
    match = _THERMOMPNN_DIR_LINE.search(config_path.read_text())
    if not match:
        return None
    value = match.group(0).split(":", 1)[1].strip().strip("\"'")
    return Path(value).expanduser().resolve() if value else None


def _print_stream(label: str, content: str) -> None:
    print(f"--- {label} ---")
    if content:
        print(content, end="" if content.endswith("\n") else "\n")
    else:
        print("<empty>")


def run_stage_command(
    stage_name: str,
    command: Sequence[str],
    cwd: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
    suggested_next_action: str = (
        "Read the first Atlas error above, resolve it, then rerun this stage. "
        "Completed provenance-matched checkpoints remain reusable."
    ),
) -> subprocess.CompletedProcess[str]:
    """Run one hard-fail stage and print complete evidence before raising."""

    working_directory = Path(cwd).resolve()
    effective_environment = dict(os.environ if environment is None else environment)
    exact_command = shlex.join(str(part) for part in command)
    print(f"\n=== {stage_name} ===", flush=True)
    print(f"Exact command: {exact_command}", flush=True)
    try:
        completed = subprocess.run(
            [str(part) for part in command],
            cwd=working_directory,
            env=effective_environment,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        completed = subprocess.CompletedProcess(
            [str(part) for part in command],
            127,
            stdout="",
            stderr=f"{type(exc).__name__}: {exc}\n",
        )
    if completed.returncode == 0:
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.stderr:
            _print_stream("stderr", completed.stderr)
        return completed

    print("\n=== ATLAS STAGE FAILED ===")
    print(f"Stage: {stage_name}")
    print(f"Exit status: {completed.returncode}")
    print(f"Working directory: {working_directory}")
    print(f"Exact command: {exact_command}")
    print("Relevant environment:")
    for key in _SAFE_ENVIRONMENT_KEYS:
        if key in effective_environment:
            print(f"  {key}={effective_environment[key]!r}")
    _print_stream("complete stdout", completed.stdout or "")
    _print_stream("complete stderr", completed.stderr or "")
    print(f"Suggested next action: {suggested_next_action}", flush=True)
    raise StageExecutionError(
        f"{stage_name} failed with exit status {completed.returncode}"
    )


def _probe_command(
    label: str,
    command: Sequence[str],
    cwd: Path,
    errors: list[str],
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    try:
        completed = subprocess.run(
            [str(part) for part in command],
            cwd=cwd,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        errors.append(f"{label} could not execute: {type(exc).__name__}: {exc}")
        return "failed"
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "no output").strip()
        errors.append(
            f"{label} failed with exit status {completed.returncode}: {detail}"
        )
        return "failed"
    return "passed"


def _bootstrap_probe_command(
    execution: UpstreamPythonExecution,
    expected_module: str,
    expected_file: Path,
    required_symbol: str,
    configuration_files: Sequence[Path],
) -> list[str]:
    probe = (
        "import importlib\n"
        "from pathlib import Path\n"
        "import runpy\n"
        "import sys\n"
        f"script = Path({str(execution.script)!r}).resolve()\n"
        "sys.path[0] = str(script.parent)\n"
        "runpy.run_path(str(script), run_name='_atlas_readiness_bootstrap')\n"
        f"module = importlib.import_module({expected_module!r})\n"
        "actual = Path(module.__file__).resolve()\n"
        f"expected = Path({str(expected_file)!r}).resolve()\n"
        "if actual != expected:\n"
        "    raise ImportError(\n"
        f"        {expected_module!r} + f' resolved from {{actual}}; expected {{expected}}'\n"
        "    )\n"
        f"getattr(module, {required_symbol!r})\n"
        "from omegaconf import OmegaConf\n"
        f"configs = {[str(path) for path in configuration_files]!r}\n"
        "OmegaConf.merge(*(OmegaConf.load(path) for path in configs))\n"
        "print(f'Bootstrap imported {module.__name__} from {actual}')\n"
    )
    return execution.python_command(["-c", probe])


def validate_colab_readiness(
    *,
    python_executable: str | Path,
    atlas_repo: str | Path,
    input_structure: str | Path,
    thermompnn_repo: str | Path,
    thermompnn_d_repo: str | Path,
    output_root: str | Path,
    run_dir: str | Path,
    required_modules: Sequence[str] = _DEFAULT_REQUIRED_MODULES,
) -> dict[str, object]:
    """Validate cheap runtime/layout boundaries immediately before Stage 3."""

    python = Path(python_executable).expanduser()
    if not python.is_absolute():
        python = Path.cwd() / python
    atlas = Path(atlas_repo).resolve()
    input_path = Path(input_structure).resolve()
    single = Path(thermompnn_repo).resolve()
    double = Path(thermompnn_d_repo).resolve()
    checkpoints = Path(output_root).resolve()
    execution = Path(run_dir).resolve()
    errors: list[str] = []

    if not python.is_file():
        errors.append(f"Python executable does not exist: {python}")
    for relative in _ATLAS_LAYOUT:
        if not (atlas / relative).exists():
            errors.append(f"Atlas repository layout is missing {atlas / relative}")
    if not input_path.is_file():
        errors.append(f"Required 23WN input does not exist: {input_path}")
    for relative in _THERMOMPNN_LAYOUT:
        if not (single / relative).exists():
            errors.append(f"ThermoMPNN required file is missing: {single / relative}")
    for relative in _THERMOMPNN_D_LAYOUT:
        if not (double / relative).exists():
            errors.append(f"ThermoMPNN-D required file is missing: {double / relative}")
    configured_single = _configured_upstream_root(single / "local.yaml")
    if configured_single != single:
        errors.append(
            "ThermoMPNN platform.thermompnn_dir must point to its checkout: "
            f"expected {single}, found {configured_single}"
        )
    configured_double = _configured_upstream_root(
        double / "examples/configs/local.yaml"
    )
    if configured_double != double:
        errors.append(
            "ThermoMPNN-D platform.thermompnn_dir must point to its checkout: "
            f"expected {double}, found {configured_double}"
        )

    output_writable = False
    if not checkpoints.is_dir():
        errors.append(f"Checkpoint root is not a writable directory: {checkpoints}")
    else:
        probe_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                prefix=".atlas-write-probe-",
                dir=checkpoints,
                delete=False,
            ) as handle:
                handle.write("Atlas Colab write probe\n")
                probe_path = Path(handle.name)
            probe_path.unlink()
            output_writable = True
        except OSError as exc:
            if probe_path and probe_path.exists():
                probe_path.unlink(missing_ok=True)
            errors.append(
                f"Checkpoint root is not writable: {checkpoints}: "
                f"{type(exc).__name__}: {exc}"
            )

    if execution.exists():
        context_path = execution / "run_context.json"
        if not execution.is_dir():
            errors.append(f"Checkpoint run path is not a directory: {execution}")
        elif not context_path.is_file():
            errors.append(
                f"Existing checkpoint has no run_context.json: {execution}. "
                "Move or remove this partial directory before retrying."
            )
        else:
            import json

            try:
                context = json.loads(context_path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                errors.append(f"Existing run_context.json is unreadable: {exc}")
            else:
                required_context = {
                    "atlas_commit",
                    "input_sha256",
                    "thermompnn_commit",
                    "thermompnn_d_commit",
                    "dynamics_mode",
                    "validation_policy",
                }
                missing_context = required_context.difference(context)
                if missing_context:
                    errors.append(
                        "Existing run_context.json is missing fields: "
                        + ", ".join(sorted(missing_context))
                    )

    cli_entrypoint = "not-run"
    import_probe = "not-run"
    thermompnn_import_probe = "not-run"
    thermompnn_d_import_probe = "not-run"
    if python.is_file() and atlas.is_dir():
        import_script = (
            "import importlib\n"
            f"modules = {list(required_modules)!r}\n"
            "for name in modules:\n"
            "    importlib.import_module(name)\n"
            "print('Imported:', ', '.join(modules))\n"
        )
        import_probe = _probe_command(
            "Required Python module import",
            [str(python), "-c", import_script],
            atlas,
            errors,
        )
        cli_entrypoint = _probe_command(
            "Atlas CLI entrypoint",
            [str(python), "-m", "atlas", "--help"],
            atlas,
            errors,
        )
        if (single / "analysis/custom_inference.py").is_file():
            single_execution = UpstreamPythonExecution.create(
                single,
                "analysis/custom_inference.py",
                python_executable=python,
            )
            thermompnn_import_probe = _probe_command(
                "ThermoMPNN inference bootstrap",
                _bootstrap_probe_command(
                    single_execution,
                    "datasets",
                    single / "datasets.py",
                    "Mutation",
                    (single / "local.yaml",),
                ),
                single_execution.cwd,
                errors,
                environment=single_execution.environment(),
            )
        if (double / "v2_ssm.py").is_file():
            double_execution = UpstreamPythonExecution.create(
                double,
                "v2_ssm.py",
                python_executable=python,
            )
            thermompnn_d_import_probe = _probe_command(
                "ThermoMPNN-D inference bootstrap",
                _bootstrap_probe_command(
                    double_execution,
                    "thermompnn.datasets.dataset_utils",
                    double / "thermompnn/datasets/dataset_utils.py",
                    "Mutation",
                    (
                        double / "examples/configs/local.yaml",
                        double / "examples/configs/epistatic.yaml",
                    ),
                ),
                double_execution.cwd,
                errors,
                environment=double_execution.environment(),
            )

    if errors:
        raise ColabReadinessError(
            "Colab readiness failed before Stage 3:\n- " + "\n- ".join(errors)
        )
    return {
        "passed": True,
        "atlas_imports": import_probe,
        "cli_entrypoint": cli_entrypoint,
        "thermompnn_imports": thermompnn_import_probe,
        "thermompnn_d_imports": thermompnn_d_import_probe,
        "repository_layout": "passed",
        "input_structure": str(input_path),
        "output_root": str(checkpoints),
        "output_writable": output_writable,
        "run_directory_state": "resumable" if execution.exists() else "new",
    }


def build_stage_command(
    *,
    python_executable: str,
    input_structure: str | Path,
    output_root: str | Path,
    atlas_repo: str | Path,
    thermompnn_repo: str | Path,
    thermompnn_d_repo: str | Path,
    run_id: str,
    dynamics_mode: str,
    stop_after: str | None = None,
    resume: bool = False,
) -> list[str]:
    """Build one staged invocation of the production Atlas CLI."""
    command = [
        python_executable,
        "-m",
        "atlas",
        "run",
        "--input",
        str(input_structure),
        "--output-root",
        str(output_root),
        "--atlas-repo",
        str(atlas_repo),
        "--thermompnn-repo",
        str(thermompnn_repo),
        "--thermompnn-d-repo",
        str(thermompnn_d_repo),
        "--dynamics-mode",
        dynamics_mode,
        "--run-id",
        run_id,
    ]
    if resume:
        command.append("--resume")
    if stop_after:
        command.extend(["--stop-after", stop_after])
    return command


def build_adaptive_stage_command(
    *,
    python_executable: str,
    input_structure: str | Path,
    output_root: str | Path,
    atlas_repo: str | Path,
    thermompnn_repo: str | Path,
    thermompnn_d_repo: str | Path,
    run_id: str,
    candidate_budget: int = 5_000,
    broad_target: int = 500,
    structure_target: int = 100,
    adversarial_target: int = 10,
    portfolio_target: int = 5,
    seed: int = 622,
    stop_after: str | None = None,
    resume: bool = False,
) -> list[str]:
    """Build one checkpointed prospective-design Colab invocation."""
    command = [
        python_executable,
        "-m",
        "atlas",
        "adaptive-run",
        "--input",
        str(input_structure),
        "--output-root",
        str(output_root),
        "--atlas-repo",
        str(atlas_repo),
        "--thermompnn-repo",
        str(thermompnn_repo),
        "--thermompnn-d-repo",
        str(thermompnn_d_repo),
        "--run-id",
        run_id,
        "--candidate-budget",
        str(candidate_budget),
        "--broad-target",
        str(broad_target),
        "--structure-target",
        str(structure_target),
        "--adversarial-target",
        str(adversarial_target),
        "--portfolio-target",
        str(portfolio_target),
        "--seed",
        str(seed),
    ]
    if resume:
        command.append("--resume")
    if stop_after:
        command.extend(["--stop-after", stop_after])
    return command
