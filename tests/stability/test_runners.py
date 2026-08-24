from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from atlas.stability.common import (
    DependencyUnavailableError,
    ScientificOutputError,
    StabilityVariant,
)
from atlas.stability.thermompnn_d_runner import ThermoMPNNDRunner
from atlas.stability.thermompnn_runner import ThermoMPNNRunner


SINGLE = StabilityVariant("Y91F", "Y91F", "Y115F")
DOUBLE = StabilityVariant("Y91F_D126A", "Y91F:D126A", "Y115F:D150A")


def _repo(tmp_path: Path, script: str) -> Path:
    repo = tmp_path / "repo"
    (repo / Path(script).parent).mkdir(parents=True)
    (repo / script).write_text("# test boundary\n")
    return repo


def test_missing_thermompnn_repository_is_actionable(tmp_path: Path) -> None:
    runner = ThermoMPNNRunner(tmp_path / "missing")
    with pytest.raises(DependencyUnavailableError, match="ThermoMPNN repository"):
        runner.run(tmp_path / "input.pdb", [SINGLE], tmp_path / "scores")


def test_nonzero_external_exit_is_preserved(tmp_path: Path) -> None:
    repo = _repo(tmp_path, "analysis/custom_inference.py")

    def fail(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 2, "stdout", "model failed")

    runner = ThermoMPNNRunner(repo, command_runner=fail)
    with pytest.raises(ScientificOutputError, match="model failed"):
        runner.run(tmp_path / "input.pdb", [SINGLE], tmp_path / "scores")


def test_thermompnn_prefers_checkout_datasets_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path, "analysis/custom_inference.py")
    conflicting = tmp_path / "site-packages"
    (conflicting / "datasets").mkdir(parents=True)
    (conflicting / "datasets/__init__.py").write_text("# unrelated package\n")
    (repo / "datasets.py").write_text("class Mutation:\n    pass\n")
    (repo / "analysis/custom_inference.py").write_text(
        "import argparse\n"
        "import csv\n"
        "import os\n"
        "import sys\n"
        "root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))\n"
        "sys.path.append(root)\n"
        "from datasets import Mutation\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--pdb')\n"
        "parser.add_argument('--chain')\n"
        "parser.add_argument('--out_dir')\n"
        "args = parser.parse_args()\n"
        "output = os.path.join(args.out_dir, 'ThermoMPNN_inference_input.csv')\n"
        "os.makedirs(args.out_dir, exist_ok=True)\n"
        "with open(output, 'w', newline='') as handle:\n"
        "    writer = csv.DictWriter(handle, fieldnames=[\n"
        "        'ddG_pred', 'position', 'wildtype', 'mutation'\n"
        "    ])\n"
        "    writer.writeheader()\n"
        "    writer.writerow({\n"
        "        'ddG_pred': -0.42, 'position': 90,\n"
        "        'wildtype': 'Y', 'mutation': 'F'\n"
        "    })\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(conflicting))

    scores = ThermoMPNNRunner(repo).run(
        tmp_path / "input.pdb", [SINGLE], tmp_path / "scores"
    )

    assert scores.loc[0, "predicted_ddg_or_score"] == pytest.approx(-0.42)


def test_thermompnn_normalizes_real_shaped_output(tmp_path: Path) -> None:
    repo = _repo(tmp_path, "analysis/custom_inference.py")

    def succeed(command, **kwargs):
        out_dir = Path(command[command.index("--out_dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{"ddG_pred": -0.42, "position": 90, "wildtype": "Y", "mutation": "F"}]
        ).to_csv(out_dir / "ThermoMPNN_inference_input.csv", index=False)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    scores = ThermoMPNNRunner(repo, command_runner=succeed).run(
        tmp_path / "input.pdb", [SINGLE], tmp_path / "scores"
    )
    assert scores.loc[0, "variant_id"] == "Y91F"
    assert scores.loc[0, "predicted_ddg_or_score"] == pytest.approx(-0.42)
    assert scores.loc[0, "model_used"] == "ThermoMPNN"
    assert "stability" in scores.loc[0, "interpretation"].lower()


def test_thermompnn_maps_zero_based_upstream_positions_to_dp622_numbering(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "ThermoMPNN_inference_input.csv"
    pd.DataFrame(
        [{"ddG_pred": -0.42, "position": 90, "wildtype": "Y", "mutation": "F"}]
    ).to_csv(raw, index=False)

    scores = ThermoMPNNRunner(tmp_path).normalize_existing(
        raw, [SINGLE], tmp_path / "normalized"
    )

    assert scores.loc[0, "variant_id"] == "Y91F"
    assert scores.loc[0, "predicted_ddg_or_score"] == pytest.approx(-0.42)


def test_thermompnn_rejects_missing_columns(tmp_path: Path) -> None:
    repo = _repo(tmp_path, "analysis/custom_inference.py")

    def malformed(command, **kwargs):
        out_dir = Path(command[command.index("--out_dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"score": 1.0}]).to_csv(
            out_dir / "ThermoMPNN_inference_input.csv", index=False
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    with pytest.raises(ScientificOutputError, match="columns"):
        ThermoMPNNRunner(repo, command_runner=malformed).run(
            tmp_path / "input.pdb", [SINGLE], tmp_path / "scores"
        )


def test_thermompnn_rejects_missing_requested_mutation(tmp_path: Path) -> None:
    repo = _repo(tmp_path, "analysis/custom_inference.py")

    def wrong_mutation(command, **kwargs):
        out_dir = Path(command[command.index("--out_dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{"ddG_pred": 0.1, "position": 10, "wildtype": "A", "mutation": "G"}]
        ).to_csv(out_dir / "ThermoMPNN_inference_input.csv", index=False)
        return subprocess.CompletedProcess(command, 0, "", "")

    with pytest.raises(ScientificOutputError, match="Y91F"):
        ThermoMPNNRunner(repo, command_runner=wrong_mutation).run(
            tmp_path / "input.pdb", [SINGLE], tmp_path / "scores"
        )


def test_thermompnn_reuses_genuine_full_sweep_for_later_selection(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path, "analysis/custom_inference.py")
    raw = tmp_path / "ThermoMPNN_inference_input.csv"
    pd.DataFrame(
        [
            {"ddG_pred": -0.42, "position": 90, "wildtype": "Y", "mutation": "F"},
            {"ddG_pred": -0.31, "position": 39, "wildtype": "D", "mutation": "A"},
        ]
    ).to_csv(raw, index=False)
    novel = StabilityVariant("D40A", "D40A", "D64A")
    scores = ThermoMPNNRunner(repo).normalize_existing(
        raw, [novel], tmp_path / "normalized"
    )
    assert scores.loc[0, "variant_id"] == "D40A"
    assert scores.loc[0, "predicted_ddg_or_score"] == pytest.approx(-0.31)


def test_thermompnn_d_normalizes_double_output(tmp_path: Path) -> None:
    repo = _repo(tmp_path, "v2_ssm.py")

    def succeed(command, **kwargs):
        prefix = Path(command[command.index("--out") + 1])
        prefix.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{"ddG (kcal/mol)": 1.7, "Mutation": "Y91F:D126A", "CA-CA Distance": 9.0}]
        ).to_csv(prefix.with_suffix(".csv"), index=False)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    scores = ThermoMPNNDRunner(repo, command_runner=succeed).run(
        tmp_path / "input.pdb", [DOUBLE], tmp_path / "scores"
    )
    assert scores.loc[0, "model_used"] == "ThermoMPNN-D epistatic"
    assert scores.loc[0, "predicted_ddg_or_score"] == pytest.approx(1.7)
    assert scores.loc[0, "mutation_set"] == "Y91F:D126A"


def test_thermompnn_d_accepts_chain_qualified_deposited_numbering(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path, "v2_ssm.py")

    def succeed(command, **kwargs):
        prefix = Path(command[command.index("--out") + 1])
        prefix.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "ddG (kcal/mol)": 1.7,
                    "Mutation": "YA115F:DA150A",
                    "CA-CA Distance": 9.0,
                }
            ]
        ).to_csv(prefix.with_suffix(".csv"), index=False)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    scores = ThermoMPNNDRunner(repo, command_runner=succeed).run(
        tmp_path / "input.pdb", [DOUBLE], tmp_path / "scores"
    )

    assert scores.loc[0, "variant_id"] == "Y91F_D126A"
    assert scores.loc[0, "deposited_numbering"] == "Y115F:D150A"
    assert scores.loc[0, "predicted_ddg_or_score"] == pytest.approx(1.7)


def test_thermompnn_d_uses_checkout_first_preserved_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path, "v2_ssm.py")
    conflicting = tmp_path / "site-packages"
    conflicting.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(conflicting))
    monkeypatch.setenv("ATLAS_UPSTREAM_SENTINEL", "preserved")
    captured: dict[str, object] = {}

    def succeed(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        prefix = Path(command[command.index("--out") + 1])
        prefix.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{"ddG (kcal/mol)": 1.7, "Mutation": "Y91F:D126A"}]
        ).to_csv(prefix.with_suffix(".csv"), index=False)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    ThermoMPNNDRunner(repo, command_runner=succeed).run(
        tmp_path / "input.pdb", [DOUBLE], tmp_path / "scores"
    )

    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["ATLAS_UPSTREAM_SENTINEL"] == "preserved"
    assert environment["PYTHONPATH"].split(os.pathsep) == [
        str(repo.resolve()),
        str(conflicting),
    ]
    assert captured["cwd"] == repo.resolve()
    assert captured["command"][:2] == [
        sys.executable,
        str((repo / "v2_ssm.py").resolve()),
    ]
