"""Official ThermoMPNN-D epistatic double-mutant inference adapter."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
from typing import Mapping, Sequence

import pandas as pd

from atlas.stability.common import (
    CommandRunner,
    ScientificOutputError,
    StabilityVariant,
    normalized_frame,
    normalized_row,
    require_columns,
    require_repository,
    require_revision,
)
from atlas.stability.upstream_execution import UpstreamPythonExecution


THERMOMPNN_D_REVISION = "df9a75aaddb674a7c4c193005031fc0536d325fb"
_DOUBLE_TOKEN = re.compile(
    r"^(?P<wt>[A-Z])(?:(?P<chain>[A-Za-z]))?(?P<position>\d+)(?P<mut>[A-Z])$"
)


def _run_thermompnn_d_command(
    command: Sequence[str], *, cwd: Path, env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _canonical_double(label: str) -> tuple[str, ...]:
    tokens = label.replace(";", ":").replace("/", ":").split(":")
    normalized: list[str] = []
    for token in tokens:
        match = _DOUBLE_TOKEN.fullmatch(token)
        if match is None:
            normalized.append(token)
            continue
        normalized.append(
            f"{match.group('wt')}{match.group('position')}{match.group('mut')}"
        )
    return tuple(sorted(normalized))


class ThermoMPNNDRunner:
    """Run official epistatic inference with an inclusive output threshold."""

    def __init__(
        self,
        repository: str | Path,
        *,
        command_runner: CommandRunner = _run_thermompnn_d_command,
        distance_cutoff_a: float = 12.0,
    ) -> None:
        self.repository = Path(repository)
        self.command_runner = command_runner
        self.distance_cutoff_a = distance_cutoff_a

    def run(
        self,
        pdb_path: str | Path,
        variants: Sequence[StabilityVariant],
        output_dir: str | Path,
    ) -> pd.DataFrame:
        script = require_repository(self.repository, "v2_ssm.py", "ThermoMPNN-D")
        require_revision(self.repository, THERMOMPNN_D_REVISION, "ThermoMPNN-D")
        execution = UpstreamPythonExecution.create(self.repository, script)
        pdb = Path(pdb_path).resolve()
        destination = Path(output_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        prefix = destination / "thermompnn_d_epistatic"
        command = execution.script_command([
            "--mode",
            "epistatic",
            "--pdb",
            str(pdb),
            "--chains",
            "A",
            "--threshold",
            "100",
            "--distance",
            str(self.distance_cutoff_a),
            "--out",
            str(prefix),
        ])
        completed = self.command_runner(
            command,
            cwd=execution.cwd,
            env=execution.environment(),
        )
        if completed.returncode:
            detail = (completed.stderr or completed.stdout or "unknown error").strip()
            if "cuda" in detail.lower():
                detail += " (the pinned official epistatic implementation requires a CUDA GPU)"
            raise ScientificOutputError(f"ThermoMPNN-D inference failed: {detail}")

        csv_path = prefix.with_suffix(".csv")
        if not csv_path.is_file():
            raise ScientificOutputError(
                f"ThermoMPNN-D completed without expected output {csv_path}"
            )
        frame = pd.read_csv(csv_path)
        require_columns(frame, {"ddG (kcal/mol)", "Mutation"}, "ThermoMPNN-D")
        lookup = {
            _canonical_double(str(row["Mutation"])): float(row["ddG (kcal/mol)"])
            for _, row in frame.iterrows()
        }
        rows: list[dict[str, object]] = []
        for variant in variants:
            dp622_key = _canonical_double(variant.mutation_set)
            deposited_key = _canonical_double(variant.deposited_numbering)
            if len(dp622_key) != 2:
                raise ScientificOutputError(
                    f"ThermoMPNN-D epistatic adapter requires two mutations: {variant.mutation_set}"
                )
            key = next(
                (candidate for candidate in (deposited_key, dp622_key) if candidate in lookup),
                None,
            )
            if key is None:
                raise ScientificOutputError(
                    f"ThermoMPNN-D output does not contain requested mutation {variant.mutation_set}; "
                    f"looked for deposited numbering {variant.deposited_numbering} and "
                    f"raw DP622 numbering; verify the {self.distance_cutoff_a:g} Å pair cutoff"
                )
            rows.append(
                normalized_row(
                    variant, "ThermoMPNN-D epistatic", lookup[key]
                )
            )
        result = normalized_frame(rows)
        result.to_csv(destination / "thermompnn_d_scores_normalized.csv", index=False)
        return result
