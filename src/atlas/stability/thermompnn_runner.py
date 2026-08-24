"""Official ThermoMPNN single-mutant inference adapter."""

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


THERMOMPNN_REVISION = "2b04fd370e399911b1fa5848112cc9013f084110"
_SINGLE_MUTATION = re.compile(r"^(?P<wt>[A-Z])(?P<position>\d+)(?P<mut>[A-Z])$")


def _upstream_zero_based_label(dp622_label: str) -> str:
    """Translate Atlas' one-based DP622 label to ThermoMPNN's sequence index."""

    match = _SINGLE_MUTATION.fullmatch(dp622_label)
    if match is None:
        raise ScientificOutputError(
            f"ThermoMPNN requires one DP622 single-mutation label: {dp622_label}"
        )
    position = int(match.group("position")) - 1
    if position < 0:
        raise ScientificOutputError(
            f"ThermoMPNN DP622 position must be one-based: {dp622_label}"
        )
    return f"{match.group('wt')}{position}{match.group('mut')}"


def _run_thermompnn_command(
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


class ThermoMPNNRunner:
    """Run the official exhaustive single-site model and select requested rows."""

    def __init__(
        self,
        repository: str | Path,
        *,
        command_runner: CommandRunner = _run_thermompnn_command,
    ) -> None:
        self.repository = Path(repository)
        self.command_runner = command_runner

    def run(
        self,
        pdb_path: str | Path,
        variants: Sequence[StabilityVariant],
        output_dir: str | Path,
    ) -> pd.DataFrame:
        script = require_repository(
            self.repository, "analysis/custom_inference.py", "ThermoMPNN"
        )
        require_revision(self.repository, THERMOMPNN_REVISION, "ThermoMPNN")
        execution = UpstreamPythonExecution.create(self.repository, script)
        pdb = Path(pdb_path).resolve()
        destination = Path(output_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        command = execution.script_command([
            "--pdb",
            str(pdb),
            "--chain",
            "A",
            "--out_dir",
            str(destination),
        ])
        completed = self.command_runner(
            command,
            cwd=execution.cwd,
            env=execution.environment(),
        )
        if completed.returncode:
            detail = (completed.stderr or completed.stdout or "unknown error").strip()
            raise ScientificOutputError(f"ThermoMPNN inference failed: {detail}")

        csv_path = destination / f"ThermoMPNN_inference_{pdb.stem}.csv"
        if not csv_path.is_file():
            raise ScientificOutputError(
                f"ThermoMPNN completed without expected output {csv_path}"
            )
        return self.normalize_existing(csv_path, variants, destination)

    def normalize_existing(
        self,
        csv_path: str | Path,
        variants: Sequence[StabilityVariant],
        output_dir: str | Path,
    ) -> pd.DataFrame:
        """Select requested variants from a genuine exhaustive inference CSV."""

        source = Path(csv_path).resolve()
        destination = Path(output_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        if not source.is_file():
            raise ScientificOutputError(
                f"ThermoMPNN output does not exist: {source}"
            )
        frame = pd.read_csv(source)
        require_columns(
            frame, {"ddG_pred", "position", "wildtype", "mutation"}, "ThermoMPNN"
        )

        lookup: dict[str, float] = {}
        for row in frame.itertuples(index=False):
            label = f"{row.wildtype}{int(row.position)}{row.mutation}"
            lookup[label] = float(row.ddG_pred)
        rows: list[dict[str, object]] = []
        for variant in variants:
            if ":" in variant.mutation_set or ";" in variant.mutation_set:
                raise ScientificOutputError(
                    f"ThermoMPNN single-mutant adapter cannot score {variant.mutation_set}"
                )
            upstream_label = _upstream_zero_based_label(variant.mutation_set)
            if upstream_label not in lookup:
                raise ScientificOutputError(
                    "ThermoMPNN output does not contain requested mutation "
                    f"{variant.mutation_set} (expected upstream label {upstream_label})"
                )
            rows.append(
                normalized_row(
                    variant, "ThermoMPNN", lookup[upstream_label]
                )
            )
        result = normalized_frame(rows)
        result.to_csv(destination / "thermompnn_scores_normalized.csv", index=False)
        return result
