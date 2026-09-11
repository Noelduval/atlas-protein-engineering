"""Small deterministic output writers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys
from typing import Iterable, Mapping

from atlas.stability.thermompnn_d_runner import THERMOMPNN_D_REVISION
from atlas.stability.thermompnn_runner import THERMOMPNN_REVISION


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_warnings(warnings: Iterable[str], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = [item for item in warnings if item]
    content = "\n".join(f"- {item}" for item in entries) or "- No pipeline warnings."
    output.write_text(
        "# Pipeline warnings\n\n"
        "Warnings identify unavailable evidence or scientific uncertainty; they are not filled with synthetic values.\n\n"
        f"{content}\n"
    )
    return output


def write_provenance(
    input_path: str | Path,
    path: str | Path,
    extra: Mapping[str, object] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    packages: dict[str, str] = {}
    for package in ("atlas-protein-engineering", "biopython", "numpy", "pandas", "matplotlib", "openmm", "torch"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not installed"
    payload: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "input_structure": str(Path(input_path).resolve()),
        "input_sha256": sha256(input_path),
        "thermompnn_revision": THERMOMPNN_REVISION,
        "thermompnn_d_revision": THERMOMPNN_D_REVISION,
        "claim_boundary": "computational predictions requiring experimental validation",
    }
    payload.update(extra or {})
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def write_run_manifest(
    input_path: str | Path,
    path: str | Path,
    *,
    run_id: str,
    checkpoint_directory: str | Path,
    run_context: Mapping[str, object],
) -> Path:
    """Write the self-contained reproduction identity for a checkpointed run."""

    return write_provenance(
        input_path,
        path,
        {
            **dict(run_context),
            "run_id": run_id,
            "checkpoint_directory": str(Path(checkpoint_directory).resolve()),
            "python_executable": sys.executable,
        },
    )
