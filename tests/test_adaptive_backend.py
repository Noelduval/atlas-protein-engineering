from __future__ import annotations

from pathlib import Path

from atlas.adaptive_backend import OfficialAdaptiveBackend
from atlas.dynamics.models import DynamicsConfig


def test_production_backend_has_no_replicated_md_evaluation_surface(
    tmp_path: Path,
) -> None:
    backend = OfficialAdaptiveBackend(
        thermompnn_repo=tmp_path / "ThermoMPNN",
        thermompnn_d_repo=tmp_path / "ThermoMPNN-D",
        relaxation_config=DynamicsConfig(),
        seed=622,
    )

    assert not hasattr(backend, "evaluate_dynamics")
