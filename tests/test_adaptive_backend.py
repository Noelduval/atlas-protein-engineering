from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import atlas.adaptive_backend as backend_module
from atlas.adaptive.models import (
    CandidateRecord,
    DesignStrategy,
    EvidenceStatus,
)
from atlas.adaptive_backend import OfficialAdaptiveBackend
from atlas.dynamics.models import (
    DynamicsConfig,
    EnsembleSummary,
    ExplicitMDConfig,
    ReplicatedMDResult,
)


def _backend(tmp_path: Path) -> OfficialAdaptiveBackend:
    return OfficialAdaptiveBackend(
        thermompnn_repo=tmp_path / "ThermoMPNN",
        thermompnn_d_repo=tmp_path / "ThermoMPNN-D",
        relaxation_config=DynamicsConfig(),
        explicit_md_config=ExplicitMDConfig(),
        seed=622,
    )


def _fake_md(monkeypatch, tmp_path: Path, summary_payload: dict) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n")
    summary = tmp_path / "ensemble_summary.json"
    summary.write_text(json.dumps(summary_payload) + "\n")

    monkeypatch.setattr(
        backend_module,
        "run_replicated_explicit_md",
        lambda *args, **kwargs: ReplicatedMDResult(
            system_label=kwargs["system_label"],
            replicas=(),
            manifest_json=manifest,
        ),
    )
    monkeypatch.setattr(
        backend_module,
        "summarize_replicated_md",
        lambda *args, **kwargs: EnsembleSummary(
            completed_replicas=2,
            invalid_replicas=0,
            replica_summary_csv=tmp_path / "replicas.csv",
            summary_json=summary,
        ),
    )


def test_reference_md_registers_canonical_geometry_after_checkpoint_resume(
    monkeypatch, tmp_path: Path
) -> None:
    reference = tmp_path / "reference.pdb"
    reference.write_text("REMARK reference\n")
    mutant = tmp_path / "mutant.pdb"
    mutant.write_text("REMARK mutant\n")
    sentinel = SimpleNamespace()
    monkeypatch.setattr(backend_module, "measure_geometry", lambda *args: sentinel)
    _fake_md(monkeypatch, tmp_path, {})
    backend = _backend(tmp_path)

    backend.evaluate_dynamics(
        reference,
        None,
        tmp_path / "md-reference",
        system_label="reference_active_like_inferred",
    )

    assert backend._reference_geometry is sentinel
    assert backend._reference_path == reference.resolve()
    with pytest.raises(RuntimeError, match="different structure"):
        backend._reference(mutant)


def test_incomplete_md_metrics_fail_closed_without_favorable_values(
    monkeypatch, tmp_path: Path
) -> None:
    reference = tmp_path / "reference.pdb"
    reference.write_text("REMARK reference\n")
    mutant = tmp_path / "mutant.pdb"
    mutant.write_text("REMARK mutant\n")
    _fake_md(
        monkeypatch,
        tmp_path,
        {"ensemble_axes": {}, "replica_disagreement": {}},
    )
    backend = _backend(tmp_path)
    backend._reference_path = reference.resolve()
    backend._reference_geometry = SimpleNamespace(
        zn_h95_ne2_distance_a=2.1,
        zn_h99_ne2_distance_a=2.1,
        zn_e122_oxygen_distance_a=2.1,
        zn_scissile_oxygen_distance_a=2.3,
    )
    candidate = CandidateRecord.create(
        reference_sequence="ACDE",
        mutations=("C2S",),
        parents=(),
        strategy=DesignStrategy.CONSERVATIVE,
        structural_region="test_region",
        round_index=1,
        hypothesis="Exercise invalid ensemble handling.",
        intended_upside="None; test boundary.",
        expected_risk="Incomplete metrics.",
    )

    result = backend.evaluate_dynamics(
        mutant,
        candidate,
        tmp_path / "md-mutant",
        system_label=candidate.candidate_id,
    )

    assert result.hard_violations[0].code == "invalid_replicated_simulation_metrics"
    assert result.evidence[0].status is EvidenceStatus.INVALID
    assert all(record.value is None for record in result.evidence)
