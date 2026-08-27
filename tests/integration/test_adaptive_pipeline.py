from __future__ import annotations

import hashlib
import json
from pathlib import Path

from atlas.adaptive.ledger import ScientificLedger
from atlas.adaptive.models import (
    DesignStrategy,
    EvidenceAxis,
    EvidenceRecord,
    HardViolation,
)
from atlas.adaptive_pipeline import (
    AdaptivePipelineConfig,
    CandidateDynamicsResult,
    CandidateStructureResult,
    run_adaptive_pipeline,
)


SOURCE = Path(__file__).parents[2] / "data" / "23WN.cif"
BENCHMARK_HISTORY = Path(__file__).parents[2] / "docs" / "benchmark_failure_analysis.md"


def _numeric(candidate, axis, value, method):
    return EvidenceRecord.numeric(
        candidate.candidate_id,
        axis,
        value=value,
        uncertainty=0.1,
        method=method,
        provenance={"test_boundary": True},
    )


class DeterministicTestBackend:
    """Fast injected scientific boundary; production never uses these values."""

    def __init__(self) -> None:
        self.scored_rounds: list[tuple[int, ...]] = []

    def score_stability(self, reference_pdb, candidates, output_dir):
        self.scored_rounds.append(tuple(sorted({c.round_index for c in candidates})))
        records = []
        for candidate in candidates:
            if candidate.strategy is DesignStrategy.REPAIR_RESCUE:
                value = -0.4
            else:
                value = 1.4 if candidate.mutations[0].position % 7 == 0 else -0.3
            records.append(
                _numeric(candidate, EvidenceAxis.STABILITY, value, "injected-test-stability")
            )
        return tuple(records)

    def evaluate_structure(self, reference_pdb, candidate, output_dir, *, seed):
        output_dir.mkdir(parents=True, exist_ok=True)
        marker = output_dir / "test_structure.json"
        marker.write_text(json.dumps({"candidate_id": candidate.candidate_id}) + "\n")
        stress = (
            0.0
            if candidate.strategy is DesignStrategy.REPAIR_RESCUE
            else (1.6 if int(candidate.candidate_id[-2:], 16) % 4 == 0 else 0.0)
        )
        return CandidateStructureResult(
            candidate_id=candidate.candidate_id,
            structure_path=Path(reference_pdb),
            evidence=(
                _numeric(candidate, EvidenceAxis.STABILITY, stress, "injected-test-structure"),
                _numeric(candidate, EvidenceAxis.STRUCTURE_QUALITY, 0.1, "injected-test-structure"),
                _numeric(candidate, EvidenceAxis.CATALYTIC_GEOMETRY, 0.2, "injected-test-structure"),
                _numeric(candidate, EvidenceAxis.SUBSTRATE_INTERFACE, 0.9, "injected-test-structure"),
            ),
            hard_violations=(),
            artifact_path=marker,
        )

    def evaluate_dynamics(self, pdb_path, candidate, output_dir, *, system_label):
        output_dir.mkdir(parents=True, exist_ok=True)
        marker = output_dir / "test_md.json"
        marker.write_text(json.dumps({"system_label": system_label}) + "\n")
        evidence = () if candidate is None else (
            _numeric(candidate, EvidenceAxis.DYNAMICS, 0.2, "injected-test-md"),
            _numeric(candidate, EvidenceAxis.SUBSTRATE_INTERFACE, 0.85, "injected-test-md"),
            _numeric(candidate, EvidenceAxis.CATALYTIC_GEOMETRY, 0.25, "injected-test-md"),
        )
        return CandidateDynamicsResult(
            candidate_id=None if candidate is None else candidate.candidate_id,
            evidence=evidence,
            hard_violations=(),
            artifact_path=marker,
            completed_replicas=3,
        )


class HardFailingMDTestBackend(DeterministicTestBackend):
    def evaluate_dynamics(self, pdb_path, candidate, output_dir, *, system_label):
        result = super().evaluate_dynamics(
            pdb_path, candidate, output_dir, system_label=system_label
        )
        if candidate is None:
            return result
        return CandidateDynamicsResult(
            candidate_id=candidate.candidate_id,
            evidence=(),
            hard_violations=(
                HardViolation(
                    candidate.candidate_id,
                    "invalid_replicated_simulation",
                    "Injected test boundary: all candidate replicas invalid.",
                ),
            ),
            artifact_path=result.artifact_path,
            completed_replicas=0,
        )


def _config(tmp_path: Path, run_id: str, **updates) -> AdaptivePipelineConfig:
    values = dict(
        input_structure=SOURCE,
        output_root=tmp_path,
        atlas_repo=Path.cwd(),
        run_id=run_id,
        candidate_budget=5_000,
        broad_target=40,
        structure_target=16,
        md_target=6,
        adversarial_target=4,
        portfolio_target=3,
        repair_parent_target=2,
    )
    values.update(updates)
    return AdaptivePipelineConfig(**values)


def test_full_adaptive_pipeline_exhausts_budget_and_repairs(tmp_path: Path) -> None:
    before = hashlib.sha256(BENCHMARK_HISTORY.read_bytes()).hexdigest()
    result = run_adaptive_pipeline(
        _config(tmp_path, "adaptive-integration"),
        backend=DeterministicTestBackend(),
    )

    assert result.status == "completed"
    assert result.completion_audit.complete is True
    assert result.completion_audit.unique_legal_evaluated >= 5_000
    assert 1 <= len(result.finalist_ids) <= 3
    run_dir = result.run_dir
    funnel = json.loads((run_dir / "reports" / "funnel_counts.json").read_text())
    assert funnel["generated_evaluated"] >= 5_000
    assert funnel["broad_survivors"] == 40
    assert funnel["structural_analyses"] >= 16
    assert funnel["md_candidates"] == 6
    assert funnel["adversarial_review"] == 4
    assert funnel["finalists"] == len(result.finalist_ids)
    assert json.loads((run_dir / "completion_audit.json").read_text())["complete"] is True
    assert json.loads((run_dir / "repair" / "repair_trajectories.json").read_text())
    assert (run_dir / "retrospective_method_characterization.json").is_file()
    assert hashlib.sha256(BENCHMARK_HISTORY.read_bytes()).hexdigest() == before

    with ScientificLedger.open(
        run_dir / "design_memory" / "atlas_science.sqlite",
        run_dir / "design_memory" / "events.jsonl",
    ) as ledger:
        assert ledger.candidate_count() > 5_000
        assert DesignStrategy.REPAIR_RESCUE.value in {
            candidate.strategy.value for candidate in ledger.candidates()
        }
        assert ledger.failures()


def test_resume_does_not_repeat_completed_round(tmp_path: Path) -> None:
    first_backend = DeterministicTestBackend()
    stopped = run_adaptive_pipeline(
        _config(tmp_path, "adaptive-resume", stop_after="round1"),
        backend=first_backend,
    )
    assert stopped.status == "stopped_after_round1"
    assert first_backend.scored_rounds == [(1,)]

    resumed_backend = DeterministicTestBackend()
    completed = run_adaptive_pipeline(
        _config(tmp_path, "adaptive-resume", resume=True),
        backend=resumed_backend,
    )
    assert completed.status == "completed"
    assert (1,) not in resumed_backend.scored_rounds
    assert (2,) in resumed_backend.scored_rounds
    assert (3,) in resumed_backend.scored_rounds


def test_zero_finalists_is_terminal_only_after_exhaustion_and_near_miss_report(
    tmp_path: Path,
) -> None:
    result = run_adaptive_pipeline(
        _config(tmp_path, "adaptive-zero-finalists"),
        backend=HardFailingMDTestBackend(),
    )

    assert result.status == "completed"
    assert result.finalist_ids == ()
    assert result.completion_audit.complete is True
    assert result.completion_audit.unique_legal_evaluated >= 5_000
    near_misses = json.loads(
        (result.run_dir / "reports" / "best_near_miss_hypotheses.json").read_text()
    )
    assert near_misses
    assert all(record["decision"] == "NEAR_MISS" for record in near_misses)
    execution = json.loads((result.run_dir / "execution_status.json").read_text())
    assert execution["scientific_conclusion"] == "ZERO_FINALISTS_AFTER_EXHAUSTIVE_SEARCH"
