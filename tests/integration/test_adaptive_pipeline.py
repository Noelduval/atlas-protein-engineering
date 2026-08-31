from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

from atlas.adaptive.ledger import ScientificLedger
from atlas.adaptive.models import (
    CandidateRecord,
    DesignStrategy,
    EvidenceAxis,
    EvidenceRecord,
    HardViolation,
)
from atlas.adaptive_pipeline import (
    AdaptivePipelineConfig,
    CandidateStructureResult,
    EARLY_OBJECTIVES,
    _evaluations,
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


def test_stability_is_normalized_within_upstream_model_without_raw_cross_model_arithmetic() -> None:
    reference = "ACDEFGHIKLMNPQRSTVWY"
    single = CandidateRecord.create(
        reference_sequence=reference,
        mutations=["C2S"],
        parents=(),
        strategy=DesignStrategy.CONSERVATIVE,
        structural_region="scaffold_surface",
        round_index=1,
        hypothesis="Single-model test.",
        intended_upside="Test model-aware screening.",
        expected_risk="Test fixture.",
    )
    double = CandidateRecord.create(
        reference_sequence=reference,
        mutations=["D3N", "E4Q"],
        parents=(),
        strategy=DesignStrategy.EVIDENCE_GUIDED_COMBINATION,
        structural_region="cross_region",
        round_index=3,
        hypothesis="Double-model test.",
        intended_upside="Test model-aware screening.",
        expected_risk="Test fixture.",
    )
    evidence = (
        _numeric(single, EvidenceAxis.STABILITY, -3.0, "ThermoMPNN"),
        _numeric(double, EvidenceAxis.STABILITY, 9.0, "ThermoMPNN-D"),
    )

    evaluations = _evaluations((single, double), evidence)

    for evaluation in evaluations:
        axes = evaluation.latest_by_axis()
        assert axes[EvidenceAxis.STABILITY].value in {-3.0, 9.0}
        assert axes[EvidenceAxis.STABILITY_MODEL_AWARE].value == 0.5
        assert axes[EvidenceAxis.STABILITY_MODEL_AWARE].payload[
            "raw_scale_not_cross_model_comparable"
        ] is True


def test_pre_structure_funnel_does_not_treat_deposited_distances_as_mutant_performance() -> None:
    assert {objective.axis for objective in EARLY_OBJECTIVES} == {
        EvidenceAxis.STABILITY_MODEL_AWARE,
        EvidenceAxis.LIABILITY,
    }


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
        raise AssertionError("replicated MD must not be called by adaptive production")


class HardFailingStructureTestBackend(DeterministicTestBackend):
    def evaluate_structure(self, reference_pdb, candidate, output_dir, *, seed):
        result = super().evaluate_structure(
            reference_pdb, candidate, output_dir, seed=seed
        )
        return CandidateStructureResult(
            candidate_id=result.candidate_id,
            structure_path=result.structure_path,
            evidence=result.evidence,
            hard_violations=(
                HardViolation(
                    candidate.candidate_id,
                    "injected_malformed_chemistry",
                    "Injected structure hard-failure boundary.",
                ),
            ),
            artifact_path=result.artifact_path,
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
        adversarial_target=4,
        portfolio_target=3,
        repair_parent_target=2,
    )
    values.update(updates)
    return AdaptivePipelineConfig(**values)


def test_full_adaptive_pipeline_exhausts_budget_and_repairs(
    tmp_path: Path, monkeypatch
) -> None:
    def legacy_path_must_not_run(*args, **kwargs):
        raise AssertionError("legacy generator/ranker influenced adaptive finalists")

    monkeypatch.setattr(
        "atlas.design.candidate_generator.generate_candidates",
        legacy_path_must_not_run,
    )
    legacy_rank_module = importlib.import_module("atlas.design.rank_candidates")
    monkeypatch.setattr(legacy_rank_module, "rank_candidates", legacy_path_must_not_run)
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
    assert "md_candidates" not in funnel
    assert funnel["adversarial_review"] == 4
    assert funnel["finalists"] == len(result.finalist_ids)
    assert json.loads((run_dir / "completion_audit.json").read_text())["complete"] is True
    assert (run_dir / "repair" / "repair_trajectories.json").is_file()
    assert (run_dir / "repair" / "repair_eligibility.json").is_file()
    assert (run_dir / "retrospective_method_characterization.json").is_file()
    generation = json.loads(
        (run_dir / "search" / "candidate_generation_summary.json").read_text()
    )
    assert generation["policy_version"]
    assert generation["counts_by_substitution_class"]
    assert set(generation["counts_by_double_category"]) <= {
        "LOCAL_COUPLED_DOUBLE",
        "FUNCTION_STABILITY_RESCUE_DOUBLE",
        "ORTHOGONAL_MECHANISM_DOUBLE",
    }
    assert hashlib.sha256(BENCHMARK_HISTORY.read_bytes()).hexdigest() == before
    reviews = json.loads((run_dir / "reports" / "adversarial_reviews.json").read_text())
    double_reviews = [review for review in reviews if "/" in review["mutation_set"]]
    assert double_reviews
    assert all(review["double_category"] for review in double_reviews)
    assert all(review["epistasis_uncertainty"] == "high" for review in double_reviews)
    assert all(review["parent_child_structural_evidence"] for review in double_reviews)

    with ScientificLedger.open(
        run_dir / "design_memory" / "atlas_science.sqlite",
        run_dir / "design_memory" / "events.jsonl",
    ) as ledger:
        assert ledger.candidate_count() >= 5_000
        assert ledger.failures()
        axes = {record.axis for record in ledger.all_evidence()}
        assert EvidenceAxis.STABILITY in axes
        assert EvidenceAxis.STABILITY_MODEL_AWARE in axes


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
        backend=HardFailingStructureTestBackend(),
    )
    assert result.status == "completed"
    assert result.finalist_ids == ()
    assert result.completion_audit.complete is True
    near_misses = json.loads(
        (result.run_dir / "reports" / "best_near_miss_hypotheses.json").read_text()
    )
    assert near_misses
    execution = json.loads((result.run_dir / "execution_status.json").read_text())
    assert execution["scientific_conclusion"] == "ZERO_FINALISTS_AFTER_EXHAUSTIVE_SEARCH"


def test_adaptive_path_records_md_methodological_exclusion(tmp_path: Path) -> None:
    result = run_adaptive_pipeline(
        _config(tmp_path, "adaptive-md-excluded"),
        backend=DeterministicTestBackend(),
    )

    exclusion = json.loads(
        (result.run_dir / "replicated_md_methodological_exclusion.json").read_text()
    )
    assert exclusion["included_in_candidate_discrimination"] is False
    assert exclusion["reference_validation"] == "FAILED"
    assert not (result.run_dir / "md").exists()
