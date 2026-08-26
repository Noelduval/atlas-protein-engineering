from __future__ import annotations

from atlas.adaptive.models import (
    CandidateRecord,
    DesignStrategy,
    EvidenceAxis,
    EvidenceRecord,
    HardViolation,
)
from atlas.adaptive.screening import (
    Direction,
    Evaluation,
    Objective,
    pareto_front,
    select_diverse_survivors,
)


REFERENCE = "ACDEFGHIKLMNPQRSTVWY"


def _candidate(mutation: str, region: str, strategy: DesignStrategy) -> CandidateRecord:
    return CandidateRecord.create(
        reference_sequence=REFERENCE,
        mutations=[mutation],
        parents=(),
        strategy=strategy,
        structural_region=region,
        round_index=1,
        hypothesis="Independent scientific hypothesis.",
        intended_upside="Improve one evidence axis.",
        expected_risk="Trade off another evidence axis.",
    )


def _evidence(candidate_id: str, axis: EvidenceAxis, value: float) -> EvidenceRecord:
    return EvidenceRecord.numeric(
        candidate_id,
        axis,
        value=value,
        uncertainty=0.1,
        method="test-method",
        provenance={"test": True},
    )


OBJECTIVES = (
    Objective(EvidenceAxis.STABILITY, Direction.MINIMIZE),
    Objective(EvidenceAxis.SUBSTRATE_INTERFACE, Direction.MAXIMIZE),
)


def test_pareto_front_preserves_soft_tradeoffs_without_universal_score() -> None:
    stable = _candidate("C2S", "distal_stability", DesignStrategy.STABILITY_SUPPORT)
    interface = _candidate(
        "D3N", "substrate_interface", DesignStrategy.SUBSTRATE_INTERFACE
    )
    dominated = _candidate("E4Q", "second_shell", DesignStrategy.CONSERVATIVE)
    evaluations = (
        Evaluation(stable, (_evidence(stable.candidate_id, EvidenceAxis.STABILITY, -1.0), _evidence(stable.candidate_id, EvidenceAxis.SUBSTRATE_INTERFACE, 0.2))),
        Evaluation(interface, (_evidence(interface.candidate_id, EvidenceAxis.STABILITY, 0.3), _evidence(interface.candidate_id, EvidenceAxis.SUBSTRATE_INTERFACE, 0.9))),
        Evaluation(dominated, (_evidence(dominated.candidate_id, EvidenceAxis.STABILITY, 0.8), _evidence(dominated.candidate_id, EvidenceAxis.SUBSTRATE_INTERFACE, 0.1))),
    )

    front = pareto_front(evaluations, OBJECTIVES)
    assert {evaluation.candidate.candidate_id for evaluation in front} == {
        stable.candidate_id,
        interface.candidate_id,
    }
    assert all(not hasattr(evaluation, "ranking_score") for evaluation in evaluations)


def test_hard_violations_are_rejected_but_bad_soft_metric_is_not_hard_failure() -> None:
    soft_bad = _candidate("C2S", "distal_stability", DesignStrategy.STABILITY_SUPPORT)
    hard_bad = _candidate("D3N", "substrate_interface", DesignStrategy.SUBSTRATE_INTERFACE)
    soft_evaluation = Evaluation(
        soft_bad,
        (_evidence(soft_bad.candidate_id, EvidenceAxis.STABILITY, 5.0),),
    )
    hard_evaluation = Evaluation(
        hard_bad,
        (),
        hard_violations=(
            HardViolation(hard_bad.candidate_id, "malformed_chemistry", "clash"),
        ),
    )
    assert soft_evaluation.hard_rejected is False
    assert hard_evaluation.hard_rejected is True


def test_diverse_survivors_span_regions_and_strategies() -> None:
    candidates = (
        _candidate("C2S", "distal_stability", DesignStrategy.STABILITY_SUPPORT),
        _candidate("D3N", "substrate_interface", DesignStrategy.SUBSTRATE_INTERFACE),
        _candidate("E4Q", "second_shell", DesignStrategy.SECOND_SHELL_PREORGANIZATION),
        _candidate("F5Y", "scaffold_surface", DesignStrategy.CONSERVATIVE),
        _candidate("G6A", "distal_stability", DesignStrategy.STABILITY_SUPPORT),
    )
    evaluations = tuple(
        Evaluation(
            candidate,
            (
                _evidence(candidate.candidate_id, EvidenceAxis.STABILITY, float(index)),
                _evidence(
                    candidate.candidate_id,
                    EvidenceAxis.SUBSTRATE_INTERFACE,
                    float(len(candidates) - index),
                ),
            ),
        )
        for index, candidate in enumerate(candidates)
    )
    survivors = select_diverse_survivors(evaluations, OBJECTIVES, target=4)
    assert len(survivors) == 4
    assert len({evaluation.candidate.structural_region for evaluation in survivors}) == 4

