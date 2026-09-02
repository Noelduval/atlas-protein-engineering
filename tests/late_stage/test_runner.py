from atlas.adaptive.models import EvidenceAxis
from atlas.late_stage.runner import select_late_stage_finalists


def test_late_stage_capabilities_are_independent_ledger_axes() -> None:
    assert {
        EvidenceAxis.ORTHOGONAL_STRUCTURE.value,
        EvidenceAxis.SUBSTRATE_SPECIFICITY.value,
        EvidenceAxis.POSE_ROBUSTNESS.value,
        EvidenceAxis.DEVELOPABILITY.value,
    } == {
        "orthogonal_structure",
        "substrate_specificity",
        "pose_robustness",
        "developability",
    }


def _record(candidate_id: str, **updates):
    record = {
        "candidate_id": candidate_id,
        "baseline_promoted": True,
        "specificity": {"status": "available", "classification": "intended_preferred"},
        "pose_robustness": {"status": "available", "classification": "robust"},
        "developability": {"status": "available", "classification": "acceptable_with_flags"},
        "orthogonal_structure": {"status": "unavailable"},
    }
    record.update(updates)
    return record


def test_finalist_gate_never_treats_missing_required_local_evidence_as_favorable() -> None:
    complete = _record("ATLAS-COMPLETE")
    missing = _record(
        "ATLAS-MISSING",
        specificity={"status": "unavailable", "classification": "indeterminate"},
    )

    selected, decisions = select_late_stage_finalists(
        (complete, missing), baseline_order=("ATLAS-MISSING", "ATLAS-COMPLETE"), target=5
    )

    assert selected == ("ATLAS-COMPLETE",)
    assert decisions["ATLAS-MISSING"]["eligible"] is False
    assert "specificity evidence is unavailable" in decisions["ATLAS-MISSING"]["blockers"]
    assert decisions["ATLAS-COMPLETE"]["orthogonal_structure_unavailable"] is True


def test_finalist_gate_preserves_baseline_order_and_caps_portfolio_without_score_mixing() -> None:
    records = tuple(_record(f"ATLAS-{index}") for index in range(7))

    selected, decisions = select_late_stage_finalists(
        records,
        baseline_order=tuple(record["candidate_id"] for record in reversed(records)),
        target=5,
    )

    assert selected == tuple(f"ATLAS-{index}" for index in range(6, 1, -1))
    assert len(selected) == 5
    assert all(item["universal_score_used"] is False for item in decisions.values())


def test_pose_robustness_is_diagnostic_and_developability_warnings_are_soft() -> None:
    record = _record(
        "ATLAS-SOFT",
        pose_robustness={"status": "available", "classification": "indeterminate"},
        developability={
            "status": "available",
            "classification": "review_required",
            "warnings": ["hydrophobic_sequence_patch"],
        },
    )
    selected, decisions = select_late_stage_finalists(
        (record,), baseline_order=("ATLAS-SOFT",), target=5
    )
    assert selected == ("ATLAS-SOFT",)
    assert decisions["ATLAS-SOFT"]["eligible"] is True


def test_candidate_introduced_developability_blocker_still_rejects() -> None:
    record = _record(
        "ATLAS-BLOCKED",
        developability={
            "status": "available",
            "classification": "review_required",
            "candidate_introduced_hard_blocker": True,
        },
    )
    selected, decisions = select_late_stage_finalists(
        (record,), baseline_order=("ATLAS-BLOCKED",), target=5
    )
    assert selected == ()
    assert decisions["ATLAS-BLOCKED"]["eligible"] is False
    assert "candidate-introduced hard blocker" in decisions["ATLAS-BLOCKED"]["blockers"][0]
