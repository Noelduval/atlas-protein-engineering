from __future__ import annotations

import json
from pathlib import Path

from atlas.adaptive.models import EvidenceAxis, EvidenceStatus
from atlas.research.activity_models import (
    activity_evidence_unavailable,
    audit_activity_models,
    write_activity_model_audit,
)


def test_activity_model_audit_covers_requested_and_current_alternatives() -> None:
    decision = audit_activity_models()
    names = {review.name for review in decision.reviews}
    assert {"ProMEP", "EnzyACT", "CatPred", "UniKP", "PLACER"} <= names
    assert all(review.prediction_target for review in decision.reviews)
    assert all(review.inputs for review in decision.reviews)
    assert all(review.outputs for review in decision.reviews)
    assert all(review.license for review in decision.reviews)
    assert all(review.primary_sources for review in decision.reviews)


def test_no_model_is_integrated_without_dp622_domain_and_reproducibility() -> None:
    decision = audit_activity_models()
    assert decision.status == "not_available"
    assert decision.selected_model is None
    assert all(review.eligible_for_atlas is False for review in decision.reviews)
    by_name = {review.name: review for review in decision.reviews}
    assert by_name["ProMEP"].prediction_target == "general_protein_fitness_log_likelihood"
    assert by_name["EnzyACT"].checkpoint_available is False
    assert by_name["CatPred"].supports_peptide_substrate is False
    assert by_name["UniKP"].metal_treatment_documented is False
    assert by_name["PLACER"].prediction_target == "conformational_ensemble"
    assert all(review.exclusion_reasons for review in decision.reviews)


def test_activity_model_decision_is_not_tuned_to_historical_benchmark() -> None:
    decision = audit_activity_models()
    assert decision.benchmark_used_for_selection is False
    assert "Y91F" not in json.dumps(decision.to_dict())
    assert "D126A" not in json.dumps(decision.to_dict())


def test_unavailable_activity_evidence_is_explicit_not_favorable() -> None:
    evidence = activity_evidence_unavailable("ATLAS-X", audit_activity_models())
    assert evidence.axis is EvidenceAxis.ACTIVITY_ORIENTED
    assert evidence.status is EvidenceStatus.UNAVAILABLE
    assert evidence.value is None
    assert evidence.payload["decision"] == "not_available"


def test_audit_writes_machine_and_human_readable_artifacts(tmp_path: Path) -> None:
    json_path = tmp_path / "activity_model_decision.json"
    markdown_path = tmp_path / "activity_model_review.md"
    write_activity_model_audit(audit_activity_models(), json_path, markdown_path)
    payload = json.loads(json_path.read_text())
    assert payload["status"] == "not_available"
    assert len(payload["reviews"]) >= 5
    assert "Why Atlas excludes all reviewed models" in markdown_path.read_text()

