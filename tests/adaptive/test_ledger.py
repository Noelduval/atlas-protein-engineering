from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlas.adaptive.ledger import DuplicateCandidateError, ScientificLedger
from atlas.adaptive.models import (
    CandidateRecord,
    DesignStrategy,
    EvidenceAxis,
    EvidenceRecord,
    FailureObservation,
)


def _candidate(mutation: str = "C2S") -> CandidateRecord:
    return CandidateRecord.create(
        reference_sequence="ACDEFGHIKLMNPQRSTVWY",
        mutations=[mutation],
        parents=(),
        strategy=DesignStrategy.CONSERVATIVE,
        structural_region="distal_stability",
        round_index=1,
        hypothesis="Conservative packing exploration.",
        intended_upside="Improve stability support.",
        expected_risk="May be neutral.",
    )


def test_ledger_persists_candidates_evidence_failures_and_jsonl_events(
    tmp_path: Path,
) -> None:
    ledger = ScientificLedger.create(tmp_path / "atlas.sqlite", tmp_path / "events.jsonl")
    candidate = _candidate()
    ledger.add_candidate(candidate)
    ledger.add_evidence(
        EvidenceRecord.numeric(
            candidate.candidate_id,
            EvidenceAxis.STABILITY,
            value=-0.4,
            uncertainty=0.2,
            method="ThermoMPNN",
            provenance={"commit": "abc123"},
        )
    )
    ledger.record_failure(
        FailureObservation(
            candidate_id=candidate.candidate_id,
            category="destabilizing_substitution",
            scope="position:2,substitution:S",
            detail="Observed stability regression.",
            evidence_count=1,
            confidence=0.35,
            source="ThermoMPNN",
        )
    )
    ledger.record_decision(
        candidate_id=candidate.candidate_id,
        route="REVISE",
        rationale="Useful conservative hypothesis with a repairable weakness.",
        round_index=1,
    )

    reopened = ScientificLedger.open(tmp_path / "atlas.sqlite", tmp_path / "events.jsonl")
    assert reopened.get_candidate(candidate.candidate_id) == candidate
    assert reopened.evidence_for(candidate.candidate_id)[0].value == -0.4
    assert reopened.failures(category="destabilizing_substitution")[0].evidence_count == 1
    assert reopened.decisions_for(candidate.candidate_id)[0]["route"] == "REVISE"

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert {event["event_type"] for event in events} >= {
        "candidate_added",
        "evidence_added",
        "failure_recorded",
        "decision_recorded",
    }


def test_sequence_uniqueness_is_enforced_transactionally(tmp_path: Path) -> None:
    ledger = ScientificLedger.create(tmp_path / "atlas.sqlite", tmp_path / "events.jsonl")
    first = _candidate("C2S")
    ledger.add_candidate(first)
    duplicate = CandidateRecord.create(
        reference_sequence="ACDEFGHIKLMNPQRSTVWY",
        mutations=["C2S"],
        parents=("DIFFERENT-PARENT",),
        strategy=DesignStrategy.REPAIR_RESCUE,
        structural_region="second_shell",
        round_index=4,
        hypothesis="Same sequence, different narrative.",
        intended_upside="None.",
        expected_risk="Duplicate.",
    )

    with pytest.raises(DuplicateCandidateError):
        ledger.add_candidate(duplicate)

    assert ledger.candidate_count() == 1
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 1


def test_checkpoint_state_is_resume_safe_and_context_bound(tmp_path: Path) -> None:
    ledger = ScientificLedger.create(tmp_path / "atlas.sqlite", tmp_path / "events.jsonl")
    ledger.mark_stage(
        run_id="run-1",
        stage="generation_round_1",
        status="completed",
        context_hash="context-a",
        artifact_hash="artifact-a",
    )
    assert ledger.stage_completed("run-1", "generation_round_1", "context-a") is True
    assert ledger.stage_completed("run-1", "generation_round_1", "context-b") is False


def test_existing_ledger_is_append_only(tmp_path: Path) -> None:
    ledger = ScientificLedger.create(tmp_path / "atlas.sqlite", tmp_path / "events.jsonl")
    ledger.add_candidate(_candidate())

    with pytest.raises(RuntimeError, match="append-only"):
        ledger.delete_candidate(_candidate().candidate_id)

