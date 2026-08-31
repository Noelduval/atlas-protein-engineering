from __future__ import annotations

import json
from pathlib import Path

from atlas.adaptive.ledger import ScientificLedger
from atlas.adaptive.models import (
    CandidateRecord,
    DesignStrategy,
    EvidenceAxis,
    EvidenceRecord,
)
from atlas.design.design_space import classify_design_space
from atlas.reporting.adaptive_outputs import build_adaptive_outputs
from atlas.reporting.portfolio import select_experimental_portfolio
from atlas.structure.reconstruct import reconstruct_active_like


SOURCE = Path(__file__).parents[2] / "data" / "23WN.cif"


def _candidate(mutation: str, region: str, strategy: DesignStrategy):
    reference = "ACDEFGHIKLMNPQRSTVWY" * 11
    reference = reference[:215]
    position = int(mutation[1:-1])
    residues = list(reference)
    residues[position - 1] = mutation[0]
    reference = "".join(residues)
    return CandidateRecord.create(
        reference_sequence=reference,
        mutations=[mutation],
        parents=(),
        strategy=strategy,
        structural_region=region,
        round_index=1,
        hypothesis="A falsifiable DP622 structural hypothesis.",
        intended_upside="Preserve a useful computational signal.",
        expected_risk="Computational evidence may not predict activity.",
    )


def _evidence(candidate, axis, value, uncertainty=0.1):
    return EvidenceRecord.numeric(
        candidate.candidate_id,
        axis,
        value=value,
        uncertainty=uncertainty,
        method=f"test-{axis.value}",
        provenance={"artifact_path": "test-artifact", "test_boundary": True},
        payload={"direction": "test-only"},
    )


def test_portfolio_preserves_regions_and_mutation_positions() -> None:
    candidates = (
        _candidate("C2S", "substrate_interface", DesignStrategy.SUBSTRATE_INTERFACE),
        _candidate("D3N", "second_shell", DesignStrategy.SECOND_SHELL_PREORGANIZATION),
        _candidate("E4Q", "distal_stability", DesignStrategy.STABILITY_SUPPORT),
    )
    evidence = {
        candidate.candidate_id: (
            _evidence(candidate, EvidenceAxis.STABILITY, float(index) / 10),
            _evidence(candidate, EvidenceAxis.SUBSTRATE_INTERFACE, 1.0 - index / 10),
            _evidence(candidate, EvidenceAxis.CATALYTIC_GEOMETRY, 0.2),
            _evidence(candidate, EvidenceAxis.LIABILITY, 0.1),
        )
        for index, candidate in enumerate(candidates)
    }

    selected = select_experimental_portfolio(candidates, evidence, target=3)

    assert len(selected) == 3
    assert len({candidate.structural_region for candidate in selected}) == 3
    assert len({candidate.mutations[0].position for candidate in selected}) == 3


def test_adaptive_outputs_write_dossier_exports_figures_and_report(tmp_path: Path) -> None:
    reference = tmp_path / "active_like.pdb"
    reconstruct_active_like(SOURCE, reference, tmp_path / "numbering.csv")
    design_space = classify_design_space(reference, SOURCE)
    actual_sequence = "".join(record.wildtype for record in design_space)
    candidate = CandidateRecord.create(
        reference_sequence=actual_sequence,
        mutations=[f"{actual_sequence[39]}40E"],
        parents=(),
        strategy=DesignStrategy.SUBSTRATE_INTERFACE,
        structural_region="substrate_interface",
        round_index=1,
        hypothesis="Test a resolved Aβ-facing chemical change.",
        intended_upside="Preserve substrate-pose support.",
        expected_risk="Static pose support may not imply cleavage activity.",
    )
    ledger = ScientificLedger.create(
        tmp_path / "design_memory" / "atlas_science.sqlite",
        tmp_path / "design_memory" / "events.jsonl",
    )
    ledger.add_candidate(candidate)
    ledger.add_evidence_many(
        tuple(
            _evidence(candidate, axis, value)
            for axis, value in (
                (EvidenceAxis.STABILITY, -0.3),
                (EvidenceAxis.STRUCTURE_QUALITY, 0.2),
                (EvidenceAxis.CATALYTIC_GEOMETRY, 0.3),
                (EvidenceAxis.SUBSTRATE_INTERFACE, 0.8),
                (EvidenceAxis.LIABILITY, 0.1),
            )
        )
    )
    reviews = [
        {
            "candidate_id": candidate.candidate_id,
            "decision": "PROMOTE",
            "strongest_case_for_synthesis": "Independent axes support a falsifiable test.",
            "strongest_case_against_synthesis": "No catalytic turnover was predicted.",
            "unresolved_uncertainty": "Static structural evidence is model-dependent.",
            "falsification_result": "No Aβ cleavage relative to reference.",
        }
    ]
    funnel = {
        "generated_evaluated": 5_001,
        "broad_survivors": 500,
        "structural_analyses": 100,
        "adversarial_review": 10,
        "finalists": 1,
        "langgraph_routes": 10,
    }

    bundle = build_adaptive_outputs(
        run_dir=tmp_path,
        ledger=ledger,
        design_space=design_space,
        finalist_ids=(candidate.candidate_id,),
        near_misses=[],
        funnel_counts=funnel,
        structure_records=[
            {
                "candidate_id": candidate.candidate_id,
                "structure_path": str(reference),
                "artifact_path": str(reference),
                "hard_violations": [],
            }
        ],
        adversarial_reviews=reviews,
        reference_pdb=reference,
        seed=622,
    )
    ledger.close()

    dossier = tmp_path / "reports" / "candidates" / f"{candidate.candidate_id}.md"
    assert dossier.is_file()
    text = dossier.read_text()
    assert "EXPERIMENTALLY UNTESTED" in text
    assert "Strongest argument against synthesis" in text
    assert "Falsifiable experimental hypothesis" in text
    assert "Replicated MD evidence" not in text
    assert (tmp_path / "exports" / "finalists.fasta").is_file()
    assert (tmp_path / "exports" / "fastas" / f"{candidate.candidate_id}.fasta").is_file()
    assert (tmp_path / "exports" / "finalist_evidence.csv").is_file()
    assert (tmp_path / "exports" / "finalist_evidence.json").is_file()
    assert (tmp_path / "exports" / "structures" / f"{candidate.candidate_id}.pdb").is_file()
    assert (tmp_path / "reports" / "candidates" / f"{candidate.candidate_id}_structure.png").is_file()
    assert bundle.final_report.is_file()
    assert "5,001" in bundle.final_report.read_text()
    assert "excluded from Atlas candidate discrimination" in bundle.final_report.read_text()
    assert (tmp_path / "figures" / "adaptive_design_funnel.png").is_file()
    assert (tmp_path / "figures" / "structural_design_map.png").is_file()
    assert (tmp_path / "README.md").is_file()
    assert (tmp_path / "FINALISTS.md").is_file()
    assert (tmp_path / "reports" / "limitations_report.md").is_file()
    assert json.loads(bundle.reproducibility_manifest.read_text())["candidate_library_sha256"]


def test_zero_finalist_report_keeps_near_miss_blockers_first_class(tmp_path: Path) -> None:
    reference = tmp_path / "active_like.pdb"
    reconstruct_active_like(SOURCE, reference, tmp_path / "numbering.csv")
    design_space = classify_design_space(reference, SOURCE)
    ledger = ScientificLedger.create(tmp_path / "ledger.sqlite", tmp_path / "events.jsonl")
    near_miss = {
        "candidate_id": "ATLAS-NEAR-MISS",
        "mutation_set": "A37P",
        "decision": "REJECT",
        "strongest_case_against_synthesis": "Static geometry did not support promotion.",
    }
    bundle = build_adaptive_outputs(
        run_dir=tmp_path,
        ledger=ledger,
        design_space=design_space,
        finalist_ids=(),
        near_misses=[near_miss],
        funnel_counts={
            "generated_evaluated": 5_000,
            "broad_survivors": 1,
            "structural_analyses": 1,
            "adversarial_review": 1,
            "finalists": 0,
            "langgraph_routes": 1,
        },
        structure_records=[],
        adversarial_reviews=[near_miss],
        reference_pdb=reference,
        seed=622,
    )
    ledger.close()

    report = bundle.final_report.read_text()
    assert "BEST_NEAR_MISS_HYPOTHESES" in report
    assert "Static geometry did not support promotion" in report
    assert "ATLAS-NEAR-MISS" in report
