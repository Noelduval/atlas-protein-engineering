"""Checkpointed late-stage evidence integration and finalist gating."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
import subprocess
from typing import Any

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

from atlas.adaptive.ledger import ScientificLedger
from atlas.adaptive.models import (
    CandidateRecord,
    EvidenceAxis,
    EvidenceRecord,
    EvidenceStatus,
)
from atlas.late_stage.developability import screen_developability
from atlas.late_stage.dossiers import write_finalist_dossiers
from atlas.late_stage.pose_robustness import assess_local_pose_robustness
from atlas.late_stage.specificity import assess_specificity
from atlas.late_stage.structure_validation import validate_structure_prediction


_REQUIRED_LOCAL = (
    ("specificity", "intended_preferred"),
)

_PROTOCOL = "atlas-v1-late-stage-evidence-v1"


@dataclass(frozen=True)
class LateStageRunResult:
    status: str
    run_dir: Path
    finalist_ids: tuple[str, ...]
    summary_path: Path


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def _structure_records(run_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative in (
        "structure/structure_results.json",
        "repair/repair_structure_results.json",
        "repair/adversarial_repair_structure_results.json",
    ):
        path = run_dir / relative
        if not path.is_file():
            continue
        for record in json.loads(path.read_text()):
            records[str(record["candidate_id"])] = dict(record)
    return records


def _reference_sequence(reference_pdb: Path) -> str:
    structure = PDBParser(QUIET=True).get_structure("reference", reference_pdb)
    return "".join(seq1(residue.resname) for residue in structure[0]["A"])


def _latest(records: Sequence[EvidenceRecord], axis: EvidenceAxis) -> dict[str, Any]:
    matching = [record for record in records if record.axis is axis]
    return {} if not matching else matching[-1].to_dict()


def _axis_record(
    candidate_id: str,
    axis: EvidenceAxis,
    *,
    status: EvidenceStatus,
    value: float | None,
    method: str,
    artifact_path: Path,
    payload: Mapping[str, Any],
    implementation_commit: str,
) -> EvidenceRecord:
    return EvidenceRecord(
        candidate_id=candidate_id,
        axis=axis,
        status=status,
        value=value,
        uncertainty=None,
        method=method,
        provenance={
            "protocol": _PROTOCOL,
            "implementation_commit": implementation_commit,
            "artifact_path": str(artifact_path),
            "evidence_tier": "COMPUTATIONAL_SUPPORT",
        },
        payload=_jsonable(payload),
    )


def select_late_stage_finalists(
    records: Iterable[Mapping[str, Any]],
    *,
    baseline_order: Iterable[str],
    target: int = 5,
) -> tuple[tuple[str, ...], dict[str, dict[str, Any]]]:
    """Apply independent hard evidence gates without constructing a universal score."""
    if not 0 <= target <= 5:
        raise ValueError("late-stage finalist target must be between zero and five")
    by_id = {str(record["candidate_id"]): record for record in records}
    decisions: dict[str, dict[str, Any]] = {}
    for candidate_id, record in by_id.items():
        blockers: list[str] = []
        if not bool(record.get("baseline_promoted")):
            blockers.append("baseline adversarial review did not promote the candidate")
        for axis, passing_classification in _REQUIRED_LOCAL:
            evidence = record.get(axis, {})
            status = str(evidence.get("status", "unavailable"))
            classification = str(evidence.get("classification", "indeterminate"))
            if status != "available":
                blockers.append(f"{axis.replace('_', ' ')} evidence is {status}")
            elif classification != passing_classification:
                blockers.append(
                    f"{axis.replace('_', ' ')} classification is {classification}"
                )
        developability = record.get("developability", {})
        if bool(developability.get("candidate_introduced_hard_blocker", False)):
            blockers.append("developability has a candidate-introduced hard blocker")
        orthogonal = record.get("orthogonal_structure", {})
        decisions[candidate_id] = {
            "eligible": not blockers,
            "blockers": blockers,
            "orthogonal_structure_unavailable": (
                str(orthogonal.get("status", "unavailable")) != "available"
            ),
            "universal_score_used": False,
        }
    ordered = [candidate_id for candidate_id in baseline_order if candidate_id in by_id]
    ordered.extend(sorted(set(by_id) - set(ordered)))
    selected = tuple(
        candidate_id
        for candidate_id in ordered
        if decisions[candidate_id]["eligible"]
    )[:target]
    return selected, decisions


def run_late_stage(
    *,
    run_dir: str | Path,
    input_structure: str | Path = Path("data/23WN.cif"),
    predictor_command: Sequence[str] | None = None,
    seed: int = 622,
    resume: bool = False,
) -> LateStageRunResult:
    """Evaluate only the persisted adversarial set and write resumable artifacts."""
    root = Path(run_dir)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )
    implementation_commit = revision.stdout.strip()
    if revision.returncode or len(implementation_commit) != 40:
        raise RuntimeError("Late-stage implementation commit cannot be resolved")
    output = root / "late_stage"
    completion_path = output / "completion.json"
    if completion_path.is_file():
        if not resume:
            raise FileExistsError(f"Late-stage outputs already exist: {completion_path}")
        completed = json.loads(completion_path.read_text())
        return LateStageRunResult(
            status=str(completed["status"]),
            run_dir=root,
            finalist_ids=tuple(completed["finalist_ids"]),
            summary_path=completion_path,
        )
    execution_path = root / "execution_status.json"
    if not execution_path.is_file() or json.loads(execution_path.read_text()).get(
        "status"
    ) != "completed":
        raise RuntimeError("Late-stage evaluation requires a completed adaptive run")
    source = Path(input_structure)
    if not source.is_file():
        raise FileNotFoundError(f"23WN input does not exist: {source}")
    reference = root / "structures" / "DP622_active_like_inferred.pdb"
    if not reference.is_file():
        raise FileNotFoundError(reference)
    candidate_ids = tuple(
        json.loads((root / "search" / "adversarial_candidate_ids.json").read_text())
    )
    if not candidate_ids:
        raise RuntimeError("Completed run contains no adversarial candidates")
    reviews = json.loads((root / "reports" / "adversarial_reviews.json").read_text())
    review_by_id = {str(review["candidate_id"]): review for review in reviews}
    baseline_finalists = tuple(
        json.loads((root / "reports" / "finalist_ids.json").read_text())
    )
    baseline_order = baseline_finalists + tuple(
        candidate_id for candidate_id in candidate_ids if candidate_id not in baseline_finalists
    )
    structures = _structure_records(root)
    reference_sequence = _reference_sequence(reference)
    ledger_path = root / "design_memory" / "atlas_science.sqlite"
    events_path = root / "design_memory" / "events.jsonl"
    ledger = ScientificLedger.open(ledger_path, events_path)
    evidence_to_add: list[EvidenceRecord] = []
    candidate_payloads: list[dict[str, Any]] = []
    try:
        already_committed = {
            (record.candidate_id, record.axis)
            for record in ledger.all_evidence()
            if record.provenance.get("protocol") == _PROTOCOL
        }
        for candidate_id in candidate_ids:
            candidate = ledger.get_candidate(candidate_id)
            if candidate is None:
                raise RuntimeError(f"Adversarial candidate is absent from ledger: {candidate_id}")
            candidate_dir = output / "candidates" / candidate_id
            structure_record = structures.get(candidate_id)
            candidate_pdb = (
                None if structure_record is None else Path(structure_record["structure_path"])
            )
            orthogonal = validate_structure_prediction(
                candidate_id=candidate_id,
                sequence=candidate.sequence,
                reference_pdb=reference,
                output_dir=candidate_dir / "orthogonal_structure",
                predictor_command=predictor_command,
            )
            orthogonal_payload = _jsonable(orthogonal)
            orthogonal_artifact = candidate_dir / "orthogonal_structure" / "structure_validation.json"
            orthogonal_value = (
                None
                if orthogonal.alignment is None
                else float(orthogonal.alignment["ca_rmsd_a"])
            )
            orthogonal_status = EvidenceStatus(orthogonal.status.value)

            if candidate_pdb is None or not candidate_pdb.is_file():
                specificity_payload = {
                    "status": "unavailable",
                    "classification": "indeterminate",
                    "warnings": ["Candidate-specific complex is unavailable."],
                }
                pose_payload = dict(specificity_payload)
                developability_payload = dict(specificity_payload)
                specificity_status = pose_status = developability_status = EvidenceStatus.UNAVAILABLE
                specificity_value = pose_value = developability_value = None
                specificity_artifact = candidate_dir / "specificity.json"
                pose_artifact = candidate_dir / "pose_robustness.json"
                developability_artifact = candidate_dir / "developability.json"
                _write_json(specificity_artifact, specificity_payload)
                _write_json(pose_artifact, pose_payload)
                _write_json(developability_artifact, developability_payload)
            else:
                specificity = assess_specificity(candidate_pdb, source, variant_id=candidate_id)
                specificity_payload = _jsonable(specificity)
                specificity_payload["status"] = "available"
                specificity_artifact = _write_json(
                    candidate_dir / "specificity.json", specificity_payload
                )
                specificity_status = EvidenceStatus.AVAILABLE
                specificity_value = specificity.preference_margin

                pose = assess_local_pose_robustness(
                    candidate_pdb,
                    reference_pdb=reference,
                    output_dir=candidate_dir / "pose_perturbations",
                    variant_id=candidate_id,
                )
                pose_payload = _jsonable(pose)
                pose_payload["status"] = (
                    "available" if pose.classification != "indeterminate" else "invalid"
                )
                pose_artifact = _write_json(
                    candidate_dir / "pose_robustness.json", pose_payload
                )
                pose_status = (
                    EvidenceStatus.AVAILABLE
                    if pose.classification != "indeterminate"
                    else EvidenceStatus.INVALID
                )
                pose_value = pose.max_substrate_pose_drift_a

                developability = screen_developability(
                    candidate_id=candidate_id,
                    sequence=candidate.sequence,
                    reference_sequence=reference_sequence,
                    structure_path=candidate_pdb,
                    output_dir=candidate_dir / "developability",
                )
                developability_payload = developability.to_dict()
                high_flags = [
                    flag for flag in developability.flags if flag.severity == "high"
                ]
                developability_payload["classification"] = (
                    "acceptable_with_flags"
                    if developability.status == "completed" and not high_flags
                    else "review_required"
                )
                developability_payload["status"] = (
                    "available" if developability.status == "completed" else "invalid"
                )
                developability_artifact = developability.artifact_path
                _write_json(developability_artifact, developability_payload)
                developability_status = (
                    EvidenceStatus.AVAILABLE
                    if developability.status == "completed"
                    else EvidenceStatus.INVALID
                )
                developability_value = float(len(developability.flags))

            axis_records = (
                _axis_record(
                    candidate_id,
                    EvidenceAxis.ORTHOGONAL_STRUCTURE,
                    status=orthogonal_status,
                    value=orthogonal_value,
                    method="real external sequence-to-structure adapter",
                    artifact_path=orthogonal_artifact,
                    payload=orthogonal_payload,
                    implementation_commit=implementation_commit,
                ),
                _axis_record(
                    candidate_id,
                    EvidenceAxis.SUBSTRATE_SPECIFICITY,
                    status=specificity_status,
                    value=specificity_value,
                    method="bounded Aβ42 local-window contact-compatibility panel",
                    artifact_path=specificity_artifact,
                    payload=specificity_payload,
                    implementation_commit=implementation_commit,
                ),
                _axis_record(
                    candidate_id,
                    EvidenceAxis.POSE_ROBUSTNESS,
                    status=pose_status,
                    value=pose_value,
                    method="deterministic resolved-pose local perturbations",
                    artifact_path=pose_artifact,
                    payload=pose_payload,
                    implementation_commit=implementation_commit,
                ),
                _axis_record(
                    candidate_id,
                    EvidenceAxis.DEVELOPABILITY,
                    status=developability_status,
                    value=developability_value,
                    method="deterministic sequence/static-structure risk heuristics",
                    artifact_path=developability_artifact,
                    payload=developability_payload,
                    implementation_commit=implementation_commit,
                ),
            )
            evidence_to_add.extend(
                record
                for record in axis_records
                if (record.candidate_id, record.axis) not in already_committed
            )
            candidate_payloads.append(
                {
                    "candidate_id": candidate_id,
                    "baseline_promoted": review_by_id.get(candidate_id, {}).get("decision")
                    == "PROMOTE",
                    "orthogonal_structure": {
                        "status": orthogonal.status.value,
                        "classification": (
                            "recovered"
                            if orthogonal.catalytic_site_recovery
                            and orthogonal.catalytic_site_recovery.get(
                                "recovered_by_engineering_threshold"
                            )
                            else "indeterminate"
                        ),
                        "evidence": orthogonal_payload,
                    },
                    "specificity": specificity_payload,
                    "pose_robustness": pose_payload,
                    "developability": developability_payload,
                    "candidate_structure": None if candidate_pdb is None else str(candidate_pdb),
                }
            )
        ledger.add_evidence_many(tuple(evidence_to_add))
        selected, decisions = select_late_stage_finalists(
            candidate_payloads, baseline_order=baseline_order, target=5
        )
        _write_json(output / "candidate_evidence.json", candidate_payloads)
        _write_json(output / "finalist_decisions.json", decisions)
        _write_json(output / "finalist_ids.json", selected)

        payload_by_id = {item["candidate_id"]: item for item in candidate_payloads}
        all_evidence = ledger.all_evidence()
        selected_candidates: list[CandidateRecord] = []
        dossier_evidence: dict[str, dict[str, Any]] = {}
        for candidate_id in selected:
            candidate = ledger.get_candidate(candidate_id)
            if candidate is None:
                raise RuntimeError(candidate_id)
            selected_candidates.append(candidate)
            records = [item for item in all_evidence if item.candidate_id == candidate_id]
            stability = [item.to_dict() for item in records if item.axis is EvidenceAxis.STABILITY]
            model_evidence = {
                "ThermoMPNN": [item for item in stability if "ThermoMPNN-D" not in item["method"]],
                "ThermoMPNN-D": [item for item in stability if "ThermoMPNN-D" in item["method"]],
            }
            candidate_late = payload_by_id[candidate_id]
            disagreements = []
            if candidate_late["orthogonal_structure"]["status"] != "available":
                disagreements.append("Orthogonal sequence-to-structure validation is unavailable.")
            disagreements.extend(candidate_late["specificity"].get("warnings", ()))
            disagreements.extend(candidate_late["pose_robustness"].get("warnings", ()))
            dossier_evidence[candidate_id] = {
                "why_atlas_proposed": {
                    "strategy": candidate.strategy.value,
                    "hypothesis": candidate.hypothesis,
                    "baseline_adversarial_review": review_by_id[candidate_id],
                    "selection_policy": "Independent gates and baseline diversity order; no universal score.",
                },
                "thermompnn": model_evidence,
                "orthogonal_structure": candidate_late["orthogonal_structure"],
                "catalytic_preorganization": _latest(records, EvidenceAxis.CATALYTIC_GEOMETRY),
                "abeta_contacts": _latest(records, EvidenceAxis.SUBSTRATE_INTERFACE),
                "specificity": candidate_late["specificity"],
                "pose_robustness": candidate_late["pose_robustness"],
                "developability": candidate_late["developability"],
                "disagreements_uncertainty": disagreements
                or ["All evidence remains computational and model-dependent."],
                "reason_it_beat_near_misses": (
                    "It retained baseline adversarial promotion and passed every required "
                    "candidate-specific late-stage local evidence gate without combining scales."
                ),
                "falsification_criteria": [
                    "Expression or folding is worse than the matched active-like reference.",
                    "Cleavage-site-resolved Aβ assays do not support the proposed context behavior.",
                    "Zn-dependence controls contradict the proposed catalytic mechanism.",
                    "Non-cognate profiling contradicts the bounded specificity hypothesis.",
                ],
            }
        near_misses = [
            {**review_by_id.get(candidate_id, {}), **decisions[candidate_id]}
            for candidate_id in candidate_ids
            if candidate_id not in selected
        ]
        write_finalist_dossiers(
            output_dir=output / "dossiers",
            finalists=selected_candidates,
            candidate_structures={
                candidate_id: payload_by_id[candidate_id]["candidate_structure"]
                for candidate_id in selected
            },
            late_stage_evidence=dossier_evidence,
            near_misses=near_misses,
        )
        ledger.record_event(
            "late_stage_portfolio_selected",
            {
                "protocol": _PROTOCOL,
                "implementation_commit": implementation_commit,
                "finalist_ids": list(selected),
                "universal_score": False,
                "experimentally_untested": True,
                "seed": seed,
            },
        )
        completed = {
            "status": "completed",
            "protocol": _PROTOCOL,
            "implementation_commit": implementation_commit,
            "seed": seed,
            "evaluated_candidate_ids": list(candidate_ids),
            "finalist_ids": list(selected),
            "orthogonal_structure_available": sum(
                item["orthogonal_structure"]["status"] == "available"
                for item in candidate_payloads
            ),
            "specificity_available": sum(
                item["specificity"].get("status") == "available"
                for item in candidate_payloads
            ),
            "pose_robustness_available": sum(
                item["pose_robustness"].get("status") == "available"
                for item in candidate_payloads
            ),
            "developability_available": sum(
                item["developability"].get("status") == "available"
                for item in candidate_payloads
            ),
        }
        _write_json(completion_path, completed)
        return LateStageRunResult("completed", root, selected, completion_path)
    finally:
        ledger.close()
