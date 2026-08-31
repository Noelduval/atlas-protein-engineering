"""Checkpointed prospective-design pipeline for the Atlas adaptive mission."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from atlas.adaptive.critic import CriticPolicy, CriticRoute, Critique
from atlas.adaptive.ledger import ScientificLedger
from atlas.adaptive.models import (
    CandidateRecord,
    CompletionAudit,
    DesignStrategy,
    EvidenceAxis,
    EvidenceRecord,
    EvidenceStatus,
    FailureObservation,
    HardViolation,
)
from atlas.adaptive.repair import RepairGenerator
from atlas.adaptive.screening import (
    Direction,
    Evaluation,
    Objective,
    select_diverse_survivors,
)
from atlas.design.adaptive_generator import AdaptiveGenerator, SearchBudget
from atlas.design.design_space import (
    ResidueDesignRecord,
    classify_design_space,
    write_design_space,
)
from atlas.dynamics.models import DynamicsConfig
from atlas.run_context import prepare_run_directory
from atlas.structure.reconstruct import reconstruct_active_like


PIPELINE_PROTOCOL = "atlas-adaptive-prospective-design-v2-md-excluded"
GENERATION_POLICY_VERSION = "atlas-mechanism-aware-substitution-policy-v1"
EARLY_OBJECTIVES = (
    Objective(EvidenceAxis.STABILITY_MODEL_AWARE, Direction.MINIMIZE),
    Objective(EvidenceAxis.LIABILITY, Direction.MINIMIZE),
)
STRUCTURAL_OBJECTIVES = (
    Objective(EvidenceAxis.STABILITY_MODEL_AWARE, Direction.MINIMIZE),
    Objective(EvidenceAxis.SUBSTRATE_INTERFACE, Direction.MAXIMIZE),
    Objective(EvidenceAxis.CATALYTIC_GEOMETRY, Direction.MINIMIZE),
    Objective(EvidenceAxis.LIABILITY, Direction.MINIMIZE),
)


@dataclass(frozen=True)
class CandidateStructureResult:
    candidate_id: str
    structure_path: Path
    evidence: tuple[EvidenceRecord, ...]
    hard_violations: tuple[HardViolation, ...]
    artifact_path: Path


class AdaptiveScientificBackend(Protocol):
    def score_stability(
        self,
        reference_pdb: Path,
        candidates: tuple[CandidateRecord, ...],
        output_dir: Path,
    ) -> tuple[EvidenceRecord, ...]: ...

    def evaluate_structure(
        self,
        reference_pdb: Path,
        candidate: CandidateRecord,
        output_dir: Path,
        *,
        seed: int,
    ) -> CandidateStructureResult: ...

@dataclass(frozen=True)
class AdaptivePipelineConfig:
    input_structure: Path = Path("data/23WN.cif")
    output_root: Path = Path("outputs")
    atlas_repo: Path = Path(".")
    run_id: str | None = None
    thermompnn_repo: Path = Path(".external/ThermoMPNN")
    thermompnn_d_repo: Path = Path(".external/ThermoMPNN-D")
    dynamics_mode: str = field(
        default="excluded-reference-validation-failed", init=False
    )
    candidate_budget: int = 5_000
    round1_target: int = 1_200
    minimum_doubles: int = 750
    broad_target: int = 500
    structure_target: int = 100
    adversarial_target: int = 10
    portfolio_target: int = 5
    repair_parent_target: int = 6
    minimum_structural_regions: int = 4
    seed: int = 622
    relaxation_config: DynamicsConfig = field(default_factory=DynamicsConfig)
    resume: bool = False
    stop_after: str | None = None

    def __post_init__(self) -> None:
        if self.candidate_budget < 5_000:
            raise ValueError("Production adaptive search requires at least 5,000 candidates")
        targets = (
            self.broad_target,
            self.structure_target,
            self.adversarial_target,
            self.portfolio_target,
        )
        if any(target < 1 for target in targets):
            raise ValueError("Funnel targets must be positive")
        if not (
            self.candidate_budget >= self.broad_target >= self.structure_target
            >= self.adversarial_target >= self.portfolio_target
        ):
            raise ValueError("Funnel targets must be monotonically non-increasing")
        allowed = {
            None,
            "setup",
            "round1",
            "round2",
            "round3",
            "broad",
            "structure",
            "repair",
            "adversarial",
            "reports",
        }
        if self.stop_after not in allowed:
            raise ValueError(f"Unsupported adaptive stop stage: {self.stop_after}")


@dataclass(frozen=True)
class AdaptivePipelineResult:
    status: str
    run_dir: Path
    completion_audit: CompletionAudit
    finalist_ids: tuple[str, ...]


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _adaptive_context(config: AdaptivePipelineConfig, run_dir: Path) -> dict[str, Any]:
    base = json.loads((run_dir / "run_context.json").read_text())
    payload = {
        "protocol": PIPELINE_PROTOCOL,
        "base_run_context": base,
        "candidate_budget": config.candidate_budget,
        "round1_target": config.round1_target,
        "minimum_doubles": config.minimum_doubles,
        "funnel_targets": {
            "broad": config.broad_target,
            "structure": config.structure_target,
            "adversarial": config.adversarial_target,
            "portfolio": config.portfolio_target,
        },
        "repair_parent_target": config.repair_parent_target,
        "repair_children_per_parent": 1,
        "repair_generation_limit": 2,
        "minimum_structural_regions": config.minimum_structural_regions,
        "seed": config.seed,
        "generation_policy_version": GENERATION_POLICY_VERSION,
        "replicated_explicit_md": {
            "included_in_candidate_discrimination": False,
            "reason": "DP622/Aβ/Zn reference failed reproducible numerical and geometry validation.",
        },
        "claim_boundary": (
            "Prospective computational experiment prioritization; not proof of improved "
            "catalytic activity or therapeutic efficacy."
        ),
    }
    # Normalize tuples and other JSON-compatible containers once so an in-memory
    # context compares exactly with the persisted representation on resume.
    return json.loads(json.dumps(payload, sort_keys=True))


def _context_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _mark_stage(
    ledger: ScientificLedger,
    *,
    run_id: str,
    stage: str,
    context_hash: str,
    artifact: Path,
) -> None:
    ledger.mark_stage(
        run_id=run_id,
        stage=stage,
        status="completed",
        context_hash=context_hash,
        artifact_hash=_sha256(artifact),
    )


def _audit_payload(audit: CompletionAudit) -> dict[str, Any]:
    return {
        "unique_legal_evaluated": audit.unique_legal_evaluated,
        "candidate_budget": audit.candidate_budget,
        "strategies_used": sorted(audit.strategies_used),
        "required_strategies": sorted(audit.required_strategies),
        "structural_regions_explored": sorted(audit.structural_regions_explored),
        "minimum_structural_regions": audit.minimum_structural_regions,
        "pending_candidates": audit.pending_candidates,
        "pending_checkpoints": audit.pending_checkpoints,
        "eligible_unrepaired_near_misses": audit.eligible_unrepaired_near_misses,
        "finalists": audit.finalists,
        "near_misses_reported": audit.near_misses_reported,
        "promotable_without_hard_relaxation": audit.promotable_without_hard_relaxation,
        "blockers": list(audit.blockers),
        "complete": audit.complete,
    }


def _incomplete_audit(config: AdaptivePipelineConfig, evaluated: int) -> CompletionAudit:
    return CompletionAudit(
        unique_legal_evaluated=evaluated,
        candidate_budget=config.candidate_budget,
        strategies_used=frozenset(),
        required_strategies=frozenset(
            strategy.value
            for strategy in DesignStrategy
            if strategy is not DesignStrategy.REPAIR_RESCUE
        ),
        structural_regions_explored=frozenset(),
        minimum_structural_regions=config.minimum_structural_regions,
        pending_candidates=max(1, config.candidate_budget - evaluated),
        pending_checkpoints=1,
        eligible_unrepaired_near_misses=0,
        finalists=0,
        near_misses_reported=0,
        promotable_without_hard_relaxation=0,
    )


def _stop(
    config: AdaptivePipelineConfig,
    run_dir: Path,
    ledger: ScientificLedger,
    stage: str,
) -> AdaptivePipelineResult | None:
    if config.stop_after != stage:
        return None
    audit = _incomplete_audit(config, ledger.candidate_count())
    _write_json(
        run_dir / "execution_status.json",
        {
            "status": f"stopped_after_{stage}",
            "stage": stage,
            "scientific_conclusion": "NOT_YET_COMPLETE",
        },
    )
    return AdaptivePipelineResult(f"stopped_after_{stage}", run_dir, audit, ())


def _records_from_json(path: Path) -> tuple[ResidueDesignRecord, ...]:
    return tuple(ResidueDesignRecord.from_dict(item) for item in json.loads(path.read_text()))


def _prior_evidence(
    candidates: Iterable[CandidateRecord],
    design_space: tuple[ResidueDesignRecord, ...],
) -> tuple[EvidenceRecord, ...]:
    by_position = {record.position: record for record in design_space}
    records: list[EvidenceRecord] = []
    hydrophobic = set("AILMVFYW")
    charged = set("DEKRH")
    for candidate in candidates:
        residue_records = [by_position[m.position] for m in candidate.mutations]
        interface_support = max(
            1.0 / (1.0 + record.min_substrate_distance_a / 5.0)
            for record in residue_records
        )
        catalytic_risk = max(
            max(0.0, (10.0 - record.min_zinc_distance_a) / 10.0)
            for record in residue_records
        )
        liability = 0.0
        reasons: list[str] = []
        for mutation, residue in zip(candidate.mutations, residue_records):
            if mutation.mutant == "P":
                liability += 0.4
                reasons.append(f"{mutation.label}:proline_introduction")
            if mutation.mutant == "C":
                liability += 0.2
                reasons.append(f"{mutation.label}:unpaired_cysteine_risk")
            if residue.relative_sasa >= 0.5 and mutation.mutant in hydrophobic:
                liability += 0.15
                reasons.append(f"{mutation.label}:exposed_hydrophobe")
            if (mutation.wildtype in charged) != (mutation.mutant in charged):
                liability += 0.1
                reasons.append(f"{mutation.label}:charge_class_change")
        common = {
            "model_label": "active_like_inferred",
            "evidence_tier": "COMPUTATIONAL_SUPPORT",
            "applicability_caveat": "Deposited-coordinate hypothesis prior, not mutant activity.",
        }
        records.extend(
            (
                EvidenceRecord.numeric(
                    candidate.candidate_id,
                    EvidenceAxis.SUBSTRATE_INTERFACE,
                    value=interface_support,
                    uncertainty=0.25,
                    method="23WN deposited-distance hypothesis prior",
                    provenance=common,
                    payload={
                        "min_substrate_distance_a": min(
                            record.min_substrate_distance_a for record in residue_records
                        ),
                        "direction": "higher support",
                    },
                ),
                EvidenceRecord.numeric(
                    candidate.candidate_id,
                    EvidenceAxis.CATALYTIC_GEOMETRY,
                    value=catalytic_risk,
                    uncertainty=0.35,
                    method="23WN catalytic-proximity perturbation risk",
                    provenance=common,
                    payload={
                        "min_zinc_distance_a": min(
                            record.min_zinc_distance_a for record in residue_records
                        ),
                        "direction": "lower risk",
                    },
                ),
                EvidenceRecord.numeric(
                    candidate.candidate_id,
                    EvidenceAxis.LIABILITY,
                    value=min(1.0, liability),
                    uncertainty=0.2,
                    method="explicit sequence/structural liability rules",
                    provenance=common,
                    payload={"reasons": reasons, "direction": "lower liability"},
                ),
            )
        )
    return tuple(records)


def _group_evidence(
    records: Iterable[EvidenceRecord],
) -> dict[str, tuple[EvidenceRecord, ...]]:
    grouped: dict[str, list[EvidenceRecord]] = defaultdict(list)
    for record in records:
        grouped[record.candidate_id].append(record)
    return {candidate_id: tuple(values) for candidate_id, values in grouped.items()}


def _model_aware_stability_evidence(
    records: Iterable[EvidenceRecord],
) -> tuple[EvidenceRecord, ...]:
    """Create within-model empirical ranks without mixing upstream raw scales."""
    groups: dict[tuple[str, str], list[EvidenceRecord]] = defaultdict(list)
    for record in records:
        if (
            record.axis is not EvidenceAxis.STABILITY
            or record.status is not EvidenceStatus.AVAILABLE
            or record.value is None
        ):
            continue
        method = record.method.lower()
        if "thermompnn" not in method and "stability" not in method:
            continue
        model_commit = str(record.provenance.get("model_commit", "unspecified"))
        groups[(record.method, model_commit)].append(record)

    normalized: list[EvidenceRecord] = []
    for (method, model_commit), group in sorted(groups.items()):
        ordered = sorted(float(record.value) for record in group)
        denominator = len(ordered) - 1
        for record in group:
            raw = float(record.value)
            if denominator == 0:
                percentile = 0.5
                uncertainty = 0.5
            else:
                matching = [index for index, value in enumerate(ordered) if value == raw]
                percentile = (sum(matching) / len(matching)) / denominator
                uncertainty = min(0.25, 1.0 / (len(ordered) ** 0.5))
            normalized.append(
                EvidenceRecord.numeric(
                    record.candidate_id,
                    EvidenceAxis.STABILITY_MODEL_AWARE,
                    value=percentile,
                    uncertainty=uncertainty,
                    method=f"within-model empirical rank: {method}",
                    provenance={
                        "model_commit": model_commit,
                        "source_method": method,
                        "normalization": "within-model empirical percentile",
                        "engineering_policy": True,
                    },
                    payload={
                        "raw_value": raw,
                        "raw_axis": EvidenceAxis.STABILITY.value,
                        "raw_scale_not_cross_model_comparable": True,
                        "direction": "lower within-model percentile",
                    },
                )
            )
    return tuple(normalized)


def _evaluations(
    candidates: Iterable[CandidateRecord],
    evidence: Iterable[EvidenceRecord],
    violations: Mapping[str, tuple[HardViolation, ...]] | None = None,
) -> tuple[Evaluation, ...]:
    evidence_tuple = tuple(evidence)
    grouped = _group_evidence(
        evidence_tuple + _model_aware_stability_evidence(evidence_tuple)
    )
    hard = {} if violations is None else violations
    return tuple(
        Evaluation(
            candidate,
            grouped.get(candidate.candidate_id, ()),
            hard.get(candidate.candidate_id, ()),
        )
        for candidate in candidates
    )


def _latest_stability(records: Iterable[EvidenceRecord]) -> dict[str, float]:
    record_tuple = tuple(records)
    values: dict[str, float] = {}
    for record in record_tuple + _model_aware_stability_evidence(record_tuple):
        if (
            record.axis is EvidenceAxis.STABILITY_MODEL_AWARE
            and record.status is EvidenceStatus.AVAILABLE
            and record.value is not None
        ):
            values[record.candidate_id] = float(record.value)
    return values


def _learn_failure_patterns(
    candidates: tuple[CandidateRecord, ...],
    evidence: tuple[EvidenceRecord, ...],
    ledger: ScientificLedger,
    *,
    round_index: int,
) -> tuple[FailureObservation, ...]:
    stability = _latest_stability(evidence)
    patterns: Counter[tuple[str, str]] = Counter()
    exact: Counter[str] = Counter()
    for candidate in candidates:
        if stability.get(candidate.candidate_id, float("-inf")) < 0.75:
            continue
        for mutation in candidate.mutations:
            patterns[(candidate.structural_region, mutation.mutant)] += 1
            exact[mutation.label] += 1
    learned: list[FailureObservation] = []
    for (region, mutant), count in sorted(patterns.items()):
        if count < 3:
            continue
        learned.append(
            FailureObservation(
                candidate_id=None,
                category="recurrent_stability_regression",
                scope=f"substitution:{region}:{mutant}",
                detail=(
                    f"Round {round_index}: {count} candidates introducing {mutant} in "
                    f"{region} fell in the within-model regressive quartile."
                ),
                evidence_count=count,
                confidence=min(0.95, 0.55 + 0.03 * count),
                source="Round-specific ThermoMPNN/ThermoMPNN-D aggregation",
            )
        )
    for mutation, count in sorted(exact.items()):
        if count < 3:
            continue
        learned.append(
            FailureObservation(
                candidate_id=None,
                category="mutation_associated_stability_regression",
                scope=f"mutation:{mutation}",
                detail=(
                    f"Round {round_index}: {mutation} appeared in {count} regressive "
                    "candidate contexts; association is not asserted as a universal law."
                ),
                evidence_count=count,
                confidence=min(0.9, 0.5 + 0.025 * count),
                source="Context-counted stability aggregation",
            )
        )
    for observation in learned:
        ledger.record_failure(observation)
    return tuple(learned)


def _write_candidate_ids(path: Path, candidates: Iterable[CandidateRecord]) -> Path:
    return _write_json(path, [candidate.candidate_id for candidate in candidates])


def _load_candidates(path: Path, ledger: ScientificLedger) -> tuple[CandidateRecord, ...]:
    result = []
    for candidate_id in json.loads(path.read_text()):
        candidate = ledger.get_candidate(candidate_id)
        if candidate is None:
            raise RuntimeError(f"Stage artifact references missing candidate {candidate_id}")
        result.append(candidate)
    return tuple(result)


def _validate_stability_batch(
    candidates: tuple[CandidateRecord, ...], records: tuple[EvidenceRecord, ...]
) -> None:
    expected = {candidate.candidate_id for candidate in candidates}
    actual = {
        record.candidate_id
        for record in records
        if record.axis is EvidenceAxis.STABILITY
        and record.status is EvidenceStatus.AVAILABLE
        and record.value is not None
    }
    if actual != expected:
        raise RuntimeError(
            f"Stability backend returned candidates {len(actual)}/{len(expected)}; "
            f"missing={sorted(expected - actual)[:5]} extra={sorted(actual - expected)[:5]}"
        )


def _structure_payload(result: CandidateStructureResult) -> dict[str, Any]:
    return {
        "candidate_id": result.candidate_id,
        "structure_path": str(result.structure_path),
        "artifact_path": str(result.artifact_path),
        "hard_violations": [asdict(violation) for violation in result.hard_violations],
    }


def _load_structure_results(path: Path) -> tuple[dict[str, Any], ...]:
    records = tuple(json.loads(path.read_text()))
    for record in records:
        if not Path(record["structure_path"]).is_file():
            raise FileNotFoundError(record["structure_path"])
    return records


def _violation_map(records: Iterable[dict[str, Any]]) -> dict[str, tuple[HardViolation, ...]]:
    return {
        record["candidate_id"]: tuple(
            HardViolation(**violation) for violation in record.get("hard_violations", [])
        )
        for record in records
    }


def _choose_repair_proposal(
    proposals,
    failures: tuple[FailureObservation, ...],
):
    exact = {
        failure.scope.removeprefix("mutation:")
        for failure in failures
        if failure.scope.startswith("mutation:")
        and failure.evidence_count >= 3
        and failure.confidence >= 0.75
    }
    scoped = {
        failure.scope.removeprefix("substitution:")
        for failure in failures
        if failure.scope.startswith("substitution:")
        and failure.evidence_count >= 3
        and failure.confidence >= 0.75
    }
    ordered = sorted(
        proposals,
        key=lambda proposal: (
            any(mutation.label in exact for mutation in proposal.child.mutations),
            any(
                f"{proposal.child.structural_region}:{mutation.mutant}" in scoped
                for mutation in proposal.child.mutations
            ),
            proposal.child.candidate_id,
        ),
    )
    return None if not ordered else ordered[0]


def _evidence_delta(parent: Evaluation, child: Evaluation) -> dict[str, float]:
    left, right = parent.latest_by_axis(), child.latest_by_axis()
    return {
        axis.value: float(right[axis].value) - float(left[axis].value)
        for axis in sorted(set(left).intersection(right), key=lambda item: item.value)
    }


def _parent_child_structural_evidence(
    evaluation: Evaluation,
    ledger: ScientificLedger,
) -> list[dict[str, Any]]:
    if len(evaluation.candidate.mutations) != 2:
        return []
    child_axes = evaluation.latest_by_axis()
    records = ledger.all_evidence()
    comparisons: list[dict[str, Any]] = []
    for parent_id in evaluation.candidate.parents:
        parent = ledger.get_candidate(parent_id)
        if parent is None:
            continue
        parent_axes = _evaluations((parent,), records)[0].latest_by_axis()
        deltas = {}
        for axis in (
            EvidenceAxis.STRUCTURE_QUALITY,
            EvidenceAxis.CATALYTIC_GEOMETRY,
            EvidenceAxis.SUBSTRATE_INTERFACE,
        ):
            if axis in child_axes and axis in parent_axes:
                deltas[axis.value] = (
                    float(child_axes[axis].value) - float(parent_axes[axis].value)
                )
        comparisons.append(
            {
                "parent_id": parent_id,
                "parent_mutation_set": parent.mutation_set,
                "structural_axis_deltas_child_minus_parent": deltas,
                "raw_stability_arithmetic_performed": False,
            }
        )
    return comparisons


def _adversarial_review(
    evaluation: Evaluation,
    critique: Critique,
    *,
    parent_child_structural_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    by_axis = evaluation.latest_by_axis()
    available = {
        axis.value: {
            "value": record.value,
            "uncertainty": record.uncertainty,
            "method": record.method,
        }
        for axis, record in by_axis.items()
    }
    largest_uncertainty = max(
        by_axis.values(),
        key=lambda record: -1.0 if record.uncertainty is None else record.uncertainty,
        default=None,
    )
    return {
        "candidate_id": evaluation.candidate.candidate_id,
        "mutation_set": evaluation.candidate.mutation_set,
        "double_category": evaluation.candidate.double_category,
        "physical_coupling": evaluation.candidate.physical_coupling,
        "epistasis_uncertainty": evaluation.candidate.epistasis_uncertainty,
        "known_experiment_conflict": evaluation.candidate.known_experiment_conflict,
        "parent_child_structural_evidence": (
            []
            if parent_child_structural_evidence is None
            else parent_child_structural_evidence
        ),
        "blind_rank_order": True,
        "decision": critique.route.value,
        "promotion_blocker": critique.diagnosed_weakness,
        "critic_rationale": critique.rationale,
        "strongest_case_for_synthesis": (
            "The candidate preserves multiple independent computational-support axes "
            "and represents a falsifiable experiment."
        ),
        "strongest_case_against_synthesis": (
            "No computational axis demonstrates catalytic turnover; the historical "
            "benchmark shows stability/static geometry can disagree with activity."
        ),
        "unresolved_uncertainty": (
            "No available evidence"
            if largest_uncertainty is None
            else f"{largest_uncertainty.axis.value}: uncertainty={largest_uncertainty.uncertainty}"
        ),
        "falsification_result": (
            "Failure to express/fold, loss of Aβ cleavage relative to reference, or loss "
            "of the hypothesized structural behavior falsifies the design hypothesis."
        ),
        "evidence": available,
    }


def _portfolio(
    evaluations: tuple[Evaluation, ...], target: int
) -> tuple[Evaluation, ...]:
    if not evaluations:
        return ()
    initial = select_diverse_survivors(
        evaluations,
        STRUCTURAL_OBJECTIVES,
        target=min(target, len(evaluations)),
    )
    selected: list[Evaluation] = []
    used_positions: set[int] = set()
    for evaluation in initial:
        positions = {mutation.position for mutation in evaluation.candidate.mutations}
        if positions - used_positions or not selected:
            selected.append(evaluation)
            used_positions.update(positions)
    for evaluation in initial:
        if evaluation not in selected and len(selected) < target:
            selected.append(evaluation)
    return tuple(selected[:target])


def run_adaptive_pipeline(
    config: AdaptivePipelineConfig,
    *,
    backend: AdaptiveScientificBackend | None = None,
) -> AdaptivePipelineResult:
    """Execute or resume the approved prospective-design workflow."""
    run_dir = prepare_run_directory(config)
    run_id = run_dir.name
    adaptive_context = _adaptive_context(config, run_dir)
    adaptive_context_path = run_dir / "adaptive_run_context.json"
    if adaptive_context_path.is_file():
        recorded = json.loads(adaptive_context_path.read_text())
        if recorded != adaptive_context:
            raise RuntimeError("Refusing adaptive checkpoint with mismatched run context")
    else:
        _write_json(adaptive_context_path, adaptive_context)
    context_hash = _context_hash(adaptive_context)

    ledger_dir = run_dir / "design_memory"
    ledger_path = ledger_dir / "atlas_science.sqlite"
    events_path = ledger_dir / "events.jsonl"
    if ledger_path.is_file():
        if not config.resume:
            raise FileExistsError(f"Adaptive ledger already exists: {ledger_path}")
        ledger = ScientificLedger.open(ledger_path, events_path)
    else:
        ledger = ScientificLedger.create(ledger_path, events_path)

    if backend is None:
        from atlas.adaptive_backend import OfficialAdaptiveBackend

        backend = OfficialAdaptiveBackend(
            thermompnn_repo=config.thermompnn_repo,
            thermompnn_d_repo=config.thermompnn_d_repo,
            relaxation_config=config.relaxation_config,
            seed=config.seed,
        )

    try:
        search_dir = run_dir / "search"
        reconstructed = run_dir / "structures" / "DP622_active_like_inferred.pdb"
        mapping_path = run_dir / "structures" / "residue_numbering_map.csv"
        design_json = run_dir / "design_space.json"
        design_csv = run_dir / "design_space.csv"
        setup_artifact = run_dir / "setup_complete.json"
        md_exclusion_path = run_dir / "replicated_md_methodological_exclusion.json"
        if not md_exclusion_path.is_file():
            _write_json(
                md_exclusion_path,
                {
                    "reference_validation": "FAILED",
                    "included_in_candidate_discrimination": False,
                    "candidate_penalty_when_missing": False,
                    "statement": (
                        "Replicated explicit-solvent MD was evaluated as a prospective "
                        "evidence layer but excluded from Atlas candidate discrimination "
                        "after the DP622/Aβ/Zn reference failed reproducible numerical and "
                        "catalytic/substrate-geometry validation."
                    ),
                },
            )
        if ledger.stage_completed(run_id, "setup", context_hash):
            design_space = _records_from_json(design_json)
            if not reconstructed.is_file():
                raise FileNotFoundError(reconstructed)
        else:
            reconstructed.parent.mkdir(parents=True, exist_ok=True)
            reconstruct_active_like(config.input_structure, reconstructed, mapping_path)
            design_space = classify_design_space(reconstructed, config.input_structure)
            write_design_space(design_space, design_csv, design_json)
            _write_json(
                run_dir / "retrospective_method_characterization.json",
                {
                    "status": "BENCHMARK_FAILED",
                    "workflow_role": "retrospective_method_characterization",
                    "prospective_design_gate": False,
                    "frozen_observations": {
                        "Y91F": 0.173690,
                        "D126A": -0.028143,
                        "H172A": 0.200503,
                        "Y91F_D126A": -0.162948,
                    },
                    "source": "docs/benchmark_failure_analysis.md",
                    "statement": (
                        "The implementation and frozen gate ran correctly; selected stability/"
                        "static-geometry evidence did not reproduce every activity outcome."
                    ),
                },
            )
            _write_json(
                setup_artifact,
                {
                    "model_label": "active_like_inferred",
                    "positions": len(design_space),
                    "input_sha256": _sha256(Path(config.input_structure)),
                },
            )
            _mark_stage(
                ledger,
                run_id=run_id,
                stage="setup",
                context_hash=context_hash,
                artifact=setup_artifact,
            )
        stopped = _stop(config, run_dir, ledger, "setup")
        if stopped:
            return stopped

        generator = AdaptiveGenerator(design_space, seed=config.seed)
        budget = SearchBudget(
            candidate_budget=config.candidate_budget,
            round1_target=config.round1_target,
            minimum_doubles=config.minimum_doubles,
        )

        round1_path = search_dir / "round1_candidate_ids.json"
        if ledger.stage_completed(run_id, "round1", context_hash):
            round1 = _load_candidates(round1_path, ledger)
        else:
            round1 = generator.generate_round1(budget, failure_memory=ledger.failures())
            ledger.add_candidates(round1)
            stability = backend.score_stability(
                reconstructed, round1, run_dir / "stability" / "round1"
            )
            _validate_stability_batch(round1, stability)
            evidence = stability + _prior_evidence(round1, design_space)
            ledger.add_evidence_many(evidence)
            _learn_failure_patterns(round1, evidence, ledger, round_index=1)
            _write_candidate_ids(round1_path, round1)
            _mark_stage(
                ledger,
                run_id=run_id,
                stage="round1",
                context_hash=context_hash,
                artifact=round1_path,
            )
        stopped = _stop(config, run_dir, ledger, "round1")
        if stopped:
            return stopped

        round2_path = search_dir / "round2_candidate_ids.json"
        if ledger.stage_completed(run_id, "round2", context_hash):
            round2 = _load_candidates(round2_path, ledger)
        else:
            round1_evaluations = _evaluations(round1, ledger.all_evidence())
            priority_evaluations = select_diverse_survivors(
                round1_evaluations,
                EARLY_OBJECTIVES,
                target=min(250, len(round1)),
            )
            position_priority = tuple(
                mutation.position
                for evaluation in priority_evaluations
                for mutation in evaluation.candidate.mutations
            )
            memory_before = ledger.failures()
            round2 = generator.generate_round2(
                budget,
                round1=round1,
                position_priority=position_priority,
                failure_memory=memory_before,
            )
            ledger.add_candidates(round2)
            stability = backend.score_stability(
                reconstructed, round2, run_dir / "stability" / "round2"
            )
            _validate_stability_batch(round2, stability)
            evidence = stability + _prior_evidence(round2, design_space)
            ledger.add_evidence_many(evidence)
            _learn_failure_patterns(round2, evidence, ledger, round_index=2)
            _write_json(
                search_dir / "failure_memory_influence_round2.json",
                {
                    "memory_records_consulted": len(memory_before),
                    "high_confidence_scopes": sorted(
                        failure.scope
                        for failure in memory_before
                        if failure.evidence_count >= 3 and failure.confidence >= 0.75
                    ),
                    "position_priority": list(dict.fromkeys(position_priority)),
                    "effect": "Scoped patterns were placed after unpenalized proposals; not banned.",
                },
            )
            _write_candidate_ids(round2_path, round2)
            _mark_stage(
                ledger,
                run_id=run_id,
                stage="round2",
                context_hash=context_hash,
                artifact=round2_path,
            )
        stopped = _stop(config, run_dir, ledger, "round2")
        if stopped:
            return stopped

        round3_path = search_dir / "round3_candidate_ids.json"
        if ledger.stage_completed(run_id, "round3", context_hash):
            round3 = _load_candidates(round3_path, ledger)
        else:
            singles = round1 + round2
            single_evaluations = _evaluations(singles, ledger.all_evidence())
            preferred = select_diverse_survivors(
                single_evaluations,
                EARLY_OBJECTIVES,
                target=min(500, len(singles)),
            )
            memory_before = ledger.failures()
            round3 = generator.generate_round3(
                budget,
                singles=singles,
                preferred_candidate_ids=(
                    evaluation.candidate.candidate_id for evaluation in preferred
                ),
                failure_memory=memory_before,
            )
            ledger.add_candidates(round3)
            stability = backend.score_stability(
                reconstructed, round3, run_dir / "stability" / "round3"
            )
            _validate_stability_batch(round3, stability)
            evidence = stability + _prior_evidence(round3, design_space)
            ledger.add_evidence_many(evidence)
            _learn_failure_patterns(round3, evidence, ledger, round_index=3)
            _write_candidate_ids(round3_path, round3)
            _mark_stage(
                ledger,
                run_id=run_id,
                stage="round3",
                context_hash=context_hash,
                artifact=round3_path,
            )
        stopped = _stop(config, run_dir, ledger, "round3")
        if stopped:
            return stopped

        seed_candidates = round1 + round2 + round3
        if len(seed_candidates) != config.candidate_budget:
            raise RuntimeError(
                f"Adaptive rounds evaluated {len(seed_candidates)}/{config.candidate_budget}"
            )
        generation_summary_path = search_dir / "candidate_generation_summary.json"
        if not generation_summary_path.is_file():
            _write_json(
                generation_summary_path,
                {
                    "policy_version": GENERATION_POLICY_VERSION,
                    "unique_legal_candidates": len(seed_candidates),
                    "counts_by_round": dict(
                        sorted(Counter(str(c.round_index) for c in seed_candidates).items())
                    ),
                    "counts_by_strategy": dict(
                        sorted(Counter(c.strategy.value for c in seed_candidates).items())
                    ),
                    "counts_by_region": dict(
                        sorted(Counter(c.structural_region for c in seed_candidates).items())
                    ),
                    "counts_by_substitution_class": dict(
                        sorted(
                            Counter(
                                policy_class
                                for candidate in seed_candidates
                                for policy_class in candidate.substitution_classes
                            ).items()
                        )
                    ),
                    "counts_by_double_category": dict(
                        sorted(
                            Counter(
                                candidate.double_category
                                for candidate in seed_candidates
                                if candidate.double_category is not None
                            ).items()
                        )
                    ),
                    "known_retrospective_construct_excluded": "Y91F/D126A",
                },
            )
        normalized_seed_path = search_dir / "model_aware_stability_evidence.json"
        if not ledger.stage_completed(
            run_id, "model_aware_stability_seed", context_hash
        ):
            normalized_seed = _model_aware_stability_evidence(
                ledger.all_evidence()
            )
            ledger.add_evidence_many(normalized_seed)
            _write_json(
                normalized_seed_path,
                [record.to_dict() for record in normalized_seed],
            )
            _mark_stage(
                ledger,
                run_id=run_id,
                stage="model_aware_stability_seed",
                context_hash=context_hash,
                artifact=normalized_seed_path,
            )
        broad_path = search_dir / "broad_survivor_ids.json"
        if ledger.stage_completed(run_id, "broad", context_hash):
            broad = _load_candidates(broad_path, ledger)
        else:
            evaluations = _evaluations(seed_candidates, ledger.all_evidence())
            broad_eval = select_diverse_survivors(
                evaluations,
                EARLY_OBJECTIVES,
                target=min(config.broad_target, len(evaluations)),
            )
            broad = tuple(evaluation.candidate for evaluation in broad_eval)
            _write_candidate_ids(broad_path, broad)
            ledger.record_event(
                "broad_screen_complete",
                {
                    "evaluated": len(seed_candidates),
                    "survivors": len(broad),
                    "policy": "uncertainty-conservative Pareto layers plus region/strategy diversity",
                    "universal_score": False,
                },
            )
            _mark_stage(
                ledger,
                run_id=run_id,
                stage="broad",
                context_hash=context_hash,
                artifact=broad_path,
            )
        stopped = _stop(config, run_dir, ledger, "broad")
        if stopped:
            return stopped

        structure_path = run_dir / "structure" / "structure_results.json"
        structure_ids_path = search_dir / "structure_candidate_ids.json"
        if ledger.stage_completed(run_id, "structure", context_hash):
            structure_records = list(_load_structure_results(structure_path))
            structure_candidates = _load_candidates(structure_ids_path, ledger)
        else:
            broad_eval = _evaluations(broad, ledger.all_evidence())
            selected = select_diverse_survivors(
                broad_eval,
                EARLY_OBJECTIVES,
                target=min(config.structure_target, len(broad_eval)),
            )
            selected_candidates = [item.candidate for item in selected]
            selected_ids = {
                candidate.candidate_id for candidate in selected_candidates
            }
            for candidate in tuple(selected_candidates):
                if len(candidate.mutations) != 2:
                    continue
                for parent_id in candidate.parents:
                    parent = ledger.get_candidate(parent_id)
                    if parent is None or parent_id in selected_ids:
                        continue
                    selected_candidates.append(parent)
                    selected_ids.add(parent_id)
            structure_candidates = tuple(selected_candidates)
            structure_records: list[dict[str, Any]] = []
            evidence_batch: list[EvidenceRecord] = []
            for index, candidate in enumerate(structure_candidates):
                result = backend.evaluate_structure(
                    reconstructed,
                    candidate,
                    run_dir / "structure" / candidate.candidate_id,
                    seed=config.seed + index,
                )
                evidence_batch.extend(result.evidence)
                structure_records.append(_structure_payload(result))
                for violation in result.hard_violations:
                    ledger.record_failure(
                        FailureObservation(
                            candidate_id=candidate.candidate_id,
                            category=violation.code,
                            scope=f"candidate:{candidate.candidate_id}",
                            detail=violation.detail,
                            evidence_count=1,
                            confidence=1.0,
                            source="mutant-complex hard constraint",
                        )
                    )
            ledger.add_evidence_many(tuple(evidence_batch))
            _write_candidate_ids(structure_ids_path, structure_candidates)
            _write_json(structure_path, structure_records)
            _mark_stage(
                ledger,
                run_id=run_id,
                stage="structure",
                context_hash=context_hash,
                artifact=structure_path,
            )
        stopped = _stop(config, run_dir, ledger, "structure")
        if stopped:
            return stopped

        repair_path = run_dir / "repair" / "repair_trajectories.json"
        repair_children_path = run_dir / "repair" / "repair_child_ids.json"
        repair_eligibility_path = run_dir / "repair" / "repair_eligibility.json"
        if ledger.stage_completed(run_id, "repair", context_hash):
            repair_children = _load_candidates(repair_children_path, ledger)
            repair_records = tuple(
                record
                for record in _load_structure_results(
                    run_dir / "repair" / "repair_structure_results.json"
                )
            )
        else:
            violations = _violation_map(structure_records)
            structure_evaluations = _evaluations(
                structure_candidates, ledger.all_evidence(), violations
            )
            policy = CriticPolicy()
            revisable = tuple(
                evaluation
                for evaluation in structure_evaluations
                if policy.critique(evaluation).route is CriticRoute.REVISE
            )
            repairable_evaluations = select_diverse_survivors(
                revisable,
                STRUCTURAL_OBJECTIVES,
                target=min(config.repair_parent_target, len(revisable)),
            )
            repairable = [
                (evaluation, policy.critique(evaluation))
                for evaluation in repairable_evaluations
            ]
            eligible_ids = {
                evaluation.candidate.candidate_id for evaluation in repairable_evaluations
            }
            eligibility_records = [
                {
                    "candidate_id": evaluation.candidate.candidate_id,
                    "mutation_set": evaluation.candidate.mutation_set,
                    "critic_route": CriticRoute.REVISE.value,
                    "repair_eligible": evaluation.candidate.candidate_id in eligible_ids,
                    "reason": (
                        "selected into bounded, Pareto/diversity-preserving repair tier"
                        if evaluation.candidate.candidate_id in eligible_ids
                        else "outside the bounded top repair tier; retained as a documented soft tradeoff"
                    ),
                    "attempted": False,
                    "child_id": None,
                }
                for evaluation in revisable
            ]
            eligibility_by_id = {
                record["candidate_id"]: record for record in eligibility_records
            }
            repair_generator = RepairGenerator(
                design_space,
                max_children_per_parent_round=3,
                max_generations=2,
            )
            existing_sequences = {candidate.sequence for candidate in ledger.candidates()}
            repair_children_list: list[CandidateRecord] = []
            repair_records_list: list[dict[str, Any]] = []
            memory = ledger.failures()
            for parent_index, (parent_eval, critique) in enumerate(repairable):
                parent = parent_eval.candidate
                ledger.record_decision(
                    candidate_id=parent.candidate_id,
                    route="REVISE",
                    rationale=critique.rationale,
                    round_index=4,
                )
                proposals = repair_generator.propose(
                    parent,
                    critique,
                    round_index=4,
                    existing_sequences=existing_sequences,
                )
                proposal = _choose_repair_proposal(proposals, memory)
                if proposal is None:
                    eligibility_by_id[parent.candidate_id]["attempted"] = True
                    eligibility_by_id[parent.candidate_id]["reason"] = (
                        "bounded repair attempted; no novel legal proposal remained"
                    )
                    continue
                child = proposal.child
                eligibility_by_id[parent.candidate_id]["attempted"] = True
                eligibility_by_id[parent.candidate_id]["child_id"] = child.candidate_id
                ledger.add_candidate(child)
                ledger.record_repair(proposal)
                existing_sequences.add(child.sequence)
                stability = backend.score_stability(
                    reconstructed,
                    (child,),
                    run_dir / "stability" / "repair" / child.candidate_id,
                )
                _validate_stability_batch((child,), stability)
                ledger.add_evidence_many(stability + _prior_evidence((child,), design_space))
                result = backend.evaluate_structure(
                    reconstructed,
                    child,
                    run_dir / "repair" / "structures" / child.candidate_id,
                    seed=config.seed + 10_000 + parent_index,
                )
                ledger.add_evidence_many(result.evidence)
                repair_records_list.append(_structure_payload(result))
                child_eval = _evaluations(
                    (child,),
                    ledger.all_evidence(),
                    {child.candidate_id: result.hard_violations},
                )[0]
                child_critique = policy.critique(child_eval)
                ledger.record_decision(
                    candidate_id=child.candidate_id,
                    route=child_critique.route.value,
                    rationale=child_critique.rationale,
                    round_index=4,
                )
                ledger.record_repair_outcome(
                    parent_id=parent.candidate_id,
                    child_id=child.candidate_id,
                    evidence_delta=_evidence_delta(parent_eval, child_eval),
                    disposition=child_critique.route.value,
                )
                repair_children_list.append(child)
            repair_children = tuple(repair_children_list)
            repair_records = tuple(repair_records_list)
            _write_candidate_ids(repair_children_path, repair_children)
            repair_structure_path = run_dir / "repair" / "repair_structure_results.json"
            _write_json(repair_structure_path, repair_records)
            trajectories = {
                parent_id: list(ledger.repair_trajectory(parent_id))
                for parent_id in {
                    child.parents[0] for child in repair_children if child.parents
                }
            }
            _write_json(repair_path, trajectories)
            _write_json(repair_eligibility_path, eligibility_records)
            _write_json(
                run_dir / "repair" / "failure_memory_consulted.json",
                {
                    "records_consulted": len(memory),
                    "selection_policy": (
                        "High-confidence scoped failures deprioritized repair proposals but "
                        "did not become hard bans."
                    ),
                },
            )
            _mark_stage(
                ledger,
                run_id=run_id,
                stage="repair",
                context_hash=context_hash,
                artifact=repair_path,
            )
        stopped = _stop(config, run_dir, ledger, "repair")
        if stopped:
            return stopped

        all_structure_records = tuple(structure_records) + tuple(repair_records)
        all_structure_candidates = structure_candidates + repair_children
        all_violations = _violation_map(all_structure_records)

        adversarial_path = run_dir / "reports" / "adversarial_reviews.json"
        adversarial_ids_path = search_dir / "adversarial_candidate_ids.json"
        adversarial_repair_ids_path = (
            run_dir / "repair" / "adversarial_repair_child_ids.json"
        )
        adversarial_repair_structure_path = (
            run_dir / "repair" / "adversarial_repair_structure_results.json"
        )
        if ledger.stage_completed(run_id, "adversarial", context_hash):
            adversarial_candidates = _load_candidates(adversarial_ids_path, ledger)
            reviews = tuple(json.loads(adversarial_path.read_text()))
            adversarial_repair_children = _load_candidates(
                adversarial_repair_ids_path, ledger
            )
            adversarial_repair_structures = _load_structure_results(
                adversarial_repair_structure_path
            )
        else:
            structure_evaluations = _evaluations(
                all_structure_candidates,
                ledger.all_evidence(),
                all_violations,
            )
            policy = CriticPolicy()
            promotable = tuple(
                evaluation
                for evaluation in structure_evaluations
                if policy.critique(evaluation).route is CriticRoute.PROMOTE
            )
            selected = select_diverse_survivors(
                promotable,
                STRUCTURAL_OBJECTIVES,
                target=min(config.adversarial_target, len(promotable)),
            )
            repair_generator = RepairGenerator(
                design_space,
                max_children_per_parent_round=3,
                max_generations=2,
            )
            existing_sequences = {candidate.sequence for candidate in ledger.candidates()}
            memory = ledger.failures()
            adversarial_candidate_list: list[CandidateRecord] = []
            adversarial_repair_children_list: list[CandidateRecord] = []
            adversarial_repair_structure_list: list[dict[str, Any]] = []
            review_list: list[dict[str, Any]] = []
            repair_index = 0
            for starting_evaluation in selected:
                evaluation = starting_evaluation
                repair_chain: list[str] = []
                repair_exhaustion_reason: str | None = None
                while True:
                    critique = policy.critique(evaluation)
                    ledger.record_decision(
                        candidate_id=evaluation.candidate.candidate_id,
                        route=critique.route.value,
                        rationale=(
                            f"Adversarial review: {critique.rationale} Strongest objection: "
                            "computational evidence does not establish catalytic turnover."
                        ),
                        round_index=5 + evaluation.candidate.revision_generation,
                    )
                    if critique.route is not CriticRoute.REVISE:
                        break
                    if evaluation.candidate.revision_generation >= 2:
                        repair_exhaustion_reason = (
                            "maximum bounded repair generation (2) reached"
                        )
                        break
                    proposals = repair_generator.propose(
                        evaluation.candidate,
                        critique,
                        round_index=6 + evaluation.candidate.revision_generation,
                        existing_sequences=existing_sequences,
                    )
                    proposal = _choose_repair_proposal(proposals, memory)
                    if proposal is None:
                        repair_exhaustion_reason = (
                            "no novel legal bounded repair proposal remained"
                        )
                        break

                    parent_evaluation = evaluation
                    child = proposal.child
                    ledger.add_candidate(child)
                    ledger.record_repair(proposal)
                    existing_sequences.add(child.sequence)
                    stability = backend.score_stability(
                        reconstructed,
                        (child,),
                        run_dir
                        / "stability"
                        / "adversarial_repair"
                        / child.candidate_id,
                    )
                    _validate_stability_batch((child,), stability)
                    ledger.add_evidence_many(
                        stability + _prior_evidence((child,), design_space)
                    )
                    structure_result = backend.evaluate_structure(
                        reconstructed,
                        child,
                        run_dir
                        / "repair"
                        / "adversarial_structures"
                        / child.candidate_id,
                        seed=config.seed + 20_000 + repair_index,
                    )
                    repair_index += 1
                    ledger.add_evidence_many(structure_result.evidence)
                    structure_payload = _structure_payload(structure_result)
                    adversarial_repair_structure_list.append(structure_payload)
                    child_violations = list(structure_result.hard_violations)
                    for violation in structure_result.hard_violations:
                        ledger.record_failure(
                            FailureObservation(
                                candidate_id=child.candidate_id,
                                category=violation.code,
                                scope=f"candidate:{child.candidate_id}",
                                detail=violation.detail,
                                evidence_count=1,
                                confidence=1.0,
                                source="adversarial repair mutant-complex hard constraint",
                            )
                        )

                    evaluation = _evaluations(
                        (child,),
                        ledger.all_evidence(),
                        {child.candidate_id: tuple(child_violations)},
                    )[0]

                    child_critique = policy.critique(evaluation)
                    ledger.record_repair_outcome(
                        parent_id=parent_evaluation.candidate.candidate_id,
                        child_id=child.candidate_id,
                        evidence_delta=_evidence_delta(parent_evaluation, evaluation),
                        disposition=child_critique.route.value,
                    )
                    adversarial_repair_children_list.append(child)
                    repair_chain.append(child.candidate_id)

                final_critique = policy.critique(evaluation)
                review = _adversarial_review(
                    evaluation,
                    final_critique,
                    parent_child_structural_evidence=_parent_child_structural_evidence(
                        evaluation, ledger
                    ),
                )
                review.update(
                    {
                        "starting_candidate_id": starting_evaluation.candidate.candidate_id,
                        "repair_chain": repair_chain,
                        "repair_exhausted": (
                            final_critique.route is CriticRoute.REVISE
                            and repair_exhaustion_reason is not None
                        ),
                        "repair_exhaustion_reason": repair_exhaustion_reason,
                    }
                )
                adversarial_candidate_list.append(evaluation.candidate)
                review_list.append(review)

            adversarial_candidates = tuple(adversarial_candidate_list)
            adversarial_repair_children = tuple(adversarial_repair_children_list)
            adversarial_repair_structures = tuple(
                adversarial_repair_structure_list
            )
            reviews = tuple(review_list)
            _write_candidate_ids(adversarial_ids_path, adversarial_candidates)
            _write_candidate_ids(
                adversarial_repair_ids_path, adversarial_repair_children
            )
            _write_json(
                adversarial_repair_structure_path,
                adversarial_repair_structures,
            )
            _write_json(adversarial_path, reviews)
            _mark_stage(
                ledger,
                run_id=run_id,
                stage="adversarial",
                context_hash=context_hash,
                artifact=adversarial_path,
            )
        repair_children = repair_children + adversarial_repair_children
        all_structure_candidates = (
            all_structure_candidates + adversarial_repair_children
        )
        all_structure_records = (
            all_structure_records + adversarial_repair_structures
        )
        stopped = _stop(config, run_dir, ledger, "adversarial")
        if stopped:
            return stopped

        orchestration_path = run_dir / "orchestration" / "langgraph_routes.json"
        if not ledger.stage_completed(run_id, "orchestration", context_hash):
            from atlas.orchestration.graph import AtlasOrchestrator
            from atlas.orchestration.roles import EvidenceBackedRoleExecutor
            from atlas.orchestration.state import initial_state

            graph_results: list[dict[str, Any]] = []
            repair_parent_ids = sorted(
                {child.parents[0] for child in repair_children if child.parents}
            )
            graph_candidates = [*repair_parent_ids]
            graph_candidates.extend(
                candidate.candidate_id
                for candidate in adversarial_candidates
                if candidate.candidate_id not in graph_candidates
            )
            with AtlasOrchestrator.create(
                checkpoint_path=run_dir / "orchestration" / "langgraph_checkpoints.sqlite",
                role_executor=EvidenceBackedRoleExecutor(),
            ) as orchestrator:
                for index, candidate_id in enumerate(graph_candidates):
                    graph_run_id = f"{run_id}-candidate-{index:03d}-{candidate_id}"
                    state = initial_state(
                        run_id=graph_run_id,
                        ledger_path=ledger_path,
                        events_path=events_path,
                        round_index=4 if candidate_id in repair_parent_ids else 5,
                        active_candidate_id=candidate_id,
                    )
                    final_state = orchestrator.run(state)
                    graph_results.append(
                        {
                            "thread_id": graph_run_id,
                            "starting_candidate_id": candidate_id,
                            "final_candidate_id": final_state["active_candidate_id"],
                            "route": final_state["route"],
                            "repair_generation": final_state["repair_generation"],
                            "role_steps": final_state["role_step"],
                            "terminal_reason": final_state["terminal_reason"],
                        }
                    )
            _write_json(orchestration_path, graph_results)
            _mark_stage(
                ledger,
                run_id=run_id,
                stage="orchestration",
                context_hash=context_hash,
                artifact=orchestration_path,
            )

        report_dir = run_dir / "reports"
        report_marker = report_dir / "funnel_counts.json"
        final_normalized_path = report_dir / "model_aware_stability_evidence.json"
        if not ledger.stage_completed(
            run_id, "model_aware_stability_final", context_hash
        ):
            final_normalized = _model_aware_stability_evidence(
                ledger.all_evidence()
            )
            ledger.add_evidence_many(final_normalized)
            _write_json(
                final_normalized_path,
                [record.to_dict() for record in final_normalized],
            )
            _mark_stage(
                ledger,
                run_id=run_id,
                stage="model_aware_stability_final",
                context_hash=context_hash,
                artifact=final_normalized_path,
            )
        if ledger.stage_completed(run_id, "reports", context_hash):
            finalist_ids = tuple(json.loads((report_dir / "finalist_ids.json").read_text()))
            audit_data = json.loads((run_dir / "completion_audit.json").read_text())
            audit = CompletionAudit(
                unique_legal_evaluated=audit_data["unique_legal_evaluated"],
                candidate_budget=audit_data["candidate_budget"],
                strategies_used=frozenset(audit_data["strategies_used"]),
                required_strategies=frozenset(audit_data["required_strategies"]),
                structural_regions_explored=frozenset(
                    audit_data["structural_regions_explored"]
                ),
                minimum_structural_regions=audit_data["minimum_structural_regions"],
                pending_candidates=audit_data["pending_candidates"],
                pending_checkpoints=audit_data["pending_checkpoints"],
                eligible_unrepaired_near_misses=audit_data[
                    "eligible_unrepaired_near_misses"
                ],
                finalists=audit_data["finalists"],
                near_misses_reported=audit_data["near_misses_reported"],
                promotable_without_hard_relaxation=audit_data[
                    "promotable_without_hard_relaxation"
                ],
            )
        else:
            review_by_id = {review["candidate_id"]: review for review in reviews}
            promoted_ids = {
                review["candidate_id"]
                for review in reviews
                if review["decision"] == CriticRoute.PROMOTE.value
            }
            adversarial_evaluations = _evaluations(
                adversarial_candidates, ledger.all_evidence()
            )
            promoted_eval = tuple(
                evaluation
                for evaluation in adversarial_evaluations
                if evaluation.candidate.candidate_id in promoted_ids
            )
            finalists = _portfolio(promoted_eval, config.portfolio_target)
            finalist_ids = tuple(item.candidate.candidate_id for item in finalists)
            near_misses = [
                review
                for review in reviews
                if review["candidate_id"] not in finalist_ids
            ]
            if not finalists and not near_misses:
                near_misses = [
                    {
                        "candidate_id": candidate.candidate_id,
                        "mutation_set": candidate.mutation_set,
                        "decision": "NEAR_MISS",
                        "strongest_case_against_synthesis": (
                            "Candidate did not reach a complete adversarial evidence tier."
                        ),
                    }
                    for candidate in structure_candidates[:5]
                ]
            all_candidates = ledger.candidates()
            required_strategies = frozenset(
                strategy.value
                for strategy in DesignStrategy
                if strategy is not DesignStrategy.REPAIR_RESCUE
            )
            strategies_used = frozenset(
                candidate.strategy.value for candidate in all_candidates
            )
            regions = frozenset(
                candidate.structural_region
                for candidate in seed_candidates
                if candidate.structural_region != "cross_region"
            )
            unrepaired = sum(
                review["decision"] == CriticRoute.REVISE.value
                and not review.get("repair_exhausted", False)
                for review in reviews
            )
            audit = CompletionAudit(
                unique_legal_evaluated=len(seed_candidates) + len(repair_children),
                candidate_budget=config.candidate_budget,
                strategies_used=strategies_used,
                required_strategies=required_strategies,
                structural_regions_explored=regions,
                minimum_structural_regions=config.minimum_structural_regions,
                pending_candidates=0,
                pending_checkpoints=0,
                eligible_unrepaired_near_misses=unrepaired,
                finalists=len(finalist_ids),
                near_misses_reported=len(near_misses),
                promotable_without_hard_relaxation=0,
            )
            _write_json(report_dir / "finalist_ids.json", list(finalist_ids))
            _write_json(report_dir / "best_near_miss_hypotheses.json", near_misses)
            funnel = {
                "generated_evaluated": len(seed_candidates) + len(repair_children),
                "broad_survivors": len(broad),
                "structural_analyses": len(all_structure_records),
                "adversarial_review": len(adversarial_candidates),
                "finalists": len(finalist_ids),
                "langgraph_routes": len(json.loads(orchestration_path.read_text())),
            }
            _write_json(report_marker, funnel)
            _write_json(run_dir / "completion_audit.json", _audit_payload(audit))
            from atlas.reporting.adaptive_outputs import build_adaptive_outputs

            build_adaptive_outputs(
                run_dir=run_dir,
                ledger=ledger,
                design_space=design_space,
                finalist_ids=finalist_ids,
                near_misses=near_misses,
                funnel_counts=funnel,
                structure_records=list(all_structure_records),
                adversarial_reviews=list(reviews),
                reference_pdb=reconstructed,
                seed=config.seed,
            )
            ledger.record_event(
                "portfolio_selected",
                {
                    "finalist_ids": list(finalist_ids),
                    "portfolio_target": config.portfolio_target,
                    "universal_score": False,
                    "experimentally_untested": True,
                },
            )
            _mark_stage(
                ledger,
                run_id=run_id,
                stage="reports",
                context_hash=context_hash,
                artifact=report_marker,
            )
        stopped = _stop(config, run_dir, ledger, "reports")
        if stopped:
            return stopped

        status = "completed" if audit.complete else "exhaustion_audit_blocked"
        _write_json(
            run_dir / "execution_status.json",
            {
                "status": status,
                "stage": "complete" if audit.complete else "completion_audit",
                "scientific_conclusion": (
                    "EXPERIMENTALLY_UNTESTED_PORTFOLIO"
                    if finalist_ids
                    else "ZERO_FINALISTS_AFTER_EXHAUSTIVE_SEARCH"
                ),
                "completion_blockers": list(audit.blockers),
            },
        )
        return AdaptivePipelineResult(status, run_dir, audit, finalist_ids)
    finally:
        ledger.close()
