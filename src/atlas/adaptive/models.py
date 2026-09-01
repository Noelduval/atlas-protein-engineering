"""Typed records shared by adaptive generation, evaluation, and orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import re
from typing import Any, Iterable, Mapping


STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_MUTATION_PATTERN = re.compile(r"^([A-Z])(\d+)([A-Z])$")


class DesignStrategy(str, Enum):
    SUBSTRATE_INTERFACE = "substrate_interface_explorer"
    STABILITY_SUPPORT = "stability_support_explorer"
    SECOND_SHELL_PREORGANIZATION = "second_shell_preorganization_explorer"
    CONSERVATIVE = "conservative_explorer"
    EVIDENCE_GUIDED_COMBINATION = "evidence_guided_combination_explorer"
    REPAIR_RESCUE = "repair_rescue_explorer"


class EvidenceAxis(str, Enum):
    STABILITY = "stability"
    STABILITY_MODEL_AWARE = "stability_model_aware"
    STRUCTURE_QUALITY = "structure_quality"
    CATALYTIC_GEOMETRY = "catalytic_geometry"
    SUBSTRATE_INTERFACE = "substrate_interface"
    DYNAMICS = "dynamics"
    LIABILITY = "liability"
    ACTIVITY_ORIENTED = "activity_oriented"
    ORTHOGONAL_STRUCTURE = "orthogonal_structure"
    SUBSTRATE_SPECIFICITY = "substrate_specificity"
    POSE_ROBUSTNESS = "pose_robustness"
    DEVELOPABILITY = "developability"


class EvidenceStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


class CandidateDisposition(str, Enum):
    PENDING = "pending"
    HARD_REJECTED = "hard_rejected"
    SOFT_REJECTED = "soft_rejected"
    REVISE = "revise"
    PROMOTED = "promoted"
    FINALIST = "finalist"
    NEAR_MISS = "near_miss"


@dataclass(frozen=True, order=True)
class Mutation:
    position: int
    wildtype: str
    mutant: str

    @classmethod
    def parse(cls, text: str) -> Mutation:
        match = _MUTATION_PATTERN.fullmatch(text.strip())
        if match is None:
            raise ValueError(f"Invalid mutation notation: {text!r}")
        wildtype, position_text, mutant = match.groups()
        position = int(position_text)
        if position < 1:
            raise ValueError(f"Mutation position must be positive: {text!r}")
        if wildtype not in STANDARD_AMINO_ACIDS or mutant not in STANDARD_AMINO_ACIDS:
            raise ValueError(f"Mutation must use standard amino acids: {text!r}")
        if wildtype == mutant:
            raise ValueError(f"Mutation does not change the sequence: {text!r}")
        return cls(position=position, wildtype=wildtype, mutant=mutant)

    @property
    def label(self) -> str:
        return f"{self.wildtype}{self.position}{self.mutant}"


def _normalize_mutations(
    reference_sequence: str, mutations: Iterable[str | Mutation]
) -> tuple[Mutation, ...]:
    sequence = reference_sequence.strip().upper()
    if not sequence or any(amino_acid not in STANDARD_AMINO_ACIDS for amino_acid in sequence):
        raise ValueError("Reference sequence must contain only standard amino acids")
    parsed = tuple(
        mutation if isinstance(mutation, Mutation) else Mutation.parse(mutation)
        for mutation in mutations
    )
    if not parsed:
        raise ValueError("A prospective candidate must contain at least one mutation")
    ordered = tuple(sorted(parsed, key=lambda item: (item.position, item.mutant)))
    positions = [mutation.position for mutation in ordered]
    if len(positions) != len(set(positions)):
        raise ValueError("A candidate cannot contain multiple mutations at one position")
    for mutation in ordered:
        if mutation.position > len(sequence):
            raise ValueError(
                f"Mutation {mutation.label} is outside the {len(sequence)}-residue sequence"
            )
        observed = sequence[mutation.position - 1]
        if observed != mutation.wildtype:
            raise ValueError(
                f"Mutation {mutation.label} expects {mutation.wildtype} at position "
                f"{mutation.position}, observed {observed}"
            )
    return ordered


def _mutated_sequence(reference_sequence: str, mutations: tuple[Mutation, ...]) -> str:
    residues = list(reference_sequence.strip().upper())
    for mutation in mutations:
        residues[mutation.position - 1] = mutation.mutant
    return "".join(residues)


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    sequence: str
    mutations: tuple[Mutation, ...]
    parents: tuple[str, ...]
    strategy: DesignStrategy
    structural_region: str
    round_index: int
    hypothesis: str
    intended_upside: str
    expected_risk: str
    intended_physical_change: str = "Not separately annotated in the legacy record."
    feature_to_preserve: str = "Resolved scaffold and active-site integrity."
    principal_biochemical_risk: str = "Unresolved mutation-specific biochemical risk."
    metal_liability: str = "none_identified"
    requires_candidate_geometry: bool = False
    double_category: str | None = None
    physical_coupling: str | None = None
    epistasis_uncertainty: str | None = None
    known_experiment_conflict: str = "none_identified"
    substitution_classes: tuple[str, ...] = ()
    revision_generation: int = 0
    disposition: CandidateDisposition = CandidateDisposition.PENDING

    @classmethod
    def create(
        cls,
        *,
        reference_sequence: str,
        mutations: Iterable[str | Mutation],
        parents: Iterable[str],
        strategy: DesignStrategy,
        structural_region: str,
        round_index: int,
        hypothesis: str,
        intended_upside: str,
        expected_risk: str,
        intended_physical_change: str | None = None,
        feature_to_preserve: str | None = None,
        principal_biochemical_risk: str | None = None,
        metal_liability: str = "none_identified",
        requires_candidate_geometry: bool = False,
        double_category: str | None = None,
        physical_coupling: str | None = None,
        epistasis_uncertainty: str | None = None,
        known_experiment_conflict: str = "none_identified",
        substitution_classes: Iterable[str] = (),
        revision_generation: int = 0,
        disposition: CandidateDisposition = CandidateDisposition.PENDING,
    ) -> CandidateRecord:
        normalized = _normalize_mutations(reference_sequence, mutations)
        sequence = _mutated_sequence(reference_sequence, normalized)
        if round_index < 1:
            raise ValueError("round_index must be positive")
        if revision_generation < 0:
            raise ValueError("revision_generation cannot be negative")
        if not structural_region.strip():
            raise ValueError("structural_region is required")
        for name, value in {
            "hypothesis": hypothesis,
            "intended_upside": intended_upside,
            "expected_risk": expected_risk,
        }.items():
            if not value.strip():
                raise ValueError(f"{name} is required")
        digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()[:16].upper()
        return cls(
            candidate_id=f"ATLAS-{digest}",
            sequence=sequence,
            mutations=normalized,
            parents=tuple(parents),
            strategy=DesignStrategy(strategy),
            structural_region=structural_region,
            round_index=round_index,
            hypothesis=hypothesis,
            intended_upside=intended_upside,
            expected_risk=expected_risk,
            intended_physical_change=(
                intended_upside
                if intended_physical_change is None
                else intended_physical_change
            ),
            feature_to_preserve=(
                hypothesis if feature_to_preserve is None else feature_to_preserve
            ),
            principal_biochemical_risk=(
                expected_risk
                if principal_biochemical_risk is None
                else principal_biochemical_risk
            ),
            metal_liability=metal_liability,
            requires_candidate_geometry=bool(requires_candidate_geometry),
            double_category=double_category,
            physical_coupling=physical_coupling,
            epistasis_uncertainty=epistasis_uncertainty,
            known_experiment_conflict=known_experiment_conflict,
            substitution_classes=tuple(substitution_classes),
            revision_generation=revision_generation,
            disposition=CandidateDisposition(disposition),
        )

    @property
    def mutation_set(self) -> str:
        return "/".join(mutation.label for mutation in self.mutations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "sequence": self.sequence,
            "mutations": [asdict(mutation) for mutation in self.mutations],
            "parents": list(self.parents),
            "strategy": self.strategy.value,
            "structural_region": self.structural_region,
            "round_index": self.round_index,
            "hypothesis": self.hypothesis,
            "intended_upside": self.intended_upside,
            "expected_risk": self.expected_risk,
            "intended_physical_change": self.intended_physical_change,
            "feature_to_preserve": self.feature_to_preserve,
            "principal_biochemical_risk": self.principal_biochemical_risk,
            "metal_liability": self.metal_liability,
            "requires_candidate_geometry": self.requires_candidate_geometry,
            "double_category": self.double_category,
            "physical_coupling": self.physical_coupling,
            "epistasis_uncertainty": self.epistasis_uncertainty,
            "known_experiment_conflict": self.known_experiment_conflict,
            "substitution_classes": list(self.substitution_classes),
            "revision_generation": self.revision_generation,
            "disposition": self.disposition.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CandidateRecord:
        return cls(
            candidate_id=str(data["candidate_id"]),
            sequence=str(data["sequence"]),
            mutations=tuple(Mutation(**item) for item in data["mutations"]),
            parents=tuple(data["parents"]),
            strategy=DesignStrategy(data["strategy"]),
            structural_region=str(data["structural_region"]),
            round_index=int(data["round_index"]),
            hypothesis=str(data["hypothesis"]),
            intended_upside=str(data["intended_upside"]),
            expected_risk=str(data["expected_risk"]),
            intended_physical_change=str(
                data.get("intended_physical_change", data["intended_upside"])
            ),
            feature_to_preserve=str(
                data.get("feature_to_preserve", data["hypothesis"])
            ),
            principal_biochemical_risk=str(
                data.get("principal_biochemical_risk", data["expected_risk"])
            ),
            metal_liability=str(data.get("metal_liability", "none_identified")),
            requires_candidate_geometry=bool(
                data.get("requires_candidate_geometry", False)
            ),
            double_category=(
                None if data.get("double_category") is None else str(data["double_category"])
            ),
            physical_coupling=(
                None
                if data.get("physical_coupling") is None
                else str(data["physical_coupling"])
            ),
            epistasis_uncertainty=(
                None
                if data.get("epistasis_uncertainty") is None
                else str(data["epistasis_uncertainty"])
            ),
            known_experiment_conflict=str(
                data.get("known_experiment_conflict", "none_identified")
            ),
            substitution_classes=tuple(data.get("substitution_classes", ())),
            revision_generation=int(data.get("revision_generation", 0)),
            disposition=CandidateDisposition(data.get("disposition", "pending")),
        )


@dataclass(frozen=True)
class EvidenceRecord:
    candidate_id: str
    axis: EvidenceAxis
    status: EvidenceStatus
    value: float | None
    uncertainty: float | None
    method: str
    provenance: Mapping[str, Any]
    payload: Mapping[str, Any]

    @classmethod
    def numeric(
        cls,
        candidate_id: str,
        axis: EvidenceAxis,
        *,
        value: float,
        uncertainty: float | None,
        method: str,
        provenance: Mapping[str, Any],
        payload: Mapping[str, Any] | None = None,
    ) -> EvidenceRecord:
        return cls(
            candidate_id=candidate_id,
            axis=EvidenceAxis(axis),
            status=EvidenceStatus.AVAILABLE,
            value=float(value),
            uncertainty=None if uncertainty is None else float(uncertainty),
            method=method,
            provenance=dict(provenance),
            payload={} if payload is None else dict(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "axis": self.axis.value,
            "status": self.status.value,
            "value": self.value,
            "uncertainty": self.uncertainty,
            "method": self.method,
            "provenance": dict(self.provenance),
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidenceRecord:
        return cls(
            candidate_id=str(data["candidate_id"]),
            axis=EvidenceAxis(data["axis"]),
            status=EvidenceStatus(data["status"]),
            value=None if data.get("value") is None else float(data["value"]),
            uncertainty=(
                None if data.get("uncertainty") is None else float(data["uncertainty"])
            ),
            method=str(data["method"]),
            provenance=dict(data.get("provenance", {})),
            payload=dict(data.get("payload", {})),
        )


@dataclass(frozen=True)
class HardViolation:
    candidate_id: str
    code: str
    detail: str
    hard: bool = True


@dataclass(frozen=True)
class FailureObservation:
    candidate_id: str | None
    category: str
    scope: str
    detail: str
    evidence_count: int
    confidence: float
    source: str

    def __post_init__(self) -> None:
        if self.evidence_count < 1:
            raise ValueError("evidence_count must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompletionAudit:
    unique_legal_evaluated: int
    candidate_budget: int
    strategies_used: frozenset[str]
    required_strategies: frozenset[str]
    structural_regions_explored: frozenset[str]
    minimum_structural_regions: int
    pending_candidates: int
    pending_checkpoints: int
    eligible_unrepaired_near_misses: int
    finalists: int
    near_misses_reported: int
    promotable_without_hard_relaxation: int

    @property
    def blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if self.unique_legal_evaluated < self.candidate_budget:
            blockers.append(
                "Candidate budget is not exhausted: "
                f"{self.unique_legal_evaluated}/{self.candidate_budget} legal variants evaluated."
            )
        missing = sorted(self.required_strategies - self.strategies_used)
        if missing:
            blockers.append("Required design strategies not exercised: " + ", ".join(missing))
        if len(self.structural_regions_explored) < self.minimum_structural_regions:
            blockers.append("Insufficient defensible structural-region coverage.")
        if self.pending_candidates:
            blockers.append(f"{self.pending_candidates} candidates remain pending.")
        if self.pending_checkpoints:
            blockers.append(f"{self.pending_checkpoints} checkpoints remain pending.")
        if self.eligible_unrepaired_near_misses:
            blockers.append(
                f"{self.eligible_unrepaired_near_misses} repair-eligible near-misses remain."
            )
        if self.finalists == 0:
            if self.near_misses_reported == 0:
                blockers.append("A zero-finalist result requires reported near-miss hypotheses.")
            if self.promotable_without_hard_relaxation:
                blockers.append(
                    f"{self.promotable_without_hard_relaxation} candidates remain promotable "
                    "without relaxing hard constraints."
                )
        return tuple(blockers)

    @property
    def complete(self) -> bool:
        return not self.blockers
