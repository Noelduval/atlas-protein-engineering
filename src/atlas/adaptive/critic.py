"""Evidence-aware scientific critic with explicit reject/revise/promote routes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from atlas.adaptive.models import EvidenceAxis
from atlas.adaptive.screening import Evaluation


class CriticRoute(str, Enum):
    REJECT = "REJECT"
    REVISE = "REVISE"
    PROMOTE = "PROMOTE"


@dataclass(frozen=True)
class Critique:
    candidate_id: str
    route: CriticRoute
    supporting_signals: tuple[str, ...]
    weakness_axis: EvidenceAxis | None
    diagnosed_weakness: str
    repairable: bool
    feature_to_preserve: str
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "route": self.route.value,
            "supporting_signals": list(self.supporting_signals),
            "weakness_axis": None if self.weakness_axis is None else self.weakness_axis.value,
            "diagnosed_weakness": self.diagnosed_weakness,
            "repairable": self.repairable,
            "feature_to_preserve": self.feature_to_preserve,
            "rationale": self.rationale,
        }


class CriticPolicy:
    """Route candidates while keeping hard failures and soft tradeoffs distinct."""

    _SUPPORT = {
        EvidenceAxis.STABILITY: ("maximum", 1.0),
        EvidenceAxis.STRUCTURE_QUALITY: ("minimum", 0.5),
        EvidenceAxis.CATALYTIC_GEOMETRY: ("maximum", 0.75),
        EvidenceAxis.SUBSTRATE_INTERFACE: ("minimum", 0.5),
        EvidenceAxis.DYNAMICS: ("minimum", 0.5),
        EvidenceAxis.LIABILITY: ("maximum", 0.4),
    }
    _WEAKNESS = {
        EvidenceAxis.STABILITY: ("minimum", 1.0, "Predicted stability is regressive."),
        EvidenceAxis.STRUCTURE_QUALITY: (
            "maximum",
            0.4,
            "Mutant-complex structural quality is weak.",
        ),
        EvidenceAxis.CATALYTIC_GEOMETRY: (
            "minimum",
            0.75,
            "Catalytic-preorganization deviation is elevated.",
        ),
        EvidenceAxis.SUBSTRATE_INTERFACE: (
            "maximum",
            0.4,
            "Substrate-interface support is weak.",
        ),
        EvidenceAxis.DYNAMICS: (
            "maximum",
            0.4,
            "Replicated dynamics support is weak or inconsistent.",
        ),
        EvidenceAxis.LIABILITY: ("minimum", 0.6, "Sequence/structure liability is elevated."),
    }

    @staticmethod
    def _meets(value: float, comparator: str, threshold: float) -> bool:
        return value <= threshold if comparator == "maximum" else value >= threshold

    def critique(self, evaluation: Evaluation) -> Critique:
        candidate_id = evaluation.candidate.candidate_id
        if evaluation.hard_violations:
            details = "; ".join(violation.detail for violation in evaluation.hard_violations)
            return Critique(
                candidate_id=candidate_id,
                route=CriticRoute.REJECT,
                supporting_signals=(),
                weakness_axis=None,
                diagnosed_weakness=details,
                repairable=False,
                feature_to_preserve="",
                rationale="Hard chemistry/structural violations cannot be relaxed or repaired in-place.",
            )

        by_axis = evaluation.latest_by_axis()
        supporting: list[str] = []
        for axis, (comparator, threshold) in self._SUPPORT.items():
            record = by_axis.get(axis)
            if record is not None and self._meets(float(record.value), comparator, threshold):
                supporting.append(f"{axis.value}={record.value:g} supports the hypothesis")

        weaknesses: list[tuple[EvidenceAxis, str]] = []
        for axis, (comparator, threshold, description) in self._WEAKNESS.items():
            record = by_axis.get(axis)
            if record is not None and self._meets(float(record.value), comparator, threshold):
                weaknesses.append((axis, description))

        if supporting and weaknesses:
            weakness_axis, weakness = weaknesses[0]
            return Critique(
                candidate_id=candidate_id,
                route=CriticRoute.REVISE,
                supporting_signals=tuple(supporting),
                weakness_axis=weakness_axis,
                diagnosed_weakness=weakness,
                repairable=True,
                feature_to_preserve=supporting[0],
                rationale=(
                    "Candidate has no hard violation, at least one supporting signal, and a "
                    "specific soft weakness suitable for bounded repair."
                ),
            )
        if supporting:
            return Critique(
                candidate_id=candidate_id,
                route=CriticRoute.PROMOTE,
                supporting_signals=tuple(supporting),
                weakness_axis=None,
                diagnosed_weakness="No policy-level repairable weakness identified.",
                repairable=False,
                feature_to_preserve=supporting[0],
                rationale="Independent evidence supports promotion to the next funnel stage.",
            )
        return Critique(
            candidate_id=candidate_id,
            route=CriticRoute.REJECT,
            supporting_signals=(),
            weakness_axis=None,
            diagnosed_weakness="No meaningful supporting signal is currently available.",
            repairable=False,
            feature_to_preserve="",
            rationale="Insufficient support for repair or promotion; no hard constraint was relaxed.",
        )

