"""Genuine stability and mutant-complex backend for adaptive Atlas."""

from __future__ import annotations

from pathlib import Path

from atlas.adaptive.models import (
    CandidateRecord,
    EvidenceAxis,
    EvidenceRecord,
    EvidenceStatus,
    HardViolation,
)
from atlas.adaptive_pipeline import CandidateStructureResult
from atlas.dynamics.models import DynamicsConfig
from atlas.geometry.catalytic_metrics import GeometryRecord, measure_geometry
from atlas.stability.common import StabilityVariant
from atlas.stability.thermompnn_d_runner import (
    THERMOMPNN_D_REVISION,
    TargetedThermoMPNNDRunner,
)
from atlas.stability.thermompnn_runner import THERMOMPNN_REVISION, ThermoMPNNRunner
from atlas.structure.mutant_complex import build_mutant_complex


_GROUNDED_GEOMETRY = (
    "zn_scissile_oxygen_distance_a",
    "zn_h95_ne2_distance_a",
    "zn_h99_ne2_distance_a",
    "zn_e122_oxygen_distance_a",
    "e96_to_scissile_carbonyl_distance_a",
    "h172_to_scissile_oxygen_distance_a",
)


def _deposited_mutation_set(candidate: CandidateRecord) -> str:
    labels = []
    for mutation in candidate.mutations:
        labels.append(
            f"{mutation.wildtype}{mutation.position + 24}{mutation.mutant}"
        )
    return ":".join(labels)


def _variant(candidate: CandidateRecord) -> StabilityVariant:
    return StabilityVariant(
        candidate.candidate_id,
        candidate.mutation_set.replace("/", ":"),
        _deposited_mutation_set(candidate),
    )


def _invalid_evidence(
    candidate: CandidateRecord,
    axis: EvidenceAxis,
    *,
    method: str,
    detail: str,
    artifact: Path | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        candidate_id=candidate.candidate_id,
        axis=axis,
        status=EvidenceStatus.INVALID,
        value=None,
        uncertainty=None,
        method=method,
        provenance={
            "artifact_path": None if artifact is None else str(artifact),
            "evidence_tier": "HARD_CONSTRAINT",
        },
        payload={
            "detail": detail,
            "claim_boundary": "Invalid evidence; no favorable value was imputed.",
        },
    )


def _unavailable_activity(candidate: CandidateRecord, artifact: Path) -> EvidenceRecord:
    return EvidenceRecord(
        candidate_id=candidate.candidate_id,
        axis=EvidenceAxis.ACTIVITY_ORIENTED,
        status=EvidenceStatus.UNAVAILABLE,
        value=None,
        uncertainty=None,
        method="Atlas activity-model eligibility audit",
        provenance={
            "artifact_path": str(artifact),
            "review": "docs/activity_model_review.md",
            "evidence_tier": "NOT_EVALUATED",
        },
        payload={
            "decision": "not_available",
            "reason": (
                "No audited model has validated DP622/Aβ multichain, Zn-site, de novo "
                "mutation-activity applicability."
            ),
        },
    )


class OfficialAdaptiveBackend:
    """Compose only genuine, pinned scientific implementations."""

    def __init__(
        self,
        *,
        thermompnn_repo: Path,
        thermompnn_d_repo: Path,
        relaxation_config: DynamicsConfig,
        seed: int,
    ) -> None:
        self.single = ThermoMPNNRunner(thermompnn_repo)
        self.double = TargetedThermoMPNNDRunner(thermompnn_d_repo)
        self.relaxation_config = relaxation_config
        self.seed = seed
        self._reference_geometry: GeometryRecord | None = None
        self._reference_path: Path | None = None

    def score_stability(
        self,
        reference_pdb: Path,
        candidates: tuple[CandidateRecord, ...],
        output_dir: Path,
    ) -> tuple[EvidenceRecord, ...]:
        self._reference(reference_pdb)
        output_dir.mkdir(parents=True, exist_ok=True)
        singles = tuple(candidate for candidate in candidates if len(candidate.mutations) == 1)
        doubles = tuple(candidate for candidate in candidates if len(candidate.mutations) == 2)
        if len(singles) + len(doubles) != len(candidates):
            raise ValueError("ThermoMPNN production funnel supports only singles and doubles")
        frames = []
        if singles:
            stability_root = output_dir.parent
            model_dir = stability_root / "thermompnn_single_model"
            raw = model_dir / f"ThermoMPNN_inference_{reference_pdb.stem}.csv"
            if raw.is_file():
                frame = self.single.normalize_existing(
                    raw, [_variant(candidate) for candidate in singles], output_dir / "single"
                )
            else:
                frame = self.single.run(
                    reference_pdb,
                    [_variant(candidate) for candidate in singles],
                    model_dir,
                )
            frames.append(frame)
        if doubles:
            frames.append(
                self.double.run(
                    reference_pdb,
                    [_variant(candidate) for candidate in doubles],
                    output_dir / "double",
                )
            )
        if not frames:
            return ()
        import pandas as pd

        table = pd.concat(frames, ignore_index=True)
        table.to_csv(output_dir / "stability_evidence.csv", index=False)
        candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
        records = []
        for row in table.itertuples(index=False):
            candidate = candidates_by_id[str(row.variant_id)]
            is_double = len(candidate.mutations) == 2
            model_commit = THERMOMPNN_D_REVISION if is_double else THERMOMPNN_REVISION
            records.append(
                EvidenceRecord.numeric(
                    candidate.candidate_id,
                    EvidenceAxis.STABILITY,
                    value=float(row.predicted_ddg_or_score),
                    uncertainty=0.75 if is_double else 0.5,
                    method=str(row.model_used),
                    provenance={
                        "model_commit": model_commit,
                        "artifact_path": str(output_dir / "stability_evidence.csv"),
                        "evidence_tier": "COMPUTATIONAL_SUPPORT",
                        "uncertainty_policy": (
                            "Fixed epistemic caution band, not a calibrated confidence interval; "
                            "retrospective activity benchmark was not faithfully separated."
                        ),
                    },
                    payload={
                        "raw_predicted_ddg_or_score": float(row.predicted_ddg_or_score),
                        "interpretation": str(row.interpretation),
                        "warnings": str(row.warnings),
                        "direction": "lower predicted stability regression",
                        "activity_claim": False,
                    },
                )
            )
        return tuple(records)

    def _reference(self, pdb_path: Path) -> GeometryRecord:
        resolved = Path(pdb_path).resolve()
        if self._reference_geometry is None:
            self._reference_geometry = measure_geometry(
                resolved, resolved, "reference_active_like_inferred"
            )
            self._reference_path = resolved
        elif self._reference_path != resolved:
            raise RuntimeError(
                "Reference geometry is already registered from a different structure: "
                f"{self._reference_path} != {resolved}"
            )
        return self._reference_geometry

    def evaluate_structure(
        self,
        reference_pdb: Path,
        candidate: CandidateRecord,
        output_dir: Path,
        *,
        seed: int,
    ) -> CandidateStructureResult:
        result = build_mutant_complex(
            reference_pdb,
            candidate,
            output_dir,
            seed=seed,
            relax=True,
            dynamics_config=self.relaxation_config,
        )
        structure = result.relaxed_pdb or result.raw_pdb
        if result.hard_violations:
            return CandidateStructureResult(
                candidate_id=candidate.candidate_id,
                structure_path=structure,
                evidence=(
                    _invalid_evidence(
                        candidate,
                        EvidenceAxis.STRUCTURE_QUALITY,
                        method="mutant-complex hard-constraint validation",
                        detail="; ".join(v.detail for v in result.hard_violations),
                        artifact=result.provenance_json,
                    ),
                ),
                hard_violations=result.hard_violations,
                artifact_path=result.provenance_json,
            )
        geometry = measure_geometry(structure, reference_pdb, candidate.candidate_id)
        reference = self._reference(reference_pdb)
        raw = geometry.csv_row()
        raw["warnings"] = list(geometry.warnings)
        grounded_deltas = {
            name: abs(float(getattr(geometry, name)) - float(getattr(reference, name)))
            for name in _GROUNDED_GEOMETRY
            if getattr(geometry, name) is not None and getattr(reference, name) is not None
        }
        if not grounded_deltas or geometry.active_site_rmsd_a is None:
            evidence = (
                _invalid_evidence(
                    candidate,
                    EvidenceAxis.CATALYTIC_GEOMETRY,
                    method="catalytic_preorganization static geometry",
                    detail="Required experimentally grounded geometry could not be measured.",
                    artifact=result.provenance_json,
                ),
            )
        else:
            pose_drift = geometry.substrate_pose_drift_a or 0.0
            substrate_rmsd = geometry.substrate_rmsd_a or 0.0
            interface_support = 1.0 / (1.0 + pose_drift + substrate_rmsd)
            uncertainty = 0.35 if result.relaxed_pdb is not None else 0.6
            provenance = {
                "model_label": "active_like_inferred",
                "artifact_path": str(result.provenance_json),
                "structure_path": str(structure),
                "evidence_tier": "COMPUTATIONAL_SUPPORT",
                "relaxation_completed": result.relaxed_pdb is not None,
            }
            evidence = (
                EvidenceRecord.numeric(
                    candidate.candidate_id,
                    EvidenceAxis.STRUCTURE_QUALITY,
                    value=float(geometry.active_site_rmsd_a),
                    uncertainty=uncertainty,
                    method="restrained mutant-complex active-site RMSD",
                    provenance=provenance,
                    payload={"geometry": raw, "direction": "lower RMSD"},
                ),
                EvidenceRecord.numeric(
                    candidate.candidate_id,
                    EvidenceAxis.CATALYTIC_GEOMETRY,
                    value=max(grounded_deltas.values()),
                    uncertainty=uncertainty,
                    method="catalytic_preorganization maximum grounded-distance deviation",
                    provenance=provenance,
                    payload={
                        "distance_deviations_a": grounded_deltas,
                        "direction": "lower deviation",
                        "not_catalytic_rate": True,
                    },
                ),
                EvidenceRecord.numeric(
                    candidate.candidate_id,
                    EvidenceAxis.SUBSTRATE_INTERFACE,
                    value=interface_support,
                    uncertainty=uncertainty,
                    method="restrained Aβ pose-preservation support",
                    provenance=provenance,
                    payload={
                        "substrate_pose_drift_a": geometry.substrate_pose_drift_a,
                        "substrate_rmsd_a": geometry.substrate_rmsd_a,
                        "direction": "higher pose-preservation support",
                    },
                ),
            )
        return CandidateStructureResult(
            candidate_id=candidate.candidate_id,
            structure_path=structure,
            evidence=evidence,
            hard_violations=result.hard_violations,
            artifact_path=result.provenance_json,
        )
