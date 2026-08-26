"""Reproducible eligibility audit for activity-oriented protein models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

from atlas.adaptive.models import (
    EvidenceAxis,
    EvidenceRecord,
    EvidenceStatus,
)


@dataclass(frozen=True)
class ActivityModelReview:
    name: str
    version: str
    prediction_target: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    code_available: bool
    checkpoint_available: bool
    mutation_support: str
    supports_peptide_substrate: bool
    complex_context_modeled: bool
    metal_treatment_documented: bool
    training_domain: str
    de_novo_metalloenzyme_validation: bool
    reproducible_in_pinned_colab: bool
    license: str
    primary_sources: tuple[str, ...]
    eligible_for_atlas: bool = False
    exclusion_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ActivityModelDecision:
    status: str
    selected_model: str | None
    benchmark_used_for_selection: bool
    reviews: tuple[ActivityModelReview, ...]
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "selected_model": self.selected_model,
            "benchmark_used_for_selection": self.benchmark_used_for_selection,
            "reviews": [review.to_dict() for review in self.reviews],
            "rationale": self.rationale,
        }


def _assess(review: ActivityModelReview) -> ActivityModelReview:
    reasons: list[str] = []
    if review.prediction_target not in {"relative_enzyme_activity", "kinetic_parameter"}:
        reasons.append("Prediction target is not enzyme activity or a kinetic parameter.")
    if not review.code_available:
        reasons.append("Runnable inference code is not available from the official source.")
    if not review.checkpoint_available:
        reasons.append("A versioned inference checkpoint is not available.")
    if review.mutation_support not in {"single_and_multiple", "arbitrary_sequence"}:
        reasons.append("Required single/double mutation support is not demonstrated.")
    if not review.supports_peptide_substrate:
        reasons.append("Aβ peptide-substrate input and validation are not documented.")
    if not review.complex_context_modeled:
        reasons.append("The DP622/Aβ complex context would not be modeled.")
    if not review.metal_treatment_documented:
        reasons.append("Zn-site treatment is not documented or validated.")
    if not review.de_novo_metalloenzyme_validation:
        reasons.append("De novo Zn-protease applicability is not validated.")
    if not review.reproducible_in_pinned_colab:
        reasons.append("The released runtime is incompatible with the pinned Atlas T4 environment.")
    return replace(
        review,
        eligible_for_atlas=not reasons,
        exclusion_reasons=tuple(reasons),
    )


def _reviewed_models() -> tuple[ActivityModelReview, ...]:
    return (
        ActivityModelReview(
            name="ProMEP",
            version="e44e8f4fd8d54255c71c80805598a09527dd0eab",
            prediction_target="general_protein_fitness_log_likelihood",
            inputs=("wild-type monomer sequence", "wild-type monomer structure"),
            outputs=("variant log-likelihood ratio",),
            code_available=True,
            checkpoint_available=False,
            mutation_support="single_and_multiple",
            supports_peptide_substrate=False,
            complex_context_modeled=False,
            metal_treatment_documented=False,
            training_domain="General protein sequence/structure pretraining and DMS fitness tasks.",
            de_novo_metalloenzyme_validation=False,
            reproducible_in_pinned_colab=False,
            license="Apache-2.0",
            primary_sources=(
                "https://www.nature.com/articles/s41422-024-00989-2",
                "https://github.com/wenjiegroup/ProMEP",
            ),
        ),
        ActivityModelReview(
            name="EnzyACT",
            version="a8ec8c30dc2239bfe1f18a03fbea97cf571a8987",
            prediction_target="relative_enzyme_activity",
            inputs=("wild/mutant sequence embeddings", "monomer residue graph"),
            outputs=("increased/decreased activity classification",),
            code_available=False,
            checkpoint_available=False,
            mutation_support="single_and_multiple",
            supports_peptide_substrate=False,
            complex_context_modeled=False,
            metal_treatment_documented=False,
            training_domain="BRENDA and D3DistalMutation activity-change labels; imbalanced positives.",
            de_novo_metalloenzyme_validation=False,
            reproducible_in_pinned_colab=False,
            license="Apache-2.0 repository license",
            primary_sources=(
                "https://doi.org/10.1021/acs.jcim.4c00920",
                "https://github.com/GenScript-IBDPE/EnzyACT",
            ),
        ),
        ActivityModelReview(
            name="CatPred",
            version="1.0.1/fabdf38c32a005acf3af916456a4727128bc10ef",
            prediction_target="kinetic_parameter",
            inputs=("enzyme sequence", "substrate SMILES", "optional monomer structure"),
            outputs=("kcat", "Km", "Ki", "prediction uncertainty"),
            code_available=True,
            checkpoint_available=True,
            mutation_support="arbitrary_sequence",
            supports_peptide_substrate=False,
            complex_context_modeled=False,
            metal_treatment_documented=False,
            training_domain="Curated in-vitro enzyme/small-molecule kinetic parameter records.",
            de_novo_metalloenzyme_validation=False,
            reproducible_in_pinned_colab=False,
            license="MIT",
            primary_sources=(
                "https://www.nature.com/articles/s41467-025-57215-9",
                "https://github.com/maranasgroup/CatPred/releases/tag/1.0.1",
            ),
        ),
        ActivityModelReview(
            name="UniKP",
            version="3ad5576aaa2c8c0dd0e0b6c283c1d365ab23c6ea",
            prediction_target="kinetic_parameter",
            inputs=("enzyme sequence", "substrate SMILES"),
            outputs=("kcat", "Km", "kcat/Km"),
            code_available=True,
            checkpoint_available=True,
            mutation_support="arbitrary_sequence",
            supports_peptide_substrate=False,
            complex_context_modeled=False,
            metal_treatment_documented=False,
            training_domain="BRENDA/SABIO-RK-like enzyme/small-molecule kinetic records.",
            de_novo_metalloenzyme_validation=False,
            reproducible_in_pinned_colab=False,
            license="GPL-3.0",
            primary_sources=(
                "https://www.nature.com/articles/s41467-023-44113-1",
                "https://github.com/Luo-SynBioLab/UniKP",
            ),
        ),
        ActivityModelReview(
            name="PLACER",
            version="7dc5563300e7143582d85636bdd413d4c7688e9a",
            prediction_target="conformational_ensemble",
            inputs=("protein structure", "small-molecule/metal chemical graph"),
            outputs=("local atomistic conformational ensemble", "structural uncertainty"),
            code_available=True,
            checkpoint_available=True,
            mutation_support="structure_rebuilding_not_activity_mutation",
            supports_peptide_substrate=False,
            complex_context_modeled=True,
            metal_treatment_documented=True,
            training_domain="PDB/CSD local protein-small-molecule conformational reconstruction.",
            de_novo_metalloenzyme_validation=False,
            reproducible_in_pinned_colab=False,
            license="Repository license; activity integration not applicable",
            primary_sources=(
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC12625923/",
                "https://github.com/baker-laboratory/PLACER",
            ),
        ),
    )


def audit_activity_models() -> ActivityModelDecision:
    reviews = tuple(_assess(review) for review in _reviewed_models())
    eligible = [review for review in reviews if review.eligible_for_atlas]
    return ActivityModelDecision(
        status="available" if eligible else "not_available",
        selected_model=eligible[0].name if len(eligible) == 1 else None,
        benchmark_used_for_selection=False,
        reviews=reviews,
        rationale=(
            "No reviewed model simultaneously provides the correct activity/kinetic target, "
            "reproducible checkpoints, Aβ peptide-complex context, documented Zn treatment, "
            "and validation for a de novo metalloenzyme. Atlas therefore records unavailable "
            "activity-oriented evidence rather than an out-of-domain favorable score."
        ),
    )


def activity_evidence_unavailable(
    candidate_id: str, decision: ActivityModelDecision
) -> EvidenceRecord:
    return EvidenceRecord(
        candidate_id=candidate_id,
        axis=EvidenceAxis.ACTIVITY_ORIENTED,
        status=EvidenceStatus.UNAVAILABLE,
        value=None,
        uncertainty=None,
        method="primary-source activity-model eligibility audit",
        provenance={
            "reviewed_models": [review.name for review in decision.reviews],
            "benchmark_used_for_selection": decision.benchmark_used_for_selection,
        },
        payload={"decision": decision.status, "rationale": decision.rationale},
    )


def write_activity_model_audit(
    decision: ActivityModelDecision,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    json_destination, markdown_destination = Path(json_path), Path(markdown_path)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    markdown_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.write_text(
        json.dumps(decision.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Activity-oriented model review",
        "",
        f"**Decision:** `{decision.status}`",
        "",
        "## Why Atlas excludes all reviewed models",
        "",
        decision.rationale,
        "",
        "The historical known-mutant benchmark was not used to select or tune a model.",
        "",
        "| Model | Target | Checkpoint | Atlas eligible | Primary exclusion |",
        "|---|---|---:|---:|---|",
    ]
    for review in decision.reviews:
        primary = review.exclusion_reasons[0] if review.exclusion_reasons else "None"
        lines.append(
            f"| {review.name} | `{review.prediction_target}` | "
            f"{'yes' if review.checkpoint_available else 'no'} | "
            f"{'yes' if review.eligible_for_atlas else 'no'} | {primary} |"
        )
    lines.extend(["", "## Sources", ""])
    for review in decision.reviews:
        lines.append(
            f"- **{review.name}:** "
            + ", ".join(f"<{source}>" for source in review.primary_sources)
        )
    lines.extend(
        [
            "",
            "## Scientific boundary",
            "",
            "Unavailable activity evidence is not treated as favorable evidence. Atlas may "
            "complete without an activity model and must not claim predicted catalytic turnover.",
        ]
    )
    markdown_destination.write_text("\n".join(lines) + "\n")

