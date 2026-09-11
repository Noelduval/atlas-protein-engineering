"""Reports, exports, and figures generated strictly from adaptive run artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

from atlas.adaptive.ledger import ScientificLedger
from atlas.adaptive.models import (
    CandidateRecord,
    EvidenceAxis,
    EvidenceRecord,
    EvidenceStatus,
)
from atlas.design.design_space import ResidueDesignRecord


@dataclass(frozen=True)
class AdaptiveOutputBundle:
    final_report: Path
    reproducibility_manifest: Path
    candidate_dossiers: tuple[Path, ...]
    fasta: Path
    structure_exports: tuple[Path, ...]
    figures: tuple[Path, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)
    return path


def _candidate_rows(candidates: Iterable[CandidateRecord]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": candidate.candidate_id,
            "mutation_set": candidate.mutation_set,
            "sequence": candidate.sequence,
            "parents": ";".join(candidate.parents),
            "strategy": candidate.strategy.value,
            "structural_region": candidate.structural_region,
            "round_index": candidate.round_index,
            "revision_generation": candidate.revision_generation,
            "hypothesis": candidate.hypothesis,
            "intended_upside": candidate.intended_upside,
            "expected_risk": candidate.expected_risk,
        }
        for candidate in candidates
    ]


def _group_evidence(records: Iterable[EvidenceRecord]) -> dict[str, tuple[EvidenceRecord, ...]]:
    grouped: dict[str, list[EvidenceRecord]] = defaultdict(list)
    for record in records:
        grouped[record.candidate_id].append(record)
    return {key: tuple(value) for key, value in grouped.items()}


def _lineage(candidate: CandidateRecord, ledger: ScientificLedger) -> list[CandidateRecord]:
    ordered: list[CandidateRecord] = []
    seen: set[str] = set()

    def visit(item: CandidateRecord) -> None:
        for parent_id in item.parents:
            parent = ledger.get_candidate(parent_id)
            if parent is not None:
                visit(parent)
        if item.candidate_id not in seen:
            seen.add(item.candidate_id)
            ordered.append(item)

    visit(candidate)
    return ordered


def _evidence_markdown(records: tuple[EvidenceRecord, ...]) -> str:
    lines = [
        "| Axis | Status | Raw value | Uncertainty | Method |",
        "|---|---|---:|---:|---|",
    ]
    for record in records:
        value = "—" if record.value is None else f"{record.value:.6g}"
        uncertainty = "—" if record.uncertainty is None else f"{record.uncertainty:.6g}"
        lines.append(
            f"| {record.axis.value} | {record.status.value} | {value} | "
            f"{uncertainty} | {record.method} |"
        )
    return "\n".join(lines)


def _wrap_fasta(sequence: str, width: int = 80) -> str:
    return "\n".join(sequence[index : index + width] for index in range(0, len(sequence), width))


def _render_structure(
    pdb_path: Path,
    candidate: CandidateRecord,
    output_path: Path,
) -> None:
    structure = PDBParser(QUIET=True).get_structure(candidate.candidate_id, pdb_path)
    model = structure[0]
    protein = {
        int(residue.id[1]): np.asarray(residue["CA"].coord, dtype=float)
        for residue in model["A"]
        if "CA" in residue
    }
    substrate = np.asarray(
        [residue["CA"].coord for residue in model["B"] if "CA" in residue],
        dtype=float,
    )
    zinc = np.asarray(
        [atom.coord for atom in model["C"].get_atoms() if atom.element == "ZN"],
        dtype=float,
    )
    protein_xyz = np.asarray([protein[position] for position in sorted(protein)])
    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(*protein_xyz.T, color="#64748b", linewidth=1.2, label="DP622 CA trace")
    if len(substrate):
        axis.plot(*substrate.T, color="#f59e0b", linewidth=3, label="resolved Aβ 34–41")
    if len(zinc):
        axis.scatter(*zinc.T, color="#7c3aed", s=90, label="Zn")
    mutation_xyz = np.asarray(
        [protein[mutation.position] for mutation in candidate.mutations if mutation.position in protein]
    )
    if len(mutation_xyz):
        axis.scatter(*mutation_xyz.T, color="#dc2626", s=75, label=candidate.mutation_set)
    axis.set_title(f"{candidate.candidate_id}: {candidate.mutation_set}\nEXPERIMENTALLY UNTESTED")
    axis.set_xlabel("x (Å)")
    axis.set_ylabel("y (Å)")
    axis.set_zlabel("z (Å)")
    axis.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_structural_map(
    reference_pdb: Path,
    design_space: tuple[ResidueDesignRecord, ...],
    finalists: tuple[CandidateRecord, ...],
    output_path: Path,
) -> None:
    structure = PDBParser(QUIET=True).get_structure("reference", reference_pdb)
    model = structure[0]
    coords = {
        int(residue.id[1]): np.asarray(residue["CA"].coord, dtype=float)
        for residue in model["A"]
        if "CA" in residue
    }
    colors = {
        "HARD_PROTECTED": "#dc2626",
        "CONTEXT_SENSITIVE": "#f59e0b",
        "DESIGNABLE": "#2563eb",
        "OUT_OF_SCOPE": "#94a3b8",
    }
    figure, axis = plt.subplots(figsize=(9, 7))
    for residue_class, color in colors.items():
        positions = [
            record.position
            for record in design_space
            if record.residue_class.value == residue_class and record.position in coords
        ]
        xyz = np.asarray([coords[position] for position in positions])
        if len(xyz):
            axis.scatter(xyz[:, 0], xyz[:, 1], s=16, alpha=0.75, c=color, label=residue_class)
    substrate = np.asarray(
        [residue["CA"].coord for residue in model["B"] if "CA" in residue], dtype=float
    )
    if len(substrate):
        axis.plot(substrate[:, 0], substrate[:, 1], color="#111827", linewidth=3, label="Aβ 34–41")
    zinc = np.asarray([atom.coord for atom in model["C"].get_atoms() if atom.element == "ZN"])
    if len(zinc):
        axis.scatter(zinc[:, 0], zinc[:, 1], marker="*", s=180, c="#7c3aed", label="Zn")
    finalist_positions = sorted(
        {mutation.position for candidate in finalists for mutation in candidate.mutations}
    )
    for position in finalist_positions:
        if position in coords:
            x, y, _ = coords[position]
            axis.scatter([x], [y], s=100, facecolors="none", edgecolors="#16a34a", linewidths=2)
            axis.annotate(str(position), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.set_title("23WN-derived active_like_inferred design map (x–y projection)")
    axis.set_xlabel("x coordinate (Å)")
    axis.set_ylabel("y coordinate (Å)")
    axis.legend(fontsize=8, ncol=2)
    axis.set_aspect("equal", adjustable="datalim")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_funnel(funnel: dict[str, int], output_path: Path) -> None:
    labels = [
        "Generated / evaluated",
        "Broad survivors",
        "Structural analyses",
        "Adversarial review",
        "Finalists",
    ]
    keys = [
        "generated_evaluated",
        "broad_survivors",
        "structural_analyses",
        "adversarial_review",
        "finalists",
    ]
    values = [int(funnel.get(key, 0)) for key in keys]
    figure, axis = plt.subplots(figsize=(10, 5))
    y = np.arange(len(labels))
    axis.barh(y, np.maximum(values, 0), color=plt.cm.viridis(np.linspace(0.15, 0.85, len(labels))))
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xscale("symlog", linthresh=1)
    for index, value in enumerate(values):
        axis.text(max(value, 0) + 0.05, index, f"{value:,}", va="center")
    axis.set_title("Atlas adaptive prospective-design funnel — actual counts")
    axis.set_xlabel("Candidates (symlog scale)")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_search_landscape(candidates: tuple[CandidateRecord, ...], output_path: Path) -> None:
    round_strategy = Counter((candidate.round_index, candidate.strategy.value) for candidate in candidates)
    strategies = sorted({candidate.strategy.value for candidate in candidates})
    rounds = sorted({candidate.round_index for candidate in candidates})
    figure, (left, right) = plt.subplots(1, 2, figsize=(14, 5))
    bottom = np.zeros(len(rounds))
    for strategy in strategies:
        values = np.asarray([round_strategy[(round_index, strategy)] for round_index in rounds])
        left.bar([str(value) for value in rounds], values, bottom=bottom, label=strategy)
        bottom += values
    left.set_title("Search allocation by round and strategy")
    left.set_xlabel("Adaptive round")
    left.set_ylabel("Unique sequences")
    if strategies:
        left.legend(fontsize=6)
    else:
        left.text(0.5, 0.5, "No candidate records available", ha="center", va="center")
    region_counts = Counter(candidate.structural_region for candidate in candidates)
    right.barh(list(region_counts), list(region_counts.values()), color="#2563eb")
    right.set_title("Structural-region coverage")
    right.set_xlabel("Unique sequences")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_failure_memory(ledger: ScientificLedger, output_path: Path) -> None:
    counts = Counter(failure.category for failure in ledger.failures())
    figure, axis = plt.subplots(figsize=(9, 5))
    if counts:
        axis.barh(list(counts), list(counts.values()), color="#b91c1c")
        axis.set_xlabel("Persisted observations")
    else:
        axis.text(0.5, 0.5, "No failure observations persisted", ha="center", va="center")
        axis.set_axis_off()
    axis.set_title("Failure memory by evidence-scoped category")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_finalist_axes(
    finalists: tuple[CandidateRecord, ...],
    evidence_by_candidate: dict[str, tuple[EvidenceRecord, ...]],
    output_path: Path,
) -> None:
    axes = (
        EvidenceAxis.STABILITY_MODEL_AWARE,
        EvidenceAxis.STRUCTURE_QUALITY,
        EvidenceAxis.CATALYTIC_GEOMETRY,
        EvidenceAxis.SUBSTRATE_INTERFACE,
        EvidenceAxis.LIABILITY,
    )
    figure, plots = plt.subplots(2, 3, figsize=(15, 8))
    for axis_name, plot in zip(axes, plots.flat):
        labels, values, errors = [], [], []
        for candidate in finalists:
            matching = [record for record in evidence_by_candidate[candidate.candidate_id] if record.axis is axis_name and record.value is not None]
            if matching:
                record = matching[-1]
                labels.append(candidate.candidate_id.replace("ATLAS-", ""))
                values.append(float(record.value))
                errors.append(0.0 if record.uncertainty is None else float(record.uncertainty))
        if labels:
            plot.errorbar(range(len(labels)), values, yerr=errors, fmt="o", capsize=3)
            plot.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
            plot.set_ylabel("Raw axis value")
        else:
            plot.text(0.5, 0.5, "Unavailable", ha="center", va="center")
            plot.set_xticks([])
        plot.set_title(axis_name.value)
    for plot in plots.flat[len(axes):]:
        plot.set_axis_off()
    figure.suptitle("Finalist independent evidence axes (not a universal score)")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _dossier(
    candidate: CandidateRecord,
    ledger: ScientificLedger,
    evidence: tuple[EvidenceRecord, ...],
    design_by_position: dict[int, ResidueDesignRecord],
    review: dict[str, Any],
    structure_record: dict[str, Any] | None,
) -> str:
    context_lines = []
    liability_lines: list[str] = []
    hydrophobic = set("AILMVFYW")
    charged = set("DEKRH")
    for mutation in candidate.mutations:
        record = design_by_position[mutation.position]
        context_lines.append(
            f"- `{mutation.label}`: {record.structural_region}; "
            f"{record.min_substrate_distance_a:.3f} Å from resolved Aβ; "
            f"{record.min_zinc_distance_a:.3f} Å from Zn; {record.burial_class}."
        )
        if mutation.mutant == "C":
            liability_lines.append(f"{mutation.label}: cysteine introduction")
        if mutation.mutant == "P":
            liability_lines.append(f"{mutation.label}: proline introduction")
        if mutation.mutant == "G" and record.secondary_structure in {"helix", "strand"}:
            liability_lines.append(
                f"{mutation.label}: glycine at a secondary-structure-sensitive site"
            )
        if record.relative_sasa >= 0.5 and mutation.mutant in hydrophobic:
            liability_lines.append(f"{mutation.label}: exposed hydrophobic substitution")
        if (
            record.burial_class == "buried"
            and mutation.mutant in charged
            and mutation.wildtype not in charged
        ):
            liability_lines.append(f"{mutation.label}: buried charge introduction")
        if (mutation.wildtype in charged) != (mutation.mutant in charged):
            liability_lines.append(f"{mutation.label}: substantial charge-class change")
    latest = {
        record.axis: record
        for record in evidence
        if record.status is EvidenceStatus.AVAILABLE and record.value is not None
    }
    stability_policy = latest.get(EvidenceAxis.STABILITY_MODEL_AWARE)
    if stability_policy is not None and float(stability_policy.value) >= 0.75:
        liability_lines.append("within-model predicted stability regression")
    if candidate.metal_liability != "none_identified":
        liability_lines.append(candidate.metal_liability)
    structural_failures = (
        [] if structure_record is None else structure_record.get("hard_violations", [])
    )
    if structural_failures:
        liability_lines.extend(
            f"structural hard warning: {item.get('code', 'unknown')}"
            for item in structural_failures
        )
    liability_text = (
        "\n".join(f"- {item}" for item in sorted(set(liability_lines)))
        if liability_lines
        else "- No listed sequence/structure liability heuristic was triggered."
    )
    lineage = _lineage(candidate, ledger)
    lineage_text = " → ".join(
        f"{item.candidate_id} ({item.mutation_set})" for item in lineage
    )
    repair_history: list[str] = []
    lineage_ids = {item.candidate_id for item in lineage}
    for parent in lineage:
        for record in ledger.repair_trajectory(parent.candidate_id):
            if record.get("child_id") not in lineage_ids:
                continue
            repair_history.append(
                f"- `{record.get('parent_id')}` → `{record.get('child_id')}`: "
                f"{record.get('diagnosed_weakness')} Change: "
                f"{record.get('change_made')}. Preserve: "
                f"{record.get('feature_to_preserve')}. Re-evaluation disposition: "
                f"`{record.get('disposition')}`. Independent-axis deltas: "
                f"`{json.dumps(record.get('evidence_delta', {}), sort_keys=True)}`."
            )
    repair_history_text = (
        "\n".join(repair_history)
        if repair_history
        else "No repair child occurs in this candidate's lineage."
    )
    activity = [record for record in evidence if record.axis is EvidenceAxis.ACTIVITY_ORIENTED]
    activity_text = (
        "No activity-oriented model output was admitted; the audited models were not "
        "validated for the DP622/Aβ/Zn prospective-design use case."
        if not activity
        else "; ".join(f"{item.status.value}: {item.method}" for item in activity)
    )
    structure_text = (
        "No exported mutant structure."
        if structure_record is None
        else f"Mutant complex: `{structure_record.get('structure_path')}`; provenance: `{structure_record.get('artifact_path')}`."
    )
    return f"""# {candidate.candidate_id} — {candidate.mutation_set}

**Status: EXPERIMENTALLY UNTESTED**

## Identity

- Candidate ID: `{candidate.candidate_id}`
- Exact mutations: `{candidate.mutation_set}`
- Reference: 23WN-derived `active_like_inferred` DP622/Aβ/Zn model
- Novelty: not identified in the examined VITA/DP622 public materials and not previously evaluated by Atlas before this run; no claim of global historical novelty.
- Complete sequence ({len(candidate.sequence)} aa):

```text
{_wrap_fasta(candidate.sequence)}
```

## Design hypothesis

{candidate.hypothesis}

Intended upside: {candidate.intended_upside}

Expected risk: {candidate.expected_risk}

Intended physical change: {candidate.intended_physical_change}

Feature intended to remain preserved: {candidate.feature_to_preserve}

Principal biochemical risk: {candidate.principal_biochemical_risk}

Metal-liability status: `{candidate.metal_liability}`; candidate-specific geometry required: `{candidate.requires_candidate_geometry}`.

Double category: `{candidate.double_category or 'not_applicable'}`; physical coupling: {candidate.physical_coupling or 'not applicable'}; epistasis uncertainty: `{candidate.epistasis_uncertainty or 'not_applicable'}`.

Known-experiment conflict: `{candidate.known_experiment_conflict}`.

## Structural context

{chr(10).join(context_lines)}

{structure_text}

## Liability/developability heuristics

{liability_text}

These are policy heuristics, not experimentally validated developability predictions.

## Independent computational evidence

{_evidence_markdown(evidence)}

ThermoMPNN/ThermoMPNN-D raw values are stability evidence only and are not cross-model commensurate. Model-aware empirical ranks are screening policy values, not biological calibration. `catalytic_preorganization` values describe preservation/deviation of an experimentally grounded arrangement, not catalytic-rate prediction.

## Activity-oriented evidence

{activity_text}

## Design and revision history

{lineage_text}

{repair_history_text}

## Scientific critic

### Strongest argument for synthesis

{review.get('strongest_case_for_synthesis', 'The candidate is a distinct, falsifiable computational hypothesis.')}

### Strongest argument against synthesis

{review.get('strongest_case_against_synthesis', 'No computational evidence establishes improved catalytic activity.')}

Unresolved uncertainty: {review.get('unresolved_uncertainty', 'Model extrapolation and limited sampling remain.')}

## Falsifiable experimental hypothesis

{candidate.hypothesis} This hypothesis is falsified by failure to preserve expression/folding or by Aβ-cleavage behavior that is not improved or is worse than the matched active-like reference under the same assay.

Specific falsification result: {review.get('falsification_result', 'No Aβ cleavage relative to reference.')}

## Recommended wet-lab assays

1. Expression yield and solubility versus the matched reference.
2. Folding/thermal-stability measurement (for example nanoDSF or DSF), interpreted separately from activity.
3. Aβ cleavage-site product analysis by LC–MS or an equivalently specific assay.
4. Initial-rate kinetics across substrate concentrations to estimate `kcat`, `Km`, and `kcat/Km` only from experiments.
5. Cleavage selectivity/specificity profiling and Zn-dependence controls.

## Claim boundary

Atlas prioritizes this sequence as an experiment. It does not claim experimentally improved activity, therapeutic efficacy, plaque dissolution, or identity with an unavailable exact DP622-S2 assay construct.
"""


def _git_sha(run_dir: Path) -> str:
    context = run_dir / "run_context.json"
    if context.is_file():
        return str(json.loads(context.read_text()).get("atlas_commit", "unknown"))
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    )
    return completed.stdout.strip() or "unknown"


def build_adaptive_outputs(
    *,
    run_dir: Path,
    ledger: ScientificLedger,
    design_space: tuple[ResidueDesignRecord, ...],
    finalist_ids: tuple[str, ...],
    near_misses: list[dict[str, Any]],
    funnel_counts: dict[str, int],
    structure_records: list[dict[str, Any]],
    adversarial_reviews: list[dict[str, Any]],
    reference_pdb: Path,
    seed: int,
) -> AdaptiveOutputBundle:
    """Build all user-facing outputs from ledger-owned values and artifacts."""
    run_dir = Path(run_dir)
    reports = run_dir / "reports"
    figures_dir = run_dir / "figures"
    exports = run_dir / "exports"
    candidates = ledger.candidates()
    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    finalists = tuple(candidates_by_id[candidate_id] for candidate_id in finalist_ids)
    evidence = ledger.all_evidence()
    evidence_by_candidate = _group_evidence(evidence)
    design_by_position = {record.position: record for record in design_space}
    structure_by_id = {
        record["candidate_id"]: record
        for record in structure_records
        if record.get("candidate_id")
    }
    review_by_id = {record["candidate_id"]: record for record in adversarial_reviews}

    candidate_table = run_dir / "candidates.csv"
    pd.DataFrame(_candidate_rows(candidates)).to_csv(candidate_table, index=False)
    evidence_table = run_dir / "evidence_ledger.csv"
    pd.DataFrame([record.to_dict() for record in evidence]).to_csv(evidence_table, index=False)

    fasta_path = exports / "finalists.fasta"
    fasta_content = "".join(
        f">{candidate.candidate_id}|{candidate.mutation_set}|EXPERIMENTALLY_UNTESTED\n"
        f"{_wrap_fasta(candidate.sequence)}\n"
        for candidate in finalists
    )
    _write_text(fasta_path, fasta_content)
    individual_fastas = []
    for candidate in finalists:
        path = exports / "fastas" / f"{candidate.candidate_id}.fasta"
        _write_text(
            path,
            f">{candidate.candidate_id}|{candidate.mutation_set}|EXPERIMENTALLY_UNTESTED\n"
            f"{_wrap_fasta(candidate.sequence)}\n",
        )
        individual_fastas.append(path)
    finalist_evidence = [
        record.to_dict()
        for record in evidence
        if record.candidate_id in set(finalist_ids)
    ]
    finalist_evidence_csv = exports / "finalist_evidence.csv"
    pd.DataFrame(finalist_evidence).to_csv(finalist_evidence_csv, index=False)
    finalist_evidence_json = exports / "finalist_evidence.json"
    _write_text(
        finalist_evidence_json,
        json.dumps(finalist_evidence, indent=2, sort_keys=True) + "\n",
    )
    structure_exports = []
    dossiers = []
    render_paths = []
    for candidate in finalists:
        source_record = structure_by_id.get(candidate.candidate_id)
        if source_record is not None:
            source = Path(source_record["structure_path"])
            destination = exports / "structures" / f"{candidate.candidate_id}.pdb"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            structure_exports.append(destination)
            render = reports / "candidates" / f"{candidate.candidate_id}_structure.png"
            _render_structure(destination, candidate, render)
            render_paths.append(render)
        dossier = reports / "candidates" / f"{candidate.candidate_id}.md"
        _write_text(
            dossier,
            _dossier(
                candidate,
                ledger,
                evidence_by_candidate.get(candidate.candidate_id, ()),
                design_by_position,
                review_by_id.get(candidate.candidate_id, {}),
                source_record,
            ),
        )
        dossiers.append(dossier)

    figures = [
        figures_dir / "adaptive_design_funnel.png",
        figures_dir / "adaptive_search_landscape.png",
        figures_dir / "failure_memory.png",
        figures_dir / "structural_design_map.png",
        figures_dir / "finalist_independent_axes.png",
    ]
    _plot_funnel(funnel_counts, figures[0])
    _plot_search_landscape(candidates, figures[1])
    _plot_failure_memory(ledger, figures[2])
    _plot_structural_map(reference_pdb, design_space, finalists, figures[3])
    _plot_finalist_axes(finalists, evidence_by_candidate, figures[4])

    repair_children = [
        candidate for candidate in candidates if candidate.revision_generation > 0
    ]
    strategy_counts = Counter(candidate.strategy.value for candidate in candidates)
    region_counts = Counter(candidate.structural_region for candidate in candidates)
    failure_counts = Counter(failure.category for failure in ledger.failures())
    repair_examples: list[str] = []
    for child in repair_children:
        if not child.parents:
            continue
        trajectory = ledger.repair_trajectory(child.parents[0])
        matching = next(
            (record for record in trajectory if record.get("child_id") == child.candidate_id),
            None,
        )
        if matching is not None:
            repair_examples.append(
                f"- `{matching.get('parent_id')}` → `{matching.get('child_id')}`: "
                f"{matching.get('change_made')}; disposition "
                f"`{matching.get('disposition')}`; deltas "
                f"`{json.dumps(matching.get('evidence_delta', {}), sort_keys=True)}`."
            )
        if len(repair_examples) == 5:
            break
    near_miss_section = ""
    if not finalists:
        rows = []
        for record in near_misses:
            rows.append(
                f"- `{record.get('candidate_id')}` `{record.get('mutation_set', '')}` — "
                f"{record.get('promotion_blocker') or record.get('strongest_case_against_synthesis', 'Promotion blocker not recorded.')}"
                + (
                    f" Bounded repair exhausted: {record.get('repair_exhaustion_reason')}."
                    if record.get("repair_exhausted")
                    else ""
                )
            )
        near_miss_section = (
            "\n## BEST_NEAR_MISS_HYPOTHESES\n\n"
            "These are not finalists. Promotion would require resolving the blockers below; "
            "hard constraints were not relaxed.\n\n"
            + ("\n".join(rows) if rows else "No defensible near-miss hypothesis remained.")
            + "\n"
        )
    finalist_lines = (
        "\n".join(
            f"- [{candidate.candidate_id} — {candidate.mutation_set}]"
            f"(candidates/{candidate.candidate_id}.md): EXPERIMENTALLY UNTESTED"
            for candidate in finalists
        )
        if finalists
        else "No candidate met the promotion standard."
    )
    final_report = reports / "atlas_final_design_report.md"
    _write_text(
        final_report,
        f"""# Atlas final adaptive DP622 design report

## Scientific question

Can an autonomous computational workflow reduce DP622-derived sequence space to a small, transparent set of defensible wet-lab experiments?

## Starting system and provenance

Atlas used the experimentally deposited 23WN DP622-associated/Aβ/Zn complex and restored the residue corresponding to catalytic E96. The canonical model label is `active_like_inferred`, never `published_exact`. The exact active DP622-S2 assay construct is not publicly recoverable with publication-grade certainty.

## Preserved historical benchmark

The Y91F/D126A retrospective result remains `BENCHMARK_FAILED`. It showed that the selected stability/static-geometry evidence did not faithfully reproduce every activity outcome. Atlas did not retune that gate; prospective design is separate.

## Search design

- Unique legal sequences evaluated: **{funnel_counts.get('generated_evaluated', 0):,}**
- Repair/revision children: **{len(repair_children):,}**
- Strategies: {dict(sorted(strategy_counts.items()))}
- Structural regions: {dict(sorted(region_counts.items()))}
- Selection: uncertainty-aware Pareto layers and explicit diversity; no universal Atlas score.
- Failure memory: {len(ledger.failures()):,} persisted evidence-scoped observations.

## Actual search funnel

`{funnel_counts.get('generated_evaluated', 0):,} → {funnel_counts.get('broad_survivors', 0):,} → {funnel_counts.get('structural_analyses', 0):,} → {funnel_counts.get('adversarial_review', 0):,} → {funnel_counts.get('finalists', 0):,}`

## Learning and revision

Round 2 consumed Round-1 position priorities and confidence-scoped failure memory. Round 3 combined evidence-supported singles. The critic sent supported candidates with a diagnosed soft weakness through bounded repair; every repair record stores the parent, preservation goal, change, and evidence delta. Failures remained contextual observations rather than universal biochemical laws.

- Failure-memory categories: {dict(sorted(failure_counts.items()))}
- Repair children evaluated: {len(repair_children):,}

{chr(10).join(repair_examples) if repair_examples else "No novel legal repair child was generated; see the repair-eligibility artifact for every attempted or excluded parent."}

## Final experimental portfolio

{finalist_lines}
{near_miss_section}
## Activity-oriented model decision

No audited activity-oriented model was admitted. ProMEP, EnzyACT, CatPred, UniKP, and PLACER lacked the combined validated target/domain/metal/multichain support required for DP622/Aβ prospective activity inference. Missing activity evidence was never converted to favorable evidence.

## Replicated explicit-solvent MD methodological exclusion

Replicated explicit-solvent MD was evaluated as a prospective evidence layer but excluded from Atlas candidate discrimination after the DP622/Aβ/Zn reference failed reproducible numerical and catalytic/substrate-geometry validation. Missing dynamics evidence never penalizes a candidate and dynamics is not a finalist evidence axis.

## Limitations

- All finalists are **EXPERIMENTALLY UNTESTED**.
- ThermoMPNN/ThermoMPNN-D provide stability-oriented evidence, not catalytic-activity predictions.
- Restrained candidate-specific structures are model-dependent and are not catalytic-activity predictions.
- The 23WN-derived active-like construct is inferred; it is not an exact recovered DP622-S2 assay sequence.
- No claim is made about therapeutic efficacy or plaque dissolution.

## Required wet-lab validation

Expression/solubility, folding/thermal stability, cleavage-site-resolved Aβ assays, matched kinetic measurements, specificity profiling, and Zn-dependence controls are required before any activity conclusion.
""",
    )

    finalists_index = run_dir / "FINALISTS.md"
    _write_text(
        finalists_index,
        "# Atlas experimentally untested finalist hypotheses\n\n"
        + finalist_lines
        + "\n\nThese are wet-lab hypotheses, not demonstrated activity improvements.\n",
    )
    limitations_report = reports / "limitations_report.md"
    _write_text(
        limitations_report,
        """# Atlas scientific limitations

- The 23WN-derived model is `active_like_inferred`, not an experimentally observed exact active DP622-S2 complex.
- The retrospective activity benchmark did not reproduce every measured trend.
- ThermoMPNN and ThermoMPNN-D are stability models with distinct raw scales.
- Restrained mutant-complex geometry and Aβ pose/contact evidence do not establish catalytic activity.
- Replicated explicit-solvent MD failed reference validation and is excluded from candidate discrimination.
- Every finalist is experimentally untested; wet-lab assays determine actual activity.
""",
    )
    run_readme = run_dir / "README.md"
    _write_text(
        run_readme,
        """# Atlas adaptive production run

Workflow: 23WN → active-like DP622/Aβ/Zn reconstruction → biology-aware design space → mechanism-aware proposals → genuine ThermoMPNN/ThermoMPNN-D stability evidence → model-aware Pareto/diversity funnel → candidate-specific mutant complexes → chemistry/catalytic/Aβ structural evidence → bounded repair → adversarial experimental portfolio.

Replicated explicit-solvent MD was tested on the reference and excluded from finalist discrimination. See `replicated_md_methodological_exclusion.json`.

Finalists, if any, are **EXPERIMENTALLY UNTESTED BEST-SUPPORTED WET-LAB HYPOTHESES**. See `FINALISTS.md`, `reports/atlas_final_design_report.md`, and `reproducibility_manifest.json`.
""",
    )
    _write_text(
        reports / "strategy_region_coverage.json",
        json.dumps(
            {
                "strategy_counts": dict(sorted(strategy_counts.items())),
                "region_counts": dict(sorted(region_counts.items())),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    _write_text(
        reports / "double_category_summary.json",
        json.dumps(
            dict(
                sorted(
                    Counter(
                        candidate.double_category
                        for candidate in candidates
                        if candidate.double_category is not None
                    ).items()
                )
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    final_selection_trace = reports / "final_selection_trace.json"
    _write_text(
        final_selection_trace,
        json.dumps(
            {
                "finalist_ids": list(finalist_ids),
                "portfolio_target": min(5, len(finalist_ids)),
                "universal_score": False,
                "experimentally_untested": True,
                "adversarial_reviews": adversarial_reviews,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )

    package_versions = {}
    for name in ("atlas-protein-engineering", "numpy", "pandas", "biopython", "openmm", "langgraph", "torch"):
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[name] = "not-installed"
    manifest_path = run_dir / "reproducibility_manifest.json"
    important = [
        candidate_table,
        evidence_table,
        final_report,
        fasta_path,
        *individual_fastas,
        finalist_evidence_csv,
        finalist_evidence_json,
        finalists_index,
        limitations_report,
        run_readme,
        final_selection_trace,
        *dossiers,
        *structure_exports,
        *figures,
        *render_paths,
    ]
    run_context = (
        json.loads((run_dir / "run_context.json").read_text())
        if (run_dir / "run_context.json").is_file()
        else {}
    )
    adaptive_context = (
        json.loads((run_dir / "adaptive_run_context.json").read_text())
        if (run_dir / "adaptive_run_context.json").is_file()
        else {}
    )
    manifest = {
        "atlas_sha": _git_sha(run_dir),
        "thermompnn_sha": run_context.get("thermompnn_commit", "unknown"),
        "thermompnn_d_sha": run_context.get("thermompnn_d_commit", "unknown"),
        "input_23wn_sha256": run_context.get("input_sha256", "unknown"),
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions,
        "candidate_library_sha256": _sha256(candidate_table),
        "input_structure_sha256": _sha256(reference_pdb),
        "random_seed": seed,
        "generation_policy_version": adaptive_context.get(
            "generation_policy_version", "unknown"
        ),
        "model_label": "active_like_inferred",
        "finalist_ids": list(finalist_ids),
        "funnel_counts": funnel_counts,
        "files": {
            str(path.relative_to(run_dir)): _sha256(path)
            for path in important
            if path.is_file()
        },
        "claim_boundary": "Computational experiment prioritization; wet-lab validation required.",
    }
    _write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return AdaptiveOutputBundle(
        final_report=final_report,
        reproducibility_manifest=manifest_path,
        candidate_dossiers=tuple(dossiers),
        fasta=fasta_path,
        structure_exports=tuple(structure_exports),
        figures=tuple([*figures, *render_paths]),
    )
