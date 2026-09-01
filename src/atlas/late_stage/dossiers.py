"""Generate bounded, claim-safe finalist wet-lab handoff dossiers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from atlas.adaptive.models import CandidateRecord


REQUIRED_HYPOTHESIS_LABEL = (
    "EXPERIMENTALLY UNTESTED BEST-SUPPORTED WET-LAB HYPOTHESIS"
)

_REQUIRED_EVIDENCE = (
    "why_atlas_proposed",
    "thermompnn",
    "orthogonal_structure",
    "catalytic_preorganization",
    "abeta_contacts",
    "specificity",
    "pose_robustness",
    "developability",
    "disagreements_uncertainty",
    "reason_it_beat_near_misses",
    "falsification_criteria",
)

_FORBIDDEN_CLAIMS = (
    "better catalysis",
    "measured specificity",
    "therapeutic efficacy",
    "plaque clearance",
    "experimental validation",
    "experimentally validated",
    "bbb penetration",
    "favorable pk",
    "non-immunogenic",
    "clinical developability",
    "clinically developable",
)

_EXPERIMENTAL_SEQUENCE = """expression/purification
→ folding/stability
→ cleavage-site-resolved Aβ assay
→ matched kinetic characterization
→ Zn-dependence controls
→ non-cognate specificity profiling
→ structural confirmation if warranted"""


def _json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)


def _has_content(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Mapping, Sequence)):
        return bool(value)
    return True


def _validate_inputs(
    candidate: CandidateRecord,
    candidate_structure: str | Path,
    late_stage_evidence: Mapping[str, object],
    near_misses: Sequence[Mapping[str, object]],
) -> None:
    if not str(candidate_structure).strip():
        raise ValueError("candidate-specific structure is required")
    missing = [
        key
        for key in _REQUIRED_EVIDENCE
        if key not in late_stage_evidence or not _has_content(late_stage_evidence[key])
    ]
    if missing:
        raise ValueError("missing required dossier evidence: " + ", ".join(missing))

    thermompnn = _json(late_stage_evidence["thermompnn"])
    if "ThermoMPNN" not in thermompnn or "ThermoMPNN-D" not in thermompnn:
        raise ValueError("thermompnn evidence must preserve both raw model identities")

    searchable = _json(
        {
            "candidate": candidate.to_dict(),
            "candidate_structure": str(candidate_structure),
            "late_stage_evidence": late_stage_evidence,
            "near_misses": near_misses,
        }
    ).casefold()
    for claim in _FORBIDDEN_CLAIMS:
        if claim in searchable:
            raise ValueError(f"forbidden scientific claim in dossier input: {claim}")


def _render(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "\n".join(f"- {item if isinstance(item, str) else _json(item)}" for item in value)
    return f"```json\n{_json(value)}\n```"


def _wrap_fasta(sequence: str, width: int = 80) -> str:
    return "\n".join(
        sequence[index : index + width] for index in range(0, len(sequence), width)
    )


def _near_miss_text(near_misses: Sequence[Mapping[str, object]]) -> str:
    if not near_misses:
        return "No near-miss records were supplied; no pairwise comparison is available."
    return "\n\n".join(
        f"### {record.get('candidate_id', 'unidentified near-miss')}\n\n"
        f"{_render(dict(record))}"
        for record in near_misses
    )


def build_finalist_dossier(
    candidate: CandidateRecord,
    *,
    candidate_structure: str | Path,
    late_stage_evidence: Mapping[str, object],
    near_misses: Sequence[Mapping[str, object]],
) -> str:
    """Render one dossier solely from candidate and supplied late-stage evidence."""
    _validate_inputs(candidate, candidate_structure, late_stage_evidence, near_misses)
    evidence = late_stage_evidence
    return f"""# {candidate.candidate_id} finalist wet-lab handoff

**{REQUIRED_HYPOTHESIS_LABEL}**

## Candidate ID

`{candidate.candidate_id}`

## Exact mutations

`{candidate.mutation_set}`

## FASTA

```fasta
>{candidate.candidate_id}|{candidate.mutation_set}
{_wrap_fasta(candidate.sequence)}
```

## Candidate-specific structure

`{candidate_structure}`

## Mechanistic hypothesis

{candidate.hypothesis}

Intended physical change: {candidate.intended_physical_change}

Feature to preserve: {candidate.feature_to_preserve}

Principal biochemical risk: {candidate.principal_biochemical_risk}

## Why Atlas proposed it

{_render(evidence['why_atlas_proposed'])}

## ThermoMPNN/ThermoMPNN-D evidence

{_render(evidence['thermompnn'])}

Raw outputs from the two stability-oriented models remain separate scales.

## Orthogonal structure evidence

{_render(evidence['orthogonal_structure'])}

## Catalytic/preorganization evidence

{_render(evidence['catalytic_preorganization'])}

## Aβ contact evidence

{_render(evidence['abeta_contacts'])}

## Specificity evidence

{_render(evidence['specificity'])}

This evidence is limited to the stated computational peptide panel.

## Pose robustness

{_render(evidence['pose_robustness'])}

This is a bounded local perturbation test of the resolved 23WN-derived pose.

## Developability risks

{_render(evidence['developability'])}

## Disagreements/uncertainty

{_render(evidence['disagreements_uncertainty'])}

## Why it beat near-misses

{_render(evidence['reason_it_beat_near_misses'])}

## Near-miss comparison

{_near_miss_text(near_misses)}

## Explicit falsification criteria

{_render(evidence['falsification_criteria'])}

## Recommended wet-lab sequence

{_EXPERIMENTAL_SEQUENCE}
"""


def write_finalist_dossiers(
    *,
    output_dir: Path,
    finalists: Sequence[CandidateRecord],
    candidate_structures: Mapping[str, str | Path],
    late_stage_evidence: Mapping[str, Mapping[str, Any]],
    near_misses: Sequence[Mapping[str, object]],
) -> tuple[Path, ...]:
    """Write one atomic Markdown dossier per finalist, with a hard five-item cap."""
    if len(finalists) > 5:
        raise ValueError("wet-lab handoff supports at most five finalists")

    rendered: list[tuple[CandidateRecord, str]] = []
    for candidate in finalists:
        if candidate.candidate_id not in candidate_structures:
            raise ValueError(f"missing candidate-specific structure: {candidate.candidate_id}")
        if candidate.candidate_id not in late_stage_evidence:
            raise ValueError(f"missing late-stage evidence: {candidate.candidate_id}")
        rendered.append(
            (
                candidate,
                build_finalist_dossier(
                    candidate,
                    candidate_structure=candidate_structures[candidate.candidate_id],
                    late_stage_evidence=late_stage_evidence[candidate.candidate_id],
                    near_misses=near_misses,
                ),
            )
        )

    destination = Path(output_dir)
    paths: list[Path] = []
    for candidate, text in rendered:
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / f"{candidate.candidate_id}.md"
        temporary = path.with_suffix(".md.tmp")
        temporary.write_text(text)
        temporary.replace(path)
        paths.append(path)
    return tuple(paths)
