# Atlas

Atlas is an autonomous computational protein-engineering system that generated
and screened 5,000 DP622-inspired metalloprotease variants and prioritized five
experimentally testable candidates for Aβ-related wet-lab evaluation.

**EXPERIMENTALLY UNTESTED:** these are computationally prioritized hypotheses,
not validated improvements in Aβ cleavage.

## Result

The completed campaign executed this funnel:

```text
5,000 unique legal variants → 500 broad survivors → 146 structure-level analyses
→ 10 adversarially reviewed candidates → 5 finalists
```

| Candidate | Mutations | Design rationale | Supporting evidence | Key uncertainty | Files |
| --- | --- | --- | --- | --- | --- |
| [ATLAS-0215A1AE2C58FD57](results/gate3/dossiers/ATLAS-0215A1AE2C58FD57.md) | G65M/Q153M | Function-oriented change paired with distal support | Independent stability, structure, and interface support | Epistatic double; wet-lab behavior unknown | [FASTA](results/gate3/fastas/ATLAS-0215A1AE2C58FD57.fasta) · [PDB](results/gate3/structures/ATLAS-0215A1AE2C58FD57.pdb) |
| [ATLAS-1FB316267B702519](results/gate3/dossiers/ATLAS-1FB316267B702519.md) | A16C/Q153C | Orthogonal-mechanism epistatic double | Preserved structure/interface support | Cysteine liability is soft; wet-lab behavior unknown | [FASTA](results/gate3/fastas/ATLAS-1FB316267B702519.fasta) · [PDB](results/gate3/structures/ATLAS-1FB316267B702519.pdb) |
| [ATLAS-2EB17A2E3DB7062E](results/gate3/dossiers/ATLAS-2EB17A2E3DB7062E.md) | A130L | Conservative distal-stability explorer | Stability with preserved required support | Expression, folding, and activity unknown | [FASTA](results/gate3/fastas/ATLAS-2EB17A2E3DB7062E.fasta) · [PDB](results/gate3/structures/ATLAS-2EB17A2E3DB7062E.pdb) |
| [ATLAS-39F921440EC002E5](results/gate3/dossiers/ATLAS-39F921440EC002E5.md) | E104V | Second-shell preorganization explorer | Preserved structure/interface support | Charge-class warning; wet-lab behavior unknown | [FASTA](results/gate3/fastas/ATLAS-39F921440EC002E5.fasta) · [PDB](results/gate3/structures/ATLAS-39F921440EC002E5.pdb) |
| [ATLAS-66C6BB6CAA1C310A](results/gate3/dossiers/ATLAS-66C6BB6CAA1C310A.md) | A37Y | Substrate-interface explorer | Strongest retained interface support | Static support does not establish cleavage | [FASTA](results/gate3/fastas/ATLAS-66C6BB6CAA1C310A.fasta) · [PDB](results/gate3/structures/ATLAS-66C6BB6CAA1C310A.pdb) |

See the [finalist summary](results/gate3/FINALIST_SUMMARY.md) and [final design report](results/gate3/atlas_final_design_report.md) for the artifact-backed result.

## Why Atlas

Atlas tackles a VITA-inspired problem: optimizing a metalloprotease scaffold for
Aβ-related cleavage while keeping catalytic chemistry, structural plausibility,
stability, interface evidence, and developability visible. The challenge is
not producing one opaque score; it is orchestrating independent evidence,
recording disagreement, remembering failures, and preserving provenance.

## How Atlas works

```text
Published structural evidence → active-like reconstruction → constrained variant generation
→ stability screening → structure construction / relaxation → catalytic + Aβ-interface analysis
→ liability / developability checks → adversarial review → frozen finalist portfolio
```

The pipeline keeps stability, structure, geometry, interface, and developability
evidence separate before combining them into a ranked shortlist. Each candidate
retains the evidence and limitations behind its selection, so the final result is
auditable rather than a single unexplained score.

## Architecture

The production path is centered on `adaptive_pipeline.py`, with mechanism-aware
design in `design/`, evidence-aware screening and critic logic in `adaptive/`,
candidate-specific structure and geometry analysis in `structure/` and
`geometry/`, late-stage checks in `late_stage/`, and artifact exports in
`reporting/`. It uses Python, BioPython, OpenMM, ThermoMPNN, ThermoMPNN-D, and
a typed orchestration/state system. The [Colab notebook](notebooks/Atlas_DP622_Colab.ipynb)
is the GPU execution entry point with pinned model revisions.

## Scientific integrity and limitations

- The starting model is the 23WN-derived `active_like_inferred` reconstruction,
  not an experimentally observed exact active DP622-S2 complex.
- The exact active DP622-S2 assay construct was not publicly recovered.
- ThermoMPNN and ThermoMPNN-D provide stability evidence, not activity predictions.
- Static geometry and interface preservation do not prove cleavage.
- Replicated MD was excluded from candidate discrimination after reference-validation failure.
- All five finalists require expression/folding, cleavage, kinetics, selectivity,
  and Zn-dependence wet-lab validation.

## Explore and reproduce

- [Finalists and funnel](results/gate3/FINALIST_SUMMARY.md)
- [Final design report](results/gate3/atlas_final_design_report.md)
- [Combined FASTA](results/gate3/fastas/finalists.fasta)
- [Reproducibility manifest](results/gate3/reproducibility_manifest.json)
- [Completion audit](results/gate3/completion_audit.json)
- [Limitations report](results/gate3/limitations_report.md)
- [Reproduction guide](docs/reproduction.md)

For a quick review, start with the finalist summary and one dossier/PDB pair.
For code verification, run `python -m pytest -q`. Full execution follows the
pinned GPU Colab path in the reproduction guide.

## What this demonstrates

Scientific workflow orchestration, computational protein design, evidence and
provenance engineering, model-disagreement handling, simulation integration,
reproducible candidate selection, and failure-aware research automation.
