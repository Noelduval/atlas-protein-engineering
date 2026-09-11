# Atlas final adaptive DP622 design report

## Scientific question

Can an autonomous computational workflow reduce DP622-derived sequence space to a small, transparent set of defensible wet-lab experiments?

## Starting system and provenance

Atlas used the experimentally deposited 23WN DP622-associated/Aβ/Zn complex and restored the residue corresponding to catalytic E96. The canonical model label is `active_like_inferred`, never `published_exact`. The exact active DP622-S2 assay construct is not publicly recoverable with publication-grade certainty.

## Preserved historical benchmark

The Y91F/D126A retrospective result remains `BENCHMARK_FAILED`. It showed that the selected stability/static-geometry evidence did not faithfully reproduce every activity outcome. Atlas did not retune that gate; prospective design is separate.

## Search design

- Unique legal sequences evaluated: **5,000**
- Repair/revision children: **0**
- Strategies: {'conservative_explorer': 518, 'evidence_guided_combination_explorer': 2969, 'second_shell_preorganization_explorer': 618, 'stability_support_explorer': 559, 'substrate_interface_explorer': 336}
- Structural regions: {'cross_region': 1957, 'distal_stability': 864, 'scaffold_surface': 729, 'second_shell': 952, 'substrate_interface': 498}
- Selection: uncertainty-aware Pareto layers and explicit diversity; no universal Atlas score.
- Failure memory: 325 persisted evidence-scoped observations.

## Actual search funnel

`5,000 → 500 → 146 → 10 → 5`

## Learning and revision

Round 2 consumed Round-1 position priorities and confidence-scoped failure memory. Round 3 combined evidence-supported singles. The critic sent supported candidates with a diagnosed soft weakness through bounded repair; every repair record stores the parent, preservation goal, change, and evidence delta. Failures remained contextual observations rather than universal biochemical laws.

- Failure-memory categories: {'catastrophic_structural_clash': 1, 'mutation_associated_stability_regression': 197, 'recurrent_stability_regression': 127}
- Repair children evaluated: 0

No novel legal repair child was generated; see the repair-eligibility artifact for every attempted or excluded parent.

## Final experimental portfolio

- [ATLAS-0215A1AE2C58FD57 — G65M/Q153M](dossiers/ATLAS-0215A1AE2C58FD57.md): EXPERIMENTALLY UNTESTED
- [ATLAS-1FB316267B702519 — A16C/Q153C](dossiers/ATLAS-1FB316267B702519.md): EXPERIMENTALLY UNTESTED
- [ATLAS-2EB17A2E3DB7062E — A130L](dossiers/ATLAS-2EB17A2E3DB7062E.md): EXPERIMENTALLY UNTESTED
- [ATLAS-39F921440EC002E5 — E104V](dossiers/ATLAS-39F921440EC002E5.md): EXPERIMENTALLY UNTESTED
- [ATLAS-66C6BB6CAA1C310A — A37Y](dossiers/ATLAS-66C6BB6CAA1C310A.md): EXPERIMENTALLY UNTESTED

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
