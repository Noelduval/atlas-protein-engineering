# Atlas v1

Atlas v1 is an adaptive computational workflow for prioritizing wet-lab mutation experiments on the DP622 metalloprotease scaffold. Its sole production authority is:

```text
atlas adaptive-run
→ active-like DP622/Aβ/Zn reconstruction
→ biology-aware design space
→ mechanism-aware adaptive proposals
→ genuine ThermoMPNN/ThermoMPNN-D stability inference
→ model-aware Pareto/diversity funnel
→ candidate-specific mutant complexes
→ chemistry, catalytic-preorganization, and Aβ pose/contact evidence
→ bounded repair and adversarial review
→ orthogonal fold-recovery adapter, bounded Aβ specificity and local-pose robustness
→ sequence/static-structure developability risk screen and wet-lab dossiers
→ ≤5 experimentally untested wet-lab hypotheses
```

Atlas prioritizes physical plausibility and experimental information value. It does not predict `kcat`, `Km`, `kcat/Km`, therapeutic efficacy, plaque clearance, or experimental validation.

## Starting system

PDB 23WN contains a pre-catalytic cryo-EM DP622 E96Q/Aβ/Zn complex. Deposited chain A residues 25–239 map to DP622 residues 1–215. Atlas retains the resolved Aβ fragment and Zn, renumbers DP622, and restores deposited Q120 to catalytic E96. The canonical model is therefore `active_like_inferred`, not an experimentally observed exact active DP622-S2 complex.

## Scientific policy

- Indispensable catalytic and Zn-coordinating residues are hard protected.
- `allowed_substitution_classes` directly controls legal mutation identities.
- Cys, Gly, and Pro have distinct context-specific policies; they are not one conservative class.
- Buried charge and exposed-hydrophobe liabilities are explicit.
- H/C/D/E introductions in the existing Zn-proximal design context receive explicit metal-liability treatment and candidate-specific geometry checks.
- Doubles are categorized as local/coupled, function plus stability-rescue, or orthogonal-mechanism experiments, with high epistasis uncertainty.
- The experimentally characterized Y91F/D126A double is retrospective evidence and cannot be presented as a novel finalist.
- ThermoMPNN and ThermoMPNN-D raw values remain separate. Pareto screening uses within-model empirical ranks and never subtracts or combines their raw scales.
- Deposited-coordinate distances are pre-structure context/stratification, not mutant-performance measurements.
- Final candidates are labeled **EXPERIMENTALLY UNTESTED BEST-SUPPORTED WET-LAB HYPOTHESES**.

## Replicated MD exclusion

Replicated explicit-solvent MD was evaluated as a prospective evidence layer but excluded from Atlas candidate discrimination after the DP622/Aβ/Zn reference failed reproducible numerical and catalytic/substrate-geometry validation.

Replicated MD is not a production adaptive stage, evidence-completeness requirement, Pareto axis, critic support/weakness, repair step, dossier axis, or finalist gate. Missing dynamics evidence never penalizes a candidate. Restrained PDBFixer/OpenMM mutant-complex preparation and minimization remain a distinct structure-building step.

## Production components

| Component | Production role |
| --- | --- |
| `design/adaptive_generator.py` | Mechanism-aware single and double proposals |
| `adaptive/screening.py` | Independent-axis, uncertainty-aware Pareto/diversity funnel |
| `adaptive/critic.py` | Hard/soft evidence routing and bounded repair decisions |
| `adaptive_pipeline.py` | Checkpointed production authority |
| `adaptive_backend.py` | Pinned stability inference and candidate-specific structures |
| `reporting/adaptive_outputs.py` | Finalists, dossiers, evidence exports, figures, and manifest |
| `late_stage/` | Checkpoint-reusing orthogonal structure, specificity, pose-robustness, developability, and dossier evidence |

`candidate_generator.py`, `rank_candidates.py`, the historical validation-gated `atlas run` workflow, and legacy orchestration paths are not authorities for adaptive finalists.

## GPU production

Use Python 3.10–3.12 and the pinned upstream repositories:

```bash
git clone https://github.com/Kuhlman-Lab/ThermoMPNN.git .external/ThermoMPNN
git -C .external/ThermoMPNN checkout 2b04fd370e399911b1fa5848112cc9013f084110
git clone https://github.com/Kuhlman-Lab/ThermoMPNN-D.git .external/ThermoMPNN-D
git -C .external/ThermoMPNN-D checkout df9a75aaddb674a7c4c193005031fc0536d325fb

atlas adaptive-run \
  --input data/23WN.cif \
  --output-root outputs \
  --atlas-repo . \
  --thermompnn-repo .external/ThermoMPNN \
  --thermompnn-d-repo .external/ThermoMPNN-D \
  --run-id atlas-adaptive-production \
  --candidate-budget 5000

atlas late-stage \
  --run-dir outputs/atlas-adaptive-production \
  --input data/23WN.cif \
  --resume
```

The Colab entry point is [`notebooks/Atlas_DP622_Colab.ipynb`](notebooks/Atlas_DP622_Colab.ipynb). It uses a T4 runtime, pinned model revisions, a single managed scientific Python environment, Google Drive checkpoints, `atlas adaptive-run --resume`, and then `atlas late-stage --resume` only on the persisted adversarial set. If the real external sequence-to-structure predictor is absent or fails, that evidence is recorded as unavailable/invalid; Atlas never substitutes synthetic structure-prediction evidence.

## Production artifacts

A completed run contains the design-space CSV/JSON, generation policy and coverage summary, full SQLite ledger and JSONL events, raw and normalized stability evidence, candidate-specific PDBs and provenance, repair trajectories, adversarial reviews, near misses, final selection trace, finalist FASTAs/PDBs/evidence/dossiers, figures, limitations report, reproducibility manifest, completion audit, and execution status.

Completion is proved by persisted artifacts, not terminal text. Wet-lab expression, folding/stability, cleavage-site-resolved Aβ assays, matched kinetics, specificity profiling, and Zn-dependence controls determine actual performance.
