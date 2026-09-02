# ATLAS-0215A1AE2C58FD57 — G65M/Q153M

**Status: EXPERIMENTALLY UNTESTED**

## Identity

- Candidate ID: `ATLAS-0215A1AE2C58FD57`
- Exact mutations: `G65M/Q153M`
- Reference: 23WN-derived `active_like_inferred` DP622/Aβ/Zn model
- Novelty: not identified in the examined VITA/DP622 public materials and not previously evaluated by Atlas before this run; no claim of global historical novelty.
- Complete sequence (215 aa):

```text
RNLELARAADVTVTVADTPEEMYEAAKVAVETVRELAAGDPRRDEYVALAERLFRTGIERGGIAMIAIYADGRRRVFVVA
PSDASDEALIYALAHELAHLIIAEDLERRGLPLSAVPPGVVEGLADVFGATAYAAYLELKGEKVTLEKWREMMLRLAEET
ERIGREAGLEAHVEGGRIAAEIARRTNEEEAQKLIEEVKPLVEFILGLLRVARTA
```

## Design hypothesis

Combine Q153M and G65M as an explicit epistasis hypothesis from complementary single-mutant designs.

Intended upside: Retain independent supporting features while testing whether their combination broadens stability/function tradeoffs.

Expected risk: Combination-specific epistasis may negate either single-mutant benefit.

Intended physical change: Combine increase local side-chain volume by approximately 19.1 Å³ to tune packing with increase local side-chain volume by approximately 102.8 Å³ to tune packing.

Feature intended to remain preserved: DP622 scaffold packing and secondary-structure integrity; resolved Aβ pose and catalytic preorganization.

Principal biochemical risk: High combination-specific epistasis uncertainty; child structure must be compared with both parents.

Metal-liability status: `none_identified`; candidate-specific geometry required: `False`.

Double category: `FUNCTION_STABILITY_RESCUE_DOUBLE`; physical coupling: function-oriented mutation paired with a distal/scaffold-support hypothesis; epistasis uncertainty: `high`.

Known-experiment conflict: `none_identified`.

## Structural context

- `G65M`: substrate_interface; 3.060 Å from resolved Aβ; 6.475 Å from Zn; buried.
- `Q153M`: distal_stability; 11.380 Å from resolved Aβ; 13.787 Å from Zn; buried.

Mutant complex: `/content/drive/MyDrive/Atlas/checkpoints/atlas-adaptive-t4-aa5f61d8a5fb-e8f61394/structure/ATLAS-0215A1AE2C58FD57/restrained_relaxation/minimized.pdb`; provenance: `/content/drive/MyDrive/Atlas/checkpoints/atlas-adaptive-t4-aa5f61d8a5fb-e8f61394/structure/ATLAS-0215A1AE2C58FD57/model_provenance.json`.

## Liability/developability heuristics

- No listed sequence/structure liability heuristic was triggered.

These are policy heuristics, not experimentally validated developability predictions.

## Independent computational evidence

| Axis | Status | Raw value | Uncertainty | Method |
|---|---|---:|---:|---|
| stability | available | -1.49974 | 0.75 | ThermoMPNN-D epistatic targeted |
| substrate_interface | available | 0.620355 | 0.25 | 23WN deposited-distance hypothesis prior |
| catalytic_geometry | available | 0.35252 | 0.35 | 23WN catalytic-proximity perturbation risk |
| liability | available | 0 | 0.2 | explicit sequence/structural liability rules |
| stability_model_aware | available | 0.0114555 | 0.0183525 | within-model empirical rank: ThermoMPNN-D epistatic targeted |
| structure_quality | available | 0.504904 | 0.35 | restrained mutant-complex active-site RMSD |
| catalytic_geometry | available | 1.15349 | 0.35 | catalytic_preorganization maximum grounded-distance deviation |
| substrate_interface | available | 0.546896 | 0.35 | restrained Aβ pose-preservation support |
| stability_model_aware | available | 0.0114555 | 0.0183525 | within-model empirical rank: ThermoMPNN-D epistatic targeted |

ThermoMPNN/ThermoMPNN-D raw values are stability evidence only and are not cross-model commensurate. Model-aware empirical ranks are screening policy values, not biological calibration. `catalytic_preorganization` values describe preservation/deviation of an experimentally grounded arrangement, not catalytic-rate prediction.

## Activity-oriented evidence

No activity-oriented model output was admitted; the audited models were not validated for the DP622/Aβ/Zn prospective-design use case.

## Design and revision history

ATLAS-7BD6CFB52D8C4EE6 (Q153M) → ATLAS-BD6DC55397A85E01 (G65M) → ATLAS-0215A1AE2C58FD57 (G65M/Q153M)

No repair child occurs in this candidate's lineage.

## Scientific critic

### Strongest argument for synthesis

The candidate preserves multiple independent computational-support axes and represents a falsifiable experiment.

### Strongest argument against synthesis

No computational axis demonstrates catalytic turnover; the historical benchmark shows stability/static geometry can disagree with activity.

Unresolved uncertainty: stability: uncertainty=0.75

## Falsifiable experimental hypothesis

Combine Q153M and G65M as an explicit epistasis hypothesis from complementary single-mutant designs. This hypothesis is falsified by failure to preserve expression/folding or by Aβ-cleavage behavior that is not improved or is worse than the matched active-like reference under the same assay.

Specific falsification result: Failure to express/fold, loss of Aβ cleavage relative to reference, or loss of the hypothesized structural behavior falsifies the design hypothesis.

## Recommended wet-lab assays

1. Expression yield and solubility versus the matched reference.
2. Folding/thermal-stability measurement (for example nanoDSF or DSF), interpreted separately from activity.
3. Aβ cleavage-site product analysis by LC–MS or an equivalently specific assay.
4. Initial-rate kinetics across substrate concentrations to estimate `kcat`, `Km`, and `kcat/Km` only from experiments.
5. Cleavage selectivity/specificity profiling and Zn-dependence controls.

## Claim boundary

Atlas prioritizes this sequence as an experiment. It does not claim experimentally improved activity, therapeutic efficacy, plaque dissolution, or identity with an unavailable exact DP622-S2 assay construct.
