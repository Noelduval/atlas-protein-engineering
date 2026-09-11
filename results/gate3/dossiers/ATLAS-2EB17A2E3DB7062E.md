# ATLAS-2EB17A2E3DB7062E — A130L

**Status: EXPERIMENTALLY UNTESTED**

## Identity

- Candidate ID: `ATLAS-2EB17A2E3DB7062E`
- Exact mutations: `A130L`
- Reference: 23WN-derived `active_like_inferred` DP622/Aβ/Zn model
- Novelty: not identified in the examined VITA/DP622 public materials and not previously evaluated by Atlas before this run; no claim of global historical novelty.
- Complete sequence (215 aa):

```text
RNLELARAADVTVTVADTPEEMYEAAKVAVETVRELAAGDPRRDEYVALAERLFRTGIERGGIAGIAIYADGRRRVFVVA
PSDASDEALIYALAHELAHLIIAEDLERRGLPLSAVPPGVVEGLADVFGLTAYAAYLELKGEKVTLEKWREMQLRLAEET
ERIGREAGLEAHVEGGRIAAEIARRTNEEEAQKLIEEVKPLVEFILGLLRVARTA
```

**Committed exports:** [FASTA](../fastas/ATLAS-2EB17A2E3DB7062E.fasta) · [PDB](../structures/ATLAS-2EB17A2E3DB7062E.pdb)

## Design hypothesis

Test whether A130L can make a conservative chemistry change within the wild-type residue family while preserving DP622 scaffold packing and secondary-structure integrity.

Intended upside: Increase scaffold-support margin for function-oriented perturbations.

Expected risk: The local packing, electrostatics, or interaction network may regress.

Intended physical change: make a conservative chemistry change within the wild-type residue family

Feature intended to remain preserved: DP622 scaffold packing and secondary-structure integrity

Principal biochemical risk: The local packing, electrostatics, or interaction network may regress.

Metal-liability status: `none_identified`; candidate-specific geometry required: `False`.

Double category: `not_applicable`; physical coupling: not applicable; epistasis uncertainty: `not_applicable`.

Known-experiment conflict: `none_identified`.

## Structural context

- `A130L`: distal_stability; 9.311 Å from resolved Aβ; 11.755 Å from Zn; buried.

Mutant complex: `/content/drive/MyDrive/Atlas/checkpoints/atlas-adaptive-t4-aa5f61d8a5fb-e8f61394/structure/ATLAS-2EB17A2E3DB7062E/restrained_relaxation/minimized.pdb`; provenance: `/content/drive/MyDrive/Atlas/checkpoints/atlas-adaptive-t4-aa5f61d8a5fb-e8f61394/structure/ATLAS-2EB17A2E3DB7062E/model_provenance.json`.

## Liability/developability heuristics

- No listed sequence/structure liability heuristic was triggered.

These are policy heuristics, not experimentally validated developability predictions.

## Independent computational evidence

| Axis | Status | Raw value | Uncertainty | Method |
|---|---|---:|---:|---|
| stability | available | -0.500645 | 0.5 | ThermoMPNN |
| substrate_interface | available | 0.349379 | 0.25 | 23WN deposited-distance hypothesis prior |
| catalytic_geometry | available | 0 | 0.35 | 23WN catalytic-proximity perturbation risk |
| liability | available | 0 | 0.2 | explicit sequence/structural liability rules |
| stability_model_aware | available | 0.0221675 | 0.0221894 | within-model empirical rank: ThermoMPNN |
| structure_quality | available | 0.494408 | 0.35 | restrained mutant-complex active-site RMSD |
| catalytic_geometry | available | 1.21022 | 0.35 | catalytic_preorganization maximum grounded-distance deviation |
| substrate_interface | available | 0.555554 | 0.35 | restrained Aβ pose-preservation support |
| stability_model_aware | available | 0.0221675 | 0.0221894 | within-model empirical rank: ThermoMPNN |

ThermoMPNN/ThermoMPNN-D raw values are stability evidence only and are not cross-model commensurate. Model-aware empirical ranks are screening policy values, not biological calibration. `catalytic_preorganization` values describe preservation/deviation of an experimentally grounded arrangement, not catalytic-rate prediction.

## Activity-oriented evidence

No activity-oriented model output was admitted; the audited models were not validated for the DP622/Aβ/Zn prospective-design use case.

## Design and revision history

ATLAS-2EB17A2E3DB7062E (A130L)

No repair child occurs in this candidate's lineage.

## Scientific critic

### Strongest argument for synthesis

The candidate preserves multiple independent computational-support axes and represents a falsifiable experiment.

### Strongest argument against synthesis

No computational axis demonstrates catalytic turnover; the historical benchmark shows stability/static geometry can disagree with activity.

Unresolved uncertainty: stability: uncertainty=0.5

## Falsifiable experimental hypothesis

Test whether A130L can make a conservative chemistry change within the wild-type residue family while preserving DP622 scaffold packing and secondary-structure integrity. This hypothesis is falsified by failure to preserve expression/folding or by Aβ-cleavage behavior that is not improved or is worse than the matched active-like reference under the same assay.

Specific falsification result: Failure to express/fold, loss of Aβ cleavage relative to reference, or loss of the hypothesized structural behavior falsifies the design hypothesis.

## Recommended wet-lab assays

1. Expression yield and solubility versus the matched reference.
2. Folding/thermal-stability measurement (for example nanoDSF or DSF), interpreted separately from activity.
3. Aβ cleavage-site product analysis by LC–MS or an equivalently specific assay.
4. Initial-rate kinetics across substrate concentrations to estimate `kcat`, `Km`, and `kcat/Km` only from experiments.
5. Cleavage selectivity/specificity profiling and Zn-dependence controls.

## Claim boundary

Atlas prioritizes this sequence as an experiment. It does not claim experimentally improved activity, therapeutic efficacy, plaque dissolution, or identity with an unavailable exact DP622-S2 assay construct.
