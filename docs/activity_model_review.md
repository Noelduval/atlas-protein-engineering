# Activity-oriented model review

**Reviewed:** 2026-08-26  
**Decision:** `not_available`

Atlas reviewed ProMEP, EnzyACT, CatPred, UniKP, and PLACER against the approved
eligibility contract: prediction target, inputs/outputs, released checkpoints,
single/double mutation support, peptide-complex representation, metal treatment,
training domain, de novo applicability, license, runtime compatibility, and
reproducibility. The historical known-mutant result was not used for method
selection or tuning.

## Why Atlas excludes all reviewed models

No reviewed method simultaneously supplies a relevant activity/kinetic target,
reproducible inference artifact, explicit DP622/Aβ peptide-complex context,
documented Zn-site treatment, and validation on a de novo metalloenzyme.

| Model | What it actually predicts | Reproduction status | DP622 decision |
|---|---|---|---|
| ProMEP | WT-conditioned general protein-fitness log-likelihood | Official code is Apache-2.0, but the referenced trained checkpoint is absent from commit `e44e8f4`; monomer context only | Exclude from the activity axis |
| EnzyACT | Increased/decreased mutation activity classification | Official Apache-2.0 repository at `a8ec8c3` contains datasets and result files, not inference code/checkpoints | Exclude as non-reproducible |
| CatPred 1.0.1 | `kcat`, `Km`, and `Ki` from sequence, substrate SMILES, and optional monomer structure | MIT code/checkpoints are available; the default workflow explicitly models one protein sequence rather than a multichain complex | Exclude as peptide/Zn/de novo out-of-domain |
| UniKP | `kcat`, `Km`, and `kcat/Km` from pooled sequence and substrate-SMILES embeddings | GPL-3.0 code and downloadable models are available | Exclude as peptide/Zn/de novo out-of-domain |
| PLACER | Local protein–small-molecule/metal conformational ensembles | Code and weights are available but require a separate CUDA 12.1 environment | Structural model, not an activity predictor; not integrated |

## Primary sources

- [ProMEP paper](https://www.nature.com/articles/s41422-024-00989-2) and [official repository](https://github.com/wenjiegroup/ProMEP)
- [EnzyACT paper](https://doi.org/10.1021/acs.jcim.4c00920) and [official repository](https://github.com/GenScript-IBDPE/EnzyACT)
- [CatPred paper](https://www.nature.com/articles/s41467-025-57215-9) and [official 1.0.1 release](https://github.com/maranasgroup/CatPred/releases/tag/1.0.1)
- [UniKP paper](https://www.nature.com/articles/s41467-023-44113-1) and [official repository](https://github.com/Luo-SynBioLab/UniKP)
- [PLACER paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12625923/) and [official repository](https://github.com/baker-laboratory/PLACER)

## Scientific boundary

Atlas records `activity_oriented` evidence as `unavailable`, with no numerical
value and no favorable imputation. The adaptive workflow remains complete
without this axis. No output may claim predicted turnover, `kcat/Km`, or improved
experimental activity.

