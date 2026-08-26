# Atlas v1 benchmark failure analysis

## Scope and decision

**VERIFIED FACT** — This diagnostic pass uses frozen run
`atlas-t4-b50fe8a6c842-e8f61394` and does not generate candidates, change a
threshold, modify the gate, add benchmark evidence, or rerun either GPU model.
The downloaded export has SHA-256
`745ac3e71c72a3fa23564ee605045a821bcc4672c5cec47e50b84829856efeda`.

**VERIFIED FACT** — The export's `run_context.json` and `run_manifest.json`
record:

- Atlas `b50fe8a6c84237454dcb7b51734fb650f85c8ddc`
- ThermoMPNN `2b04fd370e399911b1fa5848112cc9013f084110`
- ThermoMPNN-D `df9a75aaddb674a7c4c193005031fc0536d325fb`
- 23WN SHA-256 `e8f61394df86c741694affb8b3067f4e17ac72245a37e7cdd9cf875dc9efe63a`
- policy `atlas-v1-fixed-gate-2026-08-19`
- `dynamics_mode: minimize`

**COMPUTATIONAL RESULT** — The primary root-cause classification is **D —
BENCHMARK CONSTRUCT-VALIDITY PROBLEM**. The implementation evaluated the
specified mutations and the gate executed its frozen rules correctly, but the
validation target is catalytic efficiency while the evidence stack measures
predicted folding-stability change and local geometry in a static/restrained
coordinate model. Those observables do not constitute a reasonable test of the
reported kinetic phenotype.

## Question 1 — DP622-S2 computational seed

### Status: QUESTIONABLE

> **The Atlas DP622-S2 seed is a 215-residue active-like computational
> coordinate fragment extracted from 23WN author-chain A residues 25–239,
> renumbered as DP622 residues 1–215, with deposited inactive Q120 changed
> isosterically to DP622 E96 by renaming NE2 to OE2 without moving any atom; it
> retains the experimentally resolved Aβ42 chain-B segment 34–41 (LMVGGVVI)
> and the deposited Zn(II), but it is not a verified full active DP622-S2
> sequence or an experimentally observed active complex.**

### Sequence, coverage, and mapping

**VERIFIED FACT** — 23WN entity 1 is the deposited 1,513-residue `DP622 E96Q
mutant` fusion construct. Author chain A has coordinates for 225 contiguous
standard residues, author positions 15–239. Atlas deliberately excludes the
resolved author positions 15–24 (`EVLFQGPGGT`) and extracts positions 25–239.
The exact mapping implemented and emitted is:

`DP622 position = 23WN author-chain-A position - 24`

Thus deposited 25–239 maps one-to-one to output A1–A215. Deposited Q120 maps to
output E96; Y115, D150, and H196 map to Y91, D126, and H172. The exact output
protein sequence is:

```text
RNLELARAADVTVTVADTPEEMYEAAKVAVETVRELAAGDPRRDEYVALAERLFRTGIERGGIAGIAIYADGRRRVFVVAPSDASDEALIYALAHELAHLIIAEDLERRGLPLSAVPPGVVEGLADVFGATAYAAYLELKGEKVTLEKWREMQLRLAEETERIGREAGLEAHVEGGRIAAEIARRTNEEEAQKLIEEVKPLVEFILGLLRVARTA
```

**VERIFIED FACT** — Direct atom-by-atom comparison found all 1,659 retained
protein-atom coordinates identical to 23WN. The only protein identity/atom
change is deposited GLN A120 → output GLU A96 and atom `NE2` (element N) →
`OE2` (element O), at the unchanged coordinate `(122.564, 119.924, 124.648)` Å.
No residue or missing atom was built, relaxed, or otherwise modeled. Output PDB
SHA-256 is
`698a48727ed3b8c0da4f0fbc016a5c80bc5a62bb40ade05a60e36a6ed2ee0f96`.

**VERIFIED FACT** — 23WN chain B is entity 2, `Amyloid-beta protein 42`, aligned
to UniProt P05067 Aβ positions 1–42. The eight resolved standard residues are
author B34–B41, sequence `LMVGGVVI`; Atlas retains all 53 atoms with unchanged
coordinates as `ATOM` records. The deposition's `struct_conn` explicitly joins
Zn to B38 GLY O at 2.287 Å, so the substrate assignment is not inferred from a
generic `HETATM` record.

**VERIFIED FACT** — The unique deposited zinc atom is present in output chain C
as `HETATM ZN C1601` at `(117.794, 118.847, 121.647)` Å, unchanged from 23WN.

**LIMITATION** — The committed, pre-existing official-source recovery record
states that the exact active DP622-S2 sequence was unavailable after searching
the VITA article, supplement, RCSB/EMDB records, and the authors' repository.
The publication describes 23WN as a rigid fusion construct whose catalytic
E96 was mutated to Q to trap the pre-catalytic state; it does not publish a
standalone active DP622-S2 sequence. Therefore the output is internally exact
as the documented active-like fragment, but its identity as the exact active
enzyme cannot be verified. This warrants `QUESTIONABLE`, not `VERIFIED`; there
is no evidence of a different residue that would justify `INCORRECT`.

## Question 2 — Y91F/D126A identity

### Status: VERIFIED

**VERIFIED FACT** — Both stability models receive the reconstructed WT chain-A
structure plus mutation identities, as their intended inference interface.
They do not consume the edited mutant PDB. The edited benchmark PDB is consumed
by the geometry/OpenMM stages. This distinction is expected behavior, not a
mutation mismatch.

| Mutation | Canonical DP622 | Deposited 23WN | Output chain/residue | WT → mutant | ThermoMPNN raw/internal index | ThermoMPNN-D internal/output identity | Final mutant-PDB identity |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| Y91F | 91 | A115 | A91 | TYR → PHE | zero-based 90; raw `Y90F` | zero-based 90; raw/PDB-renumbered `YA91F` | PHE A91; `OH` removed |
| D126A | 126 | A150 | A126 | ASP → ALA | zero-based 125; raw `D125A` | zero-based 125; raw/PDB-renumbered `DA126A` | ALA A126; `CG/OD1/OD2` removed |

**VERIFIED FACT** — Sequence comparison of frozen
`known_mutant_pdbs/Y91F_D126A.pdb` against frozen WT found exactly two residue
changes: Y91F and D126A. Both structures contain 215 chain-A residues. No other
residue identity or atom set differs. The double-mutant PDB SHA-256 is
`181c4bdff7d571f4ac00289942f9c34d6f12db79af3728c8536d6a72c6de4e5d`.

**VERIFIED FACT** — The pinned ThermoMPNN sweep enumerates zero-based sequence
positions (`seq_pos`) and the Atlas adapter maps one-based DP622 labels to those
positions. The pinned ThermoMPNN-D tensors also use zero-based positions, its
formatter adds one, and its renumbering step adds chain-qualified PDB residue
identities. Because the input PDB is already renumbered A1–A215, the frozen raw
double label is `YA91F:DA126A`, not deposited `YA115F:DA150A`. Atlas first tries
the deposited canonical key, then correctly falls back to the raw DP622 key.

## Question 3 — score semantics and reproduction

### ThermoMPNN

**VERIFIED FACT** — The pinned model predicts thermodynamic folding-stability
change, ΔΔG°, in kcal/mol for a single point mutation. Its head predicts one
value per amino-acid identity and computes `predicted mutant ΔG° - predicted WT
ΔG°`. Therefore the baseline is the same-structure WT residue and a self
mutation is exactly zero. Negative values are stabilizing/favorable for folding
stability; positive values are destabilizing/unfavorable. This is not an
activity, affinity, catalytic-efficiency, or transition-state score. See the
[ThermoMPNN paper](https://doi.org/10.1073/pnas.2314853121), pinned
[`transfer_model.py`](https://github.com/Kuhlman-Lab/ThermoMPNN/blob/2b04fd370e399911b1fa5848112cc9013f084110/transfer_model.py), and pinned
[`SSM.py`](https://github.com/Kuhlman-Lab/ThermoMPNN/blob/2b04fd370e399911b1fa5848112cc9013f084110/analysis/SSM.py).

**VERIFIED FACT** — Atlas requires the genuine raw columns `ddG_pred`,
`position`, `wildtype`, and `mutation`; selects the requested zero-based row;
casts `ddG_pred` to `float`; and performs no sign, baseline, scale, or unit
transformation.

### ThermoMPNN-D

**VERIFIED FACT** — The pinned `epistatic` path directly scores the combined
double mutation. It encodes both WT→mutant substitutions and their pairwise
edge, evaluates AB and BA mutation order through the same direct output head,
and returns their mean. The checkpoint configuration uses `single_target:
true`, `subtract_mut: false`, `mut_types: [double]`, and no `data.epi` target
conversion. It is trained against double-mutant ΔΔG relative to the WT protein,
not absolute folding ΔG, a pair-relative value, or the interaction term
`ΔΔG_AB - (ΔΔG_A + ΔΔG_B)`. “Epistatic” describes the interaction-aware
architecture; the emitted number remains the combined mutant's predicted ΔΔG
in kcal/mol. Negative is stabilizing and positive is destabilizing. See the
pinned [README](https://github.com/Kuhlman-Lab/ThermoMPNN-D/blob/df9a75aaddb674a7c4c193005031fc0536d325fb/README.md),
[`v2_ssm.py`](https://github.com/Kuhlman-Lab/ThermoMPNN-D/blob/df9a75aaddb674a7c4c193005031fc0536d325fb/v2_ssm.py), and
[`epistatic.yaml`](https://github.com/Kuhlman-Lab/ThermoMPNN-D/blob/df9a75aaddb674a7c4c193005031fc0536d325fb/examples/configs/epistatic.yaml).

**VERIFIED FACT** — Atlas requires `ddG (kcal/mol)` and `Mutation`, strips
optional chain letters only to make a canonical lookup key, selects the exact
requested pair, casts the value to `float`, and performs no arithmetic or sign
change.

### Frozen raw-output reproduction

| Model | Requested mutation | Frozen raw identity | Frozen raw value | Atlas normalized value | Result |
| --- | --- | --- | ---: | ---: | --- |
| ThermoMPNN | Y91F | `position=90, wildtype=Y, mutation=F` | 0.1736899614334106 | 0.1736899614334106 | MATCH |
| ThermoMPNN | D126A | `position=125, wildtype=D, mutation=A` | -0.0281429290771484 | -0.0281429290771484 | MATCH |
| ThermoMPNN | H172A | `position=171, wildtype=H, mutation=A` | 0.2005031108856201 | 0.2005031108856201 | MATCH |
| ThermoMPNN-D | Y91F/D126A | `YA91F:DA126A`, Cα–Cα 8.4 Å | -0.16294758 | -0.16294758 | MATCH |

**COMPUTATIONAL RESULT** — The frozen Y91F/D126A score is **CORRECTLY
COMPUTED**. The single raw CSV SHA-256 is
`21bf4bb95d4140f7a118c611c0e17e94363e8df756e56686bd42d1764a0587b7`;
the double raw CSV SHA-256 is
`c7a0bfbd99ced4eaa5868c6cc372b602b1fe29e51f1eccbcd35fc1293b85bbb0`.

**INFERENCE** — The outputs share a nominal physical quantity, sign, WT-relative
meaning, and units, but they do not share an inference-time normalization or a
calibrated prediction scale: ThermoMPNN is a mutant-minus-WT single-site head,
whereas ThermoMPNN-D is an independently trained direct double-mutant head.
Their values may each be interpreted within their model, but ThermoMPNN singles
must not be added to or subtracted from the ThermoMPNN-D double output.

**COMPUTATIONAL RESULT** — **Cross-model additivity arithmetic is invalid.**

## Question 4 — frozen evidence and gate execution

### Published label, score, and outcome

| Variant | Published efficiency (M⁻¹ s⁻¹) | Published trend | Model score | Stability class | Static preserved | Minimized preserved | Gate outcome |
| --- | ---: | --- | ---: | --- | --- | --- | --- |
| WT | 325.260000 | reference | 0.000000 | non_regressive | true | true | reference |
| Y91F | 474.190000 | beneficial-looking | 0.173690 | non_regressive | true | true | pass |
| D126A | 384.620000 | beneficial-looking | -0.028143 | non_regressive | true | true | pass |
| H172A | 232.600000 | regressive | 0.200503 | non_regressive | false | false | separated |
| Y91F/D126A | 161.460000 | strongly regressive | -0.162948 | non_regressive | true | true | not_separated |

### Static catalytic geometry

All distances and RMSDs are Å. `—` means the required atom is absent.

| Variant | Zn–scissile O | Zn–H95 NE2 | Zn–H99 NE2 | Zn–E122 O | E96–scissile C | Active RMSD | Substrate RMSD | H172–scissile O | Y91–D126 pocket | Pose drift | Clamp | Complete |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| WT | 2.286568 | 2.301057 | 2.303568 | 2.017240 | 4.225195 | 0.000000 | 0.000000 | 2.796859 | 2.943899 | 0.000000 | 9.534685 | true |
| Y91F | 2.286568 | 2.301057 | 2.303568 | 2.017240 | 4.225195 | 0.000000 | 0.000000 | 2.796859 | 3.900153 | 0.000000 | 9.534685 | true |
| D126A | 2.286568 | 2.301057 | 2.303568 | 2.017240 | 4.225195 | 0.000000 | 0.000000 | 2.796859 | 3.856526 | 0.000000 | 9.534685 | true |
| H172A | 2.286568 | 2.301057 | 2.303568 | 2.017240 | 4.225195 | 0.000000 | 0.000000 | — | 2.943899 | 0.000000 | 10.229558 | false |
| Y91F/D126A | 2.286568 | 2.301057 | 2.303568 | 2.017240 | 4.225195 | 0.000000 | 0.000000 | 2.796859 | 4.971855 | 0.000000 | 9.534685 | true |

### Restrained OpenMM minimization

Each variant completed one minimized snapshot. All geometry columns are Å.

| Variant | Zn–scissile O | Zn–H95 NE2 | Zn–H99 NE2 | Zn–E122 O | E96–scissile C | Active RMSD | Substrate RMSD | H172–scissile O | Y91–D126 pocket | Pose drift | Clamp | Complete |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| WT | 1.811736 | 2.600639 | 1.920951 | 1.758690 | 3.026022 | 0.000000 | 0.000000 | 3.214045 | 3.416064 | 0.000000 | 9.507520 | true |
| Y91F | 1.812339 | 2.634235 | 1.922536 | 1.759578 | 3.002165 | 0.413524 | 0.065948 | 3.214255 | 3.835619 | 0.015635 | 9.421329 | true |
| D126A | 1.807553 | 2.708300 | 1.903851 | 1.771613 | 3.021891 | 0.606757 | 0.190120 | 3.211310 | 3.674474 | 0.025023 | 9.512185 | true |
| H172A | 1.826830 | 2.662388 | 1.906166 | 1.759753 | 2.999612 | 0.504605 | 0.056653 | — | 2.663627 | 0.035022 | 10.205205 | false |
| Y91F/D126A | 1.808311 | 2.691381 | 1.906478 | 1.771574 | 2.966587 | 0.600426 | 0.046607 | 3.218296 | 4.567217 | 0.024084 | 9.508168 | true |

**COMPUTATIONAL RESULT** — Initial→final restrained potential energies
(kJ/mol) were WT `6372.410980→-26011.806946`, Y91F
`12185.332657→-25999.432495`, D126A `8788.147850→-26362.137146`, H172A
`8174.978821→-26004.182373`, and Y91F/D126A
`9223.660004→-26429.803162`. These values were produced by the frozen OpenMM
stage but were not a validation-policy metric; systems with different atom
identities are not assigned a cross-variant energy threshold.

### Policy fidelity

**VERIFIED FACT** — The frozen gate uses score ≤1.0 kcal/mol as
`non_regressive`; requires complete geometry; compares five WT-relative
distances with tolerances 0.40/0.35/0.35/0.35/0.50 Å; and applies absolute
limits of 1.0 Å active-site RMSD, 1.0 Å substrate RMSD, and 0.75 Å substrate
pose drift. It applies the same geometry rules to completed minimized evidence.
It does not threshold H172 distance, Y91–D126 pocket distance, clamp distance,
or potential energy.

**COMPUTATIONAL RESULT** — Re-running the frozen `evaluate_validation` function
over the frozen score, static, and minimized CSVs reproduced every
`stability_class`, `geometry_preserved`, `dynamic_geometry_preserved`,
`geometry_gate_reason`, and `gate_outcome` exactly. The only failure reason was
`Y91F_D126A was not separated from non-regressive variants.` No required metric
was omitted, no wrong column was consumed, no sign was reversed, no threshold
was misapplied, and no condition was bypassed.

**COMPUTATIONAL RESULT** — Yes, existing numerical evidence distinguishes the
double from WT and the singles: most visibly, static Y91–D126 pocket distance is
4.971855 Å versus 2.943899/3.900153/3.856526 Å, and minimized pocket distance is
4.567217 Å versus 3.416064/3.835619/3.674474 Å. This is not a valid frozen
separation signal. The metric had no pre-specified gate threshold, both singles
also move it, and H172A—the correctly separated regressive control—moves it in
the opposite direction (2.943899 static, 2.663627 minimized). Promoting it now
would violate the anti-overfitting rule.

**VERIFIED FACT** — H172A was separated coherently and exactly as specified:
its required H172 NE2 atom is absent in both static and minimized structures,
so `geometry_complete` is false. Its ThermoMPNN score, 0.200503, is itself
non-regressive under the frozen 1.0 threshold.

## Question 5 — what the benchmark revealed

### VITA experimental evidence

The primary sources are the committed [VITA article](../references/vita_abeta_metalloprotease.pdf)
and [supplement](../references/vita_abeta_metalloprotease_supplementary.pdf),
especially article Results/Figure 3 and Supplementary Figure S5.

| Variant | kcat (s⁻¹) | Km (µM) | kcat/Km (M⁻¹ s⁻¹) | Change from WT |
| --- | ---: | ---: | ---: | ---: |
| WT | 0.00191 | 5.87 | 325.26 | reference |
| Y91F | 0.00691 ± 0.00040 | 14.57 ± 1.55 | 474.19 | +45.8% |
| D126A | 0.00768 ± 0.00025 | 19.96 ± 1.08 | 384.62 | +18.3% |
| H172A | 0.00339 ± 0.00014 | 14.57 ± 1.07 | 232.60 | -28.5% |
| Y91F/D126A | 0.00300 ± 0.00010 | 18.58 ± 1.08 | 161.46 | -50.4% |

**EXPERIMENTAL REFERENCE** — Y91F and D126A had higher measured catalytic
efficiency than WT; H172A and Y91F/D126A had lower measured catalytic
efficiency. The paper additionally reports that Y91F retained the intended
Aβ42 cleavage site. It does not report a causal molecular mechanism for the
double-mutant regression.

**AUTHOR INTERPRETATION** — The authors describe the singles as enhanced in
catalytic efficiency, driven predominantly by higher turnover, and H172A and
the double as decreased in efficiency, consistent with SDS-PAGE. They describe
the second-sphere behavior as context-dependent and catalytic control as
structurally complex; that is an interpretation, not proof of a particular
epistatic mechanism.

### Computational evidence and observability

**ATLAS COMPUTATIONAL RESULT** — ThermoMPNN classified all three singles as
non-regressive folding-stability changes. ThermoMPNN-D predicted the combined
double as mildly stabilizing (`-0.162948` kcal/mol). The static catalytic
geometry is mostly inherited unchanged from the same WT coordinates; the
deterministic double edit changes the pocket fallback distance but preserves all
gated geometry. Restrained OpenMM minimization likewise preserves all gated
double geometry. H172A is separated only because the mutation removes the
required functional atom.

**INFERENCE** — The experimental double regression is not reasonably observable
with the current Atlas evidence stack. Neither stability model predicts
catalytic activity; the static metrics do not model reaction chemistry or a
kinetic free-energy barrier; and one heavily restrained minimized structure is
not an ensemble or a turnover/affinity calculation. Failure to recover the
double's `kcat/Km` does not demonstrate a computation bug—it demonstrates that
the benchmark asks these observables to validate a phenotype outside their
defined claim boundary.

**COMPUTATIONAL RESULT** — Primary root cause: **D — BENCHMARK
CONSTRUCT-VALIDITY PROBLEM**. Class A is rejected because mutation identity and
parsing are correct. Class B is rejected because the frozen gate was executed
exactly. Class C is less precise than D: the central problem is not merely weak
predictor performance, but the prospective use of stability/local-geometry
observables as a validation gate for catalytic efficiency.

**VERIFIED FACT** — The original `BENCHMARK_FAILED` result remains the valid
outcome of the frozen policy, and candidate generation remains blocked. A
scientifically defensible next step is to acquire the exact active construct
and prospectively pre-register an activity-aligned validation benchmark using
independent controls before any new design run.

**POST-HOC HYPOTHESIS** — Any new catalytic-activity model, unrestrained
ensemble, transition-state calculation, pocket threshold, residue-specific
rule, or revised gate motivated by this known double-mutant label is post-hoc.
It must be developed and validated on independent data and must not be presented
as part of the frozen Atlas v1 benchmark.
