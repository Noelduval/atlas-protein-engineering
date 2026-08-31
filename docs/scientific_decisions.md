# Scientific decisions

## Replicated dynamics exclusion

Atlas v1 evaluated replicated explicit-solvent MD as a prospective evidence layer. The canonical DP622/Aβ/Zn reference did not pass reproducible numerical and catalytic/substrate-geometry validation, so replicated MD is excluded from mandatory or optional finalist discrimination. No replacement dynamics layer was introduced.

## Claim boundary

Atlas is a computational prioritization workflow. Four evidence classes remain
separate in code and outputs:

1. **Published experimental data** label the retrospective controls.
2. **Computational reconstruction** describes the Q120E coordinate edit.
3. **Computational validation** describes performance on the known-control gate.
4. **Computationally predicted novel variants** are prospective hypotheses whose
   design evidence remains separate from retrospective method characterization.

None of these imply experimental validation of a generated candidate.

## Structural reconstruction

PDB 23WN is retained byte-for-byte in `data/23WN.cif`. Author chain A residues
25–239 are extracted and mapped by `DP622 = deposited - 24`. The 215-residue
output is checked for contiguity. Deposited Q120 is GLN; its NE2 atom is renamed
OE2 and the residue becomes GLU A96. This preserves deposited coordinates and is
explicitly an isosteric model, not a relaxed or experimentally observed state.

The resolved Aβ chain B segment 34–41 and the single zinc ion are retained.
The deposited zinc connection identifies B38 O as the scissile carbonyl oxygen,
so Atlas uses B38 C/O without inventing a different cleavage assignment.

## Benchmark and stability models

The fixed controls are WT, Y91F, D126A, H172A, and Y91F/D126A. ThermoMPNN is
used for singles and ThermoMPNN-D epistatic mode for the double. WT ΔΔG is a
defined zero reference and is labeled as derived rather than inferred. The
adapters accept only expected official CSV schemas and requested mutation rows.

Predicted ΔΔG ≤ 1.0 kcal/mol is classified as non-regressive for this gate. That
threshold is a stability screen, not an activity threshold and was not fitted to
the four published values.

## Geometry and dynamics

Atom selectors are centralized in `atlas.geometry.selectors`. Required distances
are compared with WT using fixed tolerances: 0.40 Å for Zn–scissile O, 0.35 Å for
zinc ligand distances, and 0.50 Å for reconstructed E96 to the scissile carbonyl.
Aligned active-site RMSD must be ≤1.0 Å, substrate RMSD ≤1.0 Å, and substrate
centroid drift ≤0.75 Å. Loss of a required functional atom is a regression, not
an imputed favorable value.

OpenMM uses standard Amber templates, heavy-atom position restraints, and
explicit harmonic zinc-geometry restraints. Because 23WN contains coordinate
fragments of larger chains, hydrogens are added from OpenMM's built-in residue
definitions and Amber matching uses the documented `ignoreExternalBonds=True`
fragment mode. This does not add caps, heavy atoms, residues, or sequence. A
heavy-atom/residue-count integration test protects that boundary. A failed
system build still produces `skipped_unparameterized_system`, the original
exception, and zero snapshot rows. If real dynamic geometry exists it must also
pass; otherwise the gate transparently falls back to static geometry.

## Historical gate and prospective design

The v1 historical gate required Y91F and D126A to have non-regressive stability
and preserved geometry, while H172A and Y91F/D126A had to be separated by a
stability regression, geometry regression, or required-atom loss. Its genuine
Colab execution produced `BENCHMARK_FAILED`. That result, the original
thresholds, and the underlying artifacts are immutable.

The gate is now classified as `retrospective_method_characterization`, not a
kill-switch for `prospective_design`. The result demonstrates that folding
stability and static/restrained geometry do not validate a catalytic-efficiency
phenotype. It must limit confidence and remain visible in every final report;
it must not prevent an independently documented prospective search.

Prospective design uses the four-class residue map, independent evidence axes,
hard chemistry constraints, Pareto reasoning, adaptive exploration, failure
memory, bounded repair, and a replicated-dynamics funnel defined in
`docs/plans/2026-08-26-atlas-adaptive-system-design.md`. A soft regression is a
tradeoff rather than an automatic rejection. A zero-finalist result is complete
only after the full 5,000-candidate budget, strategy and region coverage, repair
opportunities, and near-miss audit have all been exhausted.

## Engineering choices

The typed Atlas core and append-only scientific ledger are the source of truth.
LangGraph coordinates stage routing, `REJECT / REVISE / PROMOTE` branches,
failure-memory access, checkpoint/resume, and bounded repair without duplicating
scientific records. External models remain pinned, separately licensed
repositories instead of vendored code. Fresh run directories prevent accidental
overwrite. Tests fake only explicit subprocess or compute boundaries; no
production path substitutes invented scientific evidence.
