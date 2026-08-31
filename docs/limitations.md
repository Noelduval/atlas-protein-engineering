# Limitations

## Atlas v1 production evidence boundary

Replicated explicit-solvent OpenMM MD was tested on the canonical DP622/Aβ/Zn reference and failed reproducible numerical and catalytic/substrate-geometry validation. It is therefore excluded from adaptive finalist discrimination, evidence completeness, Pareto screening, critic routing, repair, and portfolio selection. Missing dynamics evidence does not penalize candidates. Restrained mutant-complex preparation/minimization remains structure-building evidence, not validated dynamics or catalytic-activity prediction.

ThermoMPNN and ThermoMPNN-D are separate upstream stability models. Their raw outputs are preserved but not treated as commensurate; adaptive screening uses model-specific empirical ranks without claiming calibration between models.

- The starting structure is the inactive DP622 E96Q deposited complex. Q120E is
  an unrelaxed coordinate edit and not an experimentally determined active state.
- Only Aβ residues 34–41 are resolved. Metrics cannot represent the flexibility
  or contacts of full-length Aβ42.
- Heavy-atom mutation construction removes side-chain atoms deterministically;
  it does not repack side chains. Static mutant geometry is therefore a screening
  baseline, not a relaxed structural prediction.
- ThermoMPNN predicts stability change, not proteolytic rate, specificity,
  expression, aggregation, toxicity, delivery, or clinical benefit.
- The published controls are a very small retrospective benchmark. Passing them
  does not demonstrate prospective generalization, and model training-data
  overlap cannot be ruled out.
- ThermoMPNN-D's pinned epistatic script uses CUDA. CPU-only environments cannot
  complete the required double-mutant score.
- The 23WN coordinates are fragments of larger chains and are not chemically
  terminated molecules. Atlas uses OpenMM's documented fragment matching mode
  (`ignoreExternalBonds=True`) and adds hydrogens without inventing heavy atoms,
  caps, residues, or sequence. This permitted a local restrained minimization,
  but it remains an approximation to unresolved external bonding and zinc/
  substrate parameterization. Any future setup failure remains explicit and
  does not count as favorable dynamics evidence.
- Position and zinc restraints make comparisons more interpretable but can hide
  larger conformational changes. Ten-picosecond dynamics is not converged MD.
- The composite ranking is an explicit heuristic rather than a learned catalytic
  model. Lower scores mean higher computational priority only.
- Novel outputs require expression, blinded kinetic assays, cleavage-site and
  selectivity characterization, and independent experimental review.

Reasonable future work includes a validated metal-center force field, terminal
completion reviewed by a structural biologist, replicate longer MD, QM/MM,
prospective mutation controls, optional orthogonal Rosetta refinement, and wet-
lab testing.
