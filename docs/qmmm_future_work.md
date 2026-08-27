# QM/MM future-work boundary

Atlas does not report QM/MM evidence in the current prospective-design mission.
That is a scientific scope decision, not an omitted favorable result.

## Why QM/MM is deferred

The 23WN coordinates are a 2.98 Å pre-catalytic E96Q complex. They do not resolve
a catalytic water and do not assign the reactive protonation/tautomer state.
Restoring E96 supplies a plausible active-like hypothesis, but it does not by
itself define a defensible quantum region, charge state, reactant basin, or
reaction coordinate. A short, unconverged calculation on an arbitrary setup
would create precision without trustworthy evidence.

The approved T4 workflow therefore spends its bounded compute on broader
candidate discovery, genuine stability models, mutant-complex validation, and
replicated explicit-solvent sampling. Those calculations remain structural and
dynamic evidence only.

## Prerequisites for a future QM/MM study

1. Establish matched protonation/tautomer microstates for E96, H95, H99, E122,
   H172, the scissile amide, and candidate catalytic waters.
2. Equilibrate and compare multiple water placements and Zn coordination states
   without using a single hand-built water as ground truth.
3. Declare the QM region before inspecting favorable candidate results. At a
   minimum it should test inclusion of Zn, its three protein ligands, E96,
   catalytic water, and the scissile peptide unit; boundary sensitivity must be
   measured.
4. Use a reaction coordinate capable of describing water activation,
   nucleophilic attack, proton transfer, and C–N cleavage rather than a single
   distance restraint presented as a mechanism.
5. Demonstrate reactant-state equilibration, forward/reverse overlap, replica or
   window convergence, and sensitivity to QM method, basis/pseudopotential,
   embedding, and Zn treatment.
6. Run the same declared protocol on the active-like reference, adverse controls,
   and candidates. Do not interpret candidate-only barriers.

## Admissible future claims

A converged QM/MM free-energy comparison could become an additional independent
mechanistic-support axis with explicit uncertainty. It would still not establish
experimental `kcat`, `Km`, `kcat/Km`, substrate selectivity, or therapeutic
efficacy. Until those prerequisites are met, the activity-oriented axis remains
`UNAVAILABLE` rather than receiving an imputed favorable value.
