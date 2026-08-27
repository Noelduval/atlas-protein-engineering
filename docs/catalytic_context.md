# DP622/Aβ catalytic context and claim labels

This note fixes the scientific context used by the prospective Atlas search. It
separates deposited coordinates, experiments, the paper authors' mechanism,
Atlas' working inferences, and unresolved unknowns. None of these categories is
silently promoted into another.

## STRUCTURAL FACT

- PDB 23WN is the deposited DP622-associated complex used by Atlas. The
  accompanying cryo-EM material reports a 2.98 Å reconstruction.
- The deposited enzyme is a pre-catalytic E96Q construct: author residue Q120
  maps to DP622 residue 96. Atlas reconstructs Q120 as E96 and labels the result
  `active_like_inferred`, never `published_exact`.
- The resolved substrate segment is chain B residues 34–41, sequence
  `LMVGGVVI`. The modeled scissile bond is G38|V39.
- The only deposited non-polymer atom is Zn (author chain A, residue 1601). No
  crystallographic/cyo-EM water is present in the coordinate file.
- Direct measurements from the unmodified deposited coordinates give: Zn–B38
  carbonyl O 2.287 Å; Zn–H95 NE2 2.301 Å; Zn–H99 NE2 2.304 Å; and nearest
  Zn–E122 carboxylate O 2.017 Å after conversion to DP622 numbering.
- The resolved eight-residue Aβ segment and a static pre-catalytic model do not
  describe the full substrate ensemble or a chemical reaction path.

Sources: `references/vita_abeta_metalloprotease.pdf`, pp. 7–9 and 15;
`references/vita_abeta_metalloprotease_supplementary.pdf`, pp. 24, 32, and 36;
and direct coordinate inspection of `data/23WN.cif`.

## EXPERIMENTAL RESULT

- Mutation of the catalytic glutamate to alanine abolished measurable activity
  in the reported construct context.
- The reported DP622 Y91F and D126A single variants increased `kcat/Km` in the
  authors' kinetic measurements, whereas H172A and the Y91F/D126A double variant
  decreased it.
- Y91F retained the intended DP622 cleavage-site behavior in the reported
  cleavage-site analysis.
- These results are context-specific. They do not establish that stability
  scores, short simulations, or static geometry predict activity for new DP622
  variants.

Source: `references/vita_abeta_metalloprotease.pdf`, pp. 5 and 7;
`references/vita_abeta_metalloprotease_supplementary.pdf`, pp. 7 and 14.

## AUTHOR INTERPRETATION

The paper authors describe a metalloprotease mechanism in which the scissile
carbonyl oxygen coordinates Zn, the catalytic glutamate acts as a general base
for a zinc-bound water, and tyrosine/histidine interactions support an oxyanion
arrangement. This is an author-supported mechanistic interpretation, not a
reaction trajectory observed in 23WN.

Source: `references/vita_abeta_metalloprotease.pdf`, p. 2.

## ATLAS INFERENCE

- Restoring Q120 to E96 produces a chemically more active-like hypothesis than
  the deposited trapping mutant, but it does not recover an experimentally
  verified exact DP622-S2 assay construct.
- Preserving the Zn ligands H95, H99, and E122, catalytic E96, and H172 is a hard
  chemistry constraint for this search. Y91 and D126 remain context-sensitive
  rather than universally protected because their experimental effects were
  non-additive.
- Static Zn/contact distances, mutant-complex relaxation, and replicated MD are
  independent structural-support axes. They are not activity predictions and
  are never collapsed into a universal Atlas score.
- Mutations outside the protected shell may be useful experiments when they
  preserve the inferred preorganization while improving another independent
  axis. Soft objectives remain tradeoffs, not binary kill switches.

## UNKNOWN

- The exact full sequence and all preparation details of the active DP622-S2
  assay construct are not recoverable from the public materials with
  publication-grade certainty.
- Protonation and tautomer states of E96, H95, H99, H172, and the substrate in a
  reactive state are not experimentally assigned by 23WN.
- The identity, occupancy, and orientation of a catalytic water are not resolved
  in the deposited coordinates.
- Reaction barriers, transition-state stabilization, catalytic turnover,
  specificity across physiological substrates, immunogenicity, and therapeutic
  efficacy remain unknown without further computation and experiment.

## Consequence for Atlas

Atlas uses the experimentally grounded arrangement as a falsifiable design
hypothesis. Hard chemistry constraints are never relaxed. ThermoMPNN,
ThermoMPNN-D, mutant structures, and replicated explicit-solvent MD can support
or weaken a synthesis priority, but only matched wet-lab cleavage and kinetic
experiments can establish activity.
