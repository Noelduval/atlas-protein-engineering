# Atlas Adaptive Protein-Design System

**Date:** 2026-08-26
**Status:** Approved

## Purpose

Atlas will become a checkpointed, evidence-led search system for DP622-derived
protein designs. It will use the 23WN-derived DP622/Aβ complex as an
`active_like_inferred` starting model, evaluate at least 5,000 unique legal
variants in adaptive rounds, and reduce them to a transparent portfolio of at
most five experimentally untested wet-lab hypotheses.

Atlas may return no finalists only after a machine-verifiable exhaustion audit
shows that the full candidate budget was evaluated, every approved design
strategy and multiple defensible structural regions were explored, eligible
near-misses received bounded repair attempts, and no remaining candidate can be
promoted without relaxing a hard chemistry or safety constraint. In that case,
the strongest near-miss hypotheses remain first-class report artifacts.

## Scientific boundary

The system supports design hypotheses; it does not predict catalytic turnover,
`kcat/Km`, therapeutic efficacy, plaque dissolution, or experimentally improved
activity. The starting structure is never described as a published exact active
DP622-S2 structure. OpenMM trajectories are structural/dynamic simulations, not
chemical-reaction simulations.

The historical Y91F/D126A `BENCHMARK_FAILED` result is immutable and is labeled
`retrospective_method_characterization`. It limits confidence in stability and
static-geometry evidence but does not block `prospective_design`. Atlas must not
retune a threshold or select a model after seeing that benchmark merely to make
the control pass.

## Architecture

The typed Python core is the scientific source of truth. It owns candidates,
design-space classifications, evidence, decisions, failures, lineages,
checkpoints, artifacts, and provenance in an append-only SQLite ledger with a
mirrored JSONL event stream. CSV and JSON files are exports, never alternate
state stores.

LangGraph is the orchestration layer above the core. Its nodes route work among
real scientific roles and persist only ledger identifiers, run identifiers,
round counters, and routing state. It must not duplicate candidate or evidence
records. The graph supports `REJECT`, `REVISE`, and `PROMOTE` branches,
checkpoint/resume, failure-memory access, and bounded repair loops.

The production environment remains the pinned Google Colab Tesla T4 workflow.
Atlas does not add Docker or Rosetta.

## Design-space definition

Before candidate generation, every DP622 position receives exactly one class:

- `HARD_PROTECTED`: direct Zn ligands, indispensable catalytic machinery, or a
  position whose mutation would invalidate the intended catalytic system.
- `CONTEXT_SENSITIVE`: experimentally or structurally coupled positions, such
  as Y91 and D126, that require conservative/context-aware proposals rather
  than a blanket ban.
- `DESIGNABLE`: positions with a defensible substrate-interface,
  second-shell/preorganization, packing, stability, or conservative-exploration
  hypothesis.
- `OUT_OF_SCOPE`: positions excluded because of missing structural support,
  unresolved mapping, fragment artifacts, or insufficient relevance.

The classification records Zn coordination, catalytic role, Aβ contacts,
solvent accessibility/burial, packing, secondary structure, scaffold role,
structural uncertainty, known mutation evidence, and the rule that produced the
class. Distance to Aβ is supporting evidence, not the sole rule.

## Adaptive search

Atlas evaluates at least 5,000 unique legal variants, primarily singles and
doubles. It does not generate one random library and screen it once.

1. Round 1 performs broad legal single-mutant exploration.
2. Round 2 expands promising positions and substitution classes while retaining
   explicit exploration budget for under-sampled regions and strategies.
3. Round 3 creates evidence-guided doubles from promising or Pareto-nondominated
   singles, with combination-specific risk tracking.
4. Later rounds repair, rescue, or locally optimize promising near-misses.

Every candidate stores an immutable ID, full sequence, normalized mutations,
parent IDs, round, strategy, structural region, hypothesis, intended upside,
expected risk, evidence, criticism, revision lineage, and disposition. Duplicate
sequences and illegal/protected mutations are hard failures and do not count as
evaluated legal variants. Revision children do count.

Approved generation strategies are:

- substrate/interface exploration;
- stability-support exploration;
- second-shell/preorganization exploration;
- conservative exploration;
- evidence-guided combination exploration;
- repair/rescue exploration.

Round allocation enforces exploration versus exploitation so an early region or
strategy cannot consume the full budget.

## Evidence and decisions

Hard constraints are separate from soft evidence. Hard failures include illegal
mapping, protected-residue destruction, duplicate sequence, malformed chemistry,
loss of required Zn coordination, catastrophic clash, and invalid or corrupt
simulation. They are never relaxed.

Soft axes include predicted stability, substrate contacts, catalytic
preorganization, interface preservation, dynamic behavior, uncertainty, model
disagreement, and activity-oriented evidence only when independently validated.
An imperfect soft metric is a tradeoff, not an automatic rejection.

Atlas retains independent axes and uses Pareto dominance, evidence completeness,
uncertainty, and explicit scientific policies. It never hides incomparable axes
inside a universal score. Portfolio selection additionally preserves diversity
of positions, structural regions, amino-acid chemistry, hypotheses, and risk.

The target funnel is approximately 5,000+ evaluated candidates, 500 broad
survivors, 100 deeper mutant-complex analyses, 20 replicated explicit-solvent MD
candidates, 10 adversarially reviewed candidates, and at most five hypotheses.
These are compute targets rather than quotas, and actual counts are reported.

## Failure memory and repair

Failure observations record their scope, evidence count, confidence, and source.
They include recurrent destabilization, clashes, substrate displacement, Zn-site
disruption, combination-specific failures, MD instability, and model
disagreement. Later generators query this memory, but a few observations cannot
be promoted into a biochemical law without adequate support.

A candidate is repair-eligible when it has no hard violation, at least one
meaningful supporting signal, and a specifically diagnosed repairable weakness.
The route is `CRITIC -> REVISE -> RE-EVALUATE`. Each revision records its parent,
weakness, repair hypothesis, change, feature to preserve, and evidence delta.
Default bounds are three children per parent per round and two revision
generations. Stronger bounds require a written ledger justification.

At least one delivered artifact must contain a real, fully evidenced repair
trajectory.

## Structural and dynamics funnel

Promising variants receive open, reproducible side-chain rebuilding, atom-level
model provenance, chemistry validation, and evaluation in the DP622/Aβ complex.
Restrained relaxation is an intermediate screen; the original substrate pose is
not assumed to remain valid.

The final survivor tier receives replicated explicit-solvent OpenMM MD with
independent seeds, documented preparation and equilibration, checkpoint/resume,
persistent trajectories, a reference simulation, and useful controls where
appropriate. Any Zn coordination restraints are explicit in the protocol and
interpretation.

Ensemble readouts include Zn-coordination persistence, substrate drift,
active-site geometry distributions, preorganization metrics, contact occupancy,
local RMSD/RMSF, and inter-replica agreement.

## Activity-oriented models

Research evaluates ProMEP, EnzyACT, and appropriate current alternatives for
target, inputs, outputs, checkpoints, mutation support, structural/metal
treatment, training domain, de novo applicability, licensing, and
reproducibility. No model is integrated unless this audit supports its use for
DP622. An explicit exclusion is a complete and acceptable decision.

## Scientific roles and graph

LangGraph routes these roles over the typed core:

`Research/Evidence -> Hypothesis -> Generation -> Screening -> Structure ->
Critic -> Repair if needed -> Deeper Evaluation -> Simulation -> Activity
Evidence if valid -> Adversarial Critic -> Portfolio Selection`

Each role must persist a real state transition, scientific artifact, evidence,
or decision. The roles are not conversational personas.

## Termination and zero-finalist invariant

Normal completion requires at least 5,000 unique evaluated legal variants and a
completed adversarial portfolio review. A zero-finalist result additionally
requires all of the following:

- the configured adaptive budget is exhausted;
- every approved strategy has generated and evaluated candidates;
- multiple defensible structural regions were explored;
- every eligible near-miss was either repaired within bounds or has a recorded
  reason that repair was infeasible;
- no pending checkpoint, unevaluated legal candidate, or repair branch remains;
- no candidate can be promoted without relaxing a hard constraint;
- strongest near-miss hypotheses are reported with exact promotion blockers.

Scientific caution cannot satisfy this invariant by itself. The system actively
optimizes for useful candidates and treats soft evidence as tradeoffs.

## Delivered artifacts

The run directory contains the design-space table, candidate and lineage tables,
SQLite ledger, JSONL events, failure memory, evidence tables, structures,
trajectories/checkpoints, repair traces, funnel/strategy/region figures,
near-miss report, finalist dossiers, scientific report, reproducibility
manifest, machine-readable completion audit, and immutable historical benchmark
artifacts. Every finalist is labeled `EXPERIMENTALLY UNTESTED`.

