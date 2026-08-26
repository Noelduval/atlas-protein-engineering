# Atlas Adaptive System Implementation Plan

> Execute test-first. Commit after each coherent task. Preserve the immutable
> historical benchmark artifacts and the user-authored failure analysis.

**Goal:** Implement and execute the approved LangGraph-coordinated adaptive
DP622 design workflow with at least 5,000 unique evaluated legal variants,
failure-memory-guided repair, replicated explicit-solvent MD, and transparent
portfolio selection.

**Architecture:** A typed Python scientific core owns an append-only SQLite and
JSONL ledger. LangGraph routes identifiers and stage state over the core. The
existing reconstruction, stability, geometry, OpenMM, checkpoint, provenance,
and reporting modules are extended rather than replaced.

**Production environment:** Python 3.10, pandas, NumPy, Biopython, ThermoMPNN,
ThermoMPNN-D, OpenMM 8.2, LangGraph, Google Colab Tesla T4.

## Task 1: Freeze the approved contract and benchmark history

**Files:**
- Add: `docs/plans/2026-08-26-atlas-adaptive-system-design.md`
- Add: `docs/plans/2026-08-26-atlas-adaptive-implementation-plan.md`
- Add: `docs/benchmark_failure_analysis.md`
- Modify: `docs/scientific_decisions.md`

1. Record the approved architecture, evidence boundary, and zero-finalist
   exhaustion invariant.
2. Add the preserved Colab benchmark failure analysis without modifying its
   observed values.
3. Run `git diff --check` and inspect the documentation diff.
4. Commit the design baseline.

## Task 2: Add typed domain models and durable ledger

**Files:**
- Add: `tests/adaptive/test_models.py`
- Add: `tests/adaptive/test_ledger.py`
- Add: `src/atlas/adaptive/__init__.py`
- Add: `src/atlas/adaptive/models.py`
- Add: `src/atlas/adaptive/ledger.py`

1. Write failing tests for normalized mutation sets, stable candidate IDs,
   independent evidence axes, dispositions, lineage, hard violations, and the
   zero-finalist completion audit.
2. Write failing tests for SQLite schema initialization, append-only events,
   candidate uniqueness, evidence/failure/decision persistence, transactions,
   JSONL mirroring, and resume reads.
3. Implement the minimum typed dataclasses/enums and SQLite ledger.
4. Run `pytest tests/adaptive/test_models.py tests/adaptive/test_ledger.py -q`.
5. Commit the domain and ledger layer.

## Task 3: Build the evidence-based design-space classifier

**Files:**
- Add: `tests/design/test_design_space.py`
- Add: `src/atlas/design/design_space.py`
- Add: `src/atlas/design/residue_evidence.py`
- Modify: `src/atlas/structure/reconstruct.py`

1. Write failing tests for the four residue classes, protection of E96 and
   direct Zn ligands, context-sensitive Y91/D126 handling, structural-region
   assignment, and reproducible CSV/JSON export.
2. Implement evidence extraction for Zn coordination, catalytic annotations,
   substrate contacts, burial, packing, secondary structure, scaffold role,
   uncertainty, and known mutations.
3. Generate class decisions with explicit rule/evidence fields.
4. Verify against `data/23WN.cif` and commit.

## Task 4: Implement adaptive generation strategies and 5,000-budget policy

**Files:**
- Add: `tests/design/test_adaptive_generation.py`
- Add: `src/atlas/design/strategies.py`
- Add: `src/atlas/design/adaptive_generator.py`
- Replace: `src/atlas/design/candidate_generator.py`

1. Write failing tests for all six strategies, legal single/double generation,
   full sequences, deterministic IDs, exploration/exploitation quotas,
   duplicate rejection, parentage, and at least 5,000 unique evaluated variants.
2. Implement broad singles, substitution-class expansion, evidence-guided
   combinations, and repair/rescue generators.
3. Make later rounds consult ledger evidence and scoped failure memory.
4. Add deterministic seeds and resume-safe allocation.
5. Run design tests and commit.

## Task 5: Implement screening, Pareto selection, critic, and repair

**Files:**
- Add: `tests/adaptive/test_screening.py`
- Add: `tests/adaptive/test_critic.py`
- Add: `src/atlas/adaptive/screening.py`
- Add: `src/atlas/adaptive/critic.py`
- Add: `src/atlas/adaptive/repair.py`
- Replace: `src/atlas/design/rank_candidates.py`

1. Write failing tests proving hard constraints are never relaxed and soft axes
   are not converted into binary kill switches or a universal scalar score.
2. Implement axis normalization metadata, evidence completeness, Pareto fronts,
   uncertainty, model disagreement, and diversity-aware survivor sampling.
3. Implement repair eligibility and the three-child/two-generation bounds.
4. Persist criticism, repair hypotheses, preservation goals, and evidence deltas.
5. Test a complete candidate-to-repair-child trajectory and commit.

## Task 6: Add LangGraph orchestration without duplicating state

**Files:**
- Add: `tests/orchestration/test_graph.py`
- Add: `src/atlas/orchestration/__init__.py`
- Add: `src/atlas/orchestration/state.py`
- Add: `src/atlas/orchestration/graph.py`
- Add: `src/atlas/orchestration/roles.py`
- Modify: `pyproject.toml`

1. Write failing tests that graph state contains only run/stage/round/ledger IDs,
   routing state, and counters—not candidate/evidence payloads.
2. Test `REJECT`, `REVISE`, and `PROMOTE` routing, repair bounds, failure-memory
   lookup, checkpoint/resume, and terminal exhaustion validation.
3. Add a pinned-compatible LangGraph dependency and implement role nodes whose
   work always persists through the typed core.
4. Run orchestration tests and commit.

## Task 7: Generalize mutant-complex construction and chemistry validation

**Files:**
- Add: `tests/structure/test_mutant_complex.py`
- Add: `src/atlas/structure/mutant_complex.py`
- Add: `src/atlas/structure/chemistry.py`
- Modify: `src/atlas/structure/mutate.py`
- Modify: `pyproject.toml`

1. Write failing tests for arbitrary legal amino-acid substitutions, atom
   provenance, substrate retention, malformed chemistry, clash detection, and
   required Zn coordination.
2. Implement open side-chain rebuilding and restrained relaxation with explicit
   provenance; do not add Rosetta.
3. Reject only hard chemistry failures; persist unavailable evidence otherwise.
4. Run structure and OpenMM regression tests and commit.

## Task 8: Extend explicit-solvent replicated MD

**Files:**
- Add: `tests/dynamics/test_explicit_md.py`
- Add: `src/atlas/dynamics/explicit_md.py`
- Add: `src/atlas/dynamics/ensemble_analysis.py`
- Modify: `src/atlas/dynamics/models.py`
- Modify: `pyproject.toml`

1. Write failing tests for independent seeds, solvation metadata, equilibration,
   production, checkpoints, trajectories, reference/control inclusion, corrupt
   simulation handling, and ensemble summary schema.
2. Implement explicit-solvent preparation and checkpointed replicas with
   documented Zn restraint treatment.
3. Implement geometry distributions, contacts, RMSD/RMSF, substrate drift, and
   replica-agreement summaries.
4. Use short test-mode steps locally and production settings only in Colab.
5. Run dynamics tests and commit.

## Task 9: Audit activity-oriented models

**Files:**
- Add: `tests/research/test_activity_models.py`
- Add: `src/atlas/research/activity_models.py`
- Add: `docs/activity_model_review.md`

1. Define a testable eligibility schema covering prediction target, I/O,
   checkpoints, mutation and metal support, domain, licensing, and reproduction.
2. Research ProMEP, EnzyACT, and current alternatives from primary sources.
3. Record citations and an explicit include/exclude decision without fitting to
   the historical benchmark.
4. Integrate only if eligibility passes; otherwise persist `not_available`.
5. Commit the audit.

## Task 10: Integrate the adaptive production pipeline and CLI

**Files:**
- Add: `tests/integration/test_adaptive_pipeline.py`
- Modify: `src/atlas/pipeline.py`
- Modify: `src/atlas/cli.py`
- Modify: `src/atlas/run_context.py`
- Modify: `src/atlas/colab.py`
- Modify: `notebooks/Atlas_DP622_Colab.ipynb`

1. Write failing integration tests for complete staged execution, stop/resume,
   5,000-budget accounting, funnel counts, immutable benchmark characterization,
   repair artifacts, MD routing, and completion/zero-finalist audits.
2. Add adaptive CLI commands and checkpointed stage boundaries.
3. Update the notebook for generation, screening, structure, MD, adversarial
   review, report, and export stages.
4. Run local integration tests and commit.

## Task 11: Build reports, figures, dossiers, and portfolio selection

**Files:**
- Add: `tests/reporting/test_adaptive_outputs.py`
- Add: `src/atlas/reporting/adaptive_outputs.py`
- Add: `src/atlas/reporting/portfolio.py`
- Modify: `src/atlas/reporting/plots.py`

1. Write failing tests for funnel/round/strategy/region/failure-memory figures,
   machine-readable manifests, near-miss reporting, repair traces, and finalist
   dossier completeness.
2. Implement diversity-aware Pareto portfolio selection for at most five
   complementary hypotheses.
3. Label every finalist `EXPERIMENTALLY UNTESTED` and render arguments for and
   against synthesis, falsifiable hypothesis, assays, and uncertainty.
4. Implement the zero-finalist near-miss report and blocker audit.
5. Run reporting tests and commit.

## Task 12: Execute locally, then in the real Colab T4 runtime

**Files:**
- Modify as required by observed failures only.
- Produce run artifacts in the configured Google Drive checkpoint directory.

1. Run `pytest -q`, static import/compile checks, and a local test-mode adaptive
   workflow.
2. In the existing authenticated Colab notebook, fetch the exact Atlas commit,
   pass preflight, and execute/resume every production stage.
3. Confirm at least 5,000 unique evaluated legal variants and all exhaustion
   invariants before portfolio completion.
4. Execute the real structural funnel and replicated explicit-solvent MD on T4.
5. Download/copy the final reproducibility bundle into the repository run area
   without modifying immutable historical artifacts.
6. Fix reproducible failures test-first and resume from the last valid stage.

## Task 13: Final verification and documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/reproduction.md`
- Modify: `docs/limitations.md`
- Modify: `docs/scientific_decisions.md`
- Add final run artifacts under `runs/<production-run-id>/`.

1. Validate artifact hashes, ledger/event consistency, candidate uniqueness,
   strategy/region coverage, repair bounds, MD replicas, dossier completeness,
   and completion audit.
2. Render and visually inspect report figures and any generated PDF.
3. Run the complete test suite and `git diff --check`.
4. Update the recruiter-readable README and reproducibility instructions with
   actual counts and paths.
5. Commit final artifacts and documentation.
6. Record final git SHA and confirm the worktree is clean.

