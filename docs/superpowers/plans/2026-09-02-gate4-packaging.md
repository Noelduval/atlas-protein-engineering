# Gate 4 Campaign Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the verified Gate-3 export as a durable, reviewable Atlas case study without rerunning scientific computation.

**Architecture:** Copy a curated finalist/provenance subset into `results/gate3/`, add a machine-readable summary and evidence index, and update the README with the observed funnel, methodology, claim boundary, and limitations. Full transient checkpoints remain outside git and are referenced by the manifest.

**Tech Stack:** Markdown, JSON/CSV/PDB/FASTA/PNG artifacts, existing Python test suite.

**Spec:** User-provided Gate-4 packaging request in the attached prompt.

## Global Constraints

- Do not rerun Gate 3, regenerate candidates, rerun ThermoMPNN/ThermoMPNN-D, or change scientific policy.
- Treat the uploaded ZIP as authoritative for exact values.
- Describe all five finalists as computationally prioritized and experimentally untested.
- Do not commit huge transient ledgers, SQLite memory, or non-finalist structures.

### Task 1: Curated result package

**Files:** Create `results/gate3/` and copy verified finalist-facing artifacts, reports, figures, audit, trace, provenance, and limitations.

- [x] Copy the five finalist PDBs and FASTAs, five dossiers, finalist evidence exports, funnel/selection figures, completion audit, run contexts, reproducibility manifest, selection trace, and limitations report.
- [x] Add an artifact README describing the source ZIP and omitted transient files.

### Task 2: Repository narrative

**Files:** Modify `README.md`; create `results/gate3/FINALIST_SUMMARY.md`.

- [x] Document the observed funnel, frozen methodology, provenance, claim boundary, five finalists, and wet-lab next step.
- [x] Keep unsupported activity, therapeutic, and global-novelty claims out of the narrative.

### Task 3: Verification and delivery

- [x] Verify IDs, mutation sets, counts, checksums, links, and absence of unsupported claims.
- [x] Run source and documentation-oriented tests, inspect git diff, commit, and push the existing branch.
