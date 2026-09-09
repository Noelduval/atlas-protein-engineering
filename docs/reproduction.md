# Reproduction

## Supported paths

The reviewer path is the GPU Colab notebook. Local CPU execution supports the
structure, geometry, gate logic, tests, and an honest OpenMM attempt, but the
pinned ThermoMPNN-D epistatic implementation requires CUDA.

## Local setup

```bash
git clone https://github.com/Noelduval/atlas-therapeutic-optimization.git Atlas
cd Atlas
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,dynamics]'
python -m pytest -q
```

Structural-only command:

```bash
atlas reconstruct --input data/23WN.cif --output-dir outputs/reconstruction
```

## External stability models

Atlas does not vendor or silently install scientific models. Clone the official
repositories and check out the exact commits recorded in `provenance.json`:

```bash
mkdir -p .external
git clone https://github.com/Kuhlman-Lab/ThermoMPNN.git .external/ThermoMPNN
git -C .external/ThermoMPNN checkout 2b04fd370e399911b1fa5848112cc9013f084110
git clone https://github.com/Kuhlman-Lab/ThermoMPNN-D.git .external/ThermoMPNN-D
git -C .external/ThermoMPNN-D checkout df9a75aaddb674a7c4c193005031fc0536d325fb
python -m pip install omegaconf wandb pytorch-lightning scipy scikit-learn joblib tqdm torchmetrics
```

Both pinned upstream repositories ship `platform.thermompnn_dir` values for the
authors' machines. Their model constructors use that documented local setting
to locate the committed ProteinMPNN weights. The Colab wrapper changes only
that path in each checkout's `local.yaml` to the exact Colab checkout directory;
it does not modify model code, weights, inference arguments, or Git `HEAD`.
`atlas.colab.validate_colab_readiness` verifies both configured paths and all
required weight files before reconstruction begins.

Confirm that the active Python environment sees CUDA before the full run:

```bash
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

Exact full command:

```bash
atlas run --input data/23WN.cif --output-root outputs --atlas-repo . --thermompnn-repo .external/ThermoMPNN --thermompnn-d-repo .external/ThermoMPNN-D --dynamics-mode minimize
```

Use `--dynamics-mode short-md` for restrained 10 ps dynamics or
`--dynamics-mode skip` for an explicitly static-only run. Skipping OpenMM is not
the same as passing dynamic geometry. Every command creates a new UTC-stamped
directory and refuses to overwrite an existing run.

## Colab

1. Open `notebooks/Atlas_DP622_Colab.ipynb` in Google Colab.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Confirm the visible `ATLAS_REF` is the immutable Atlas execution commit
   `aa5f61d8a5fbaffb50e1c2fef3b6b3cc275f707c`, which matches the completed
   Gate 3 provenance records.
4. Run all cells in order. The preflight stops before model work if the runtime,
   checkout, model revisions, input structure, or imports are wrong.
5. Inspect `known_mutation_validation.csv` before interpreting any novel files.
6. Download the generated ZIP from the final cell.

The notebook uses the repository's committed 23WN file and mounts Google Drive
by default at `MyDrive/Atlas/checkpoints`. Stages call the production CLI with a
configuration-derived run ID; resume is accepted only when the stored Atlas,
input, model, dynamics, and validation-policy context matches. Completed raw
model CSVs and scientific outputs survive a Colab runtime restart. Each model
stage runs in its own process, and the candidate stage reuses the genuine
ThermoMPNN exhaustive single-mutant CSV produced before validation. No Rosetta
license, Docker image, premium Colab tier, or separate cloud VM is required.
The host kernel only orchestrates setup and display. Pinned `uv==0.8.13`
creates `/content/atlas-science` from managed Python 3.10 because both pinned
upstream repositories specify Python 3.10 and CUDA 11.7/11.8. PyTorch 2.5.1,
torchvision 0.20.1, and torchaudio 2.5.1 use the official CUDA 11.8 wheel index;
all other direct scientific dependencies are version-pinned in the notebook.
The host interpreter never imports Atlas, and every scientific stage uses
`/content/atlas-science/bin/python`.
Before Stage 3, the notebook executes the installed Atlas imports and CLI, both
upstream inference-script imports, repository/model layout checks, and a real
checkpoint-directory write probe. Every later CLI failure prints its exact
command, working directory, complete stdout/stderr, safe environment context,
and a next action before stopping.

## Expected runtime

| Stage | Typical reviewer-scale estimate |
| --- | --- |
| Environment/model installation | 5–15 min |
| Reconstruction + five static geometries | <1 min |
| ThermoMPNN single-site sweep | 1–5 min on GPU; hardware dependent |
| ThermoMPNN-D epistatic sweep | 1–15 min on GPU; hardware dependent |
| OpenMM minimization attempts | 1–10 min total; may skip during setup |
| Conditional candidate phase | 2–15 min; stability rows reuse the earlier genuine sweep |

Allow roughly 20–60 minutes for a first standard-T4 run. These are operational
estimates, not a measured Atlas benchmark or performance guarantee. `short-md`
can add 10–60 minutes if all systems parameterize successfully.

## Reproducibility records

`provenance.json` records the input SHA-256, Python/platform details, model
commits, dynamics mode, UTC time, and claim boundary. `pipeline_warnings.md`
records missing evidence. Raw external CSVs remain under the run's `stability/`
subdirectories and normalized scores are copied to stable top-level schemas.
`run_context.json` is the resume identity. `run_manifest.json` records the run
ID, absolute checkpoint directory, Atlas/input/model identities, validation
policy, scientific Python executable/version, and installed package versions.
`execution_status.json` separates
`NOT_EVALUATED`, `EXTERNALLY_BLOCKED`, `BENCHMARK FAILED`, and `VALIDATED`
outcomes without treating infrastructure failure as scientific evidence.
