from __future__ import annotations

import json
from pathlib import Path
import stat
import sys

from Bio.PDB import PDBIO, PDBParser, Select

from atlas.late_stage.structure_validation import (
    StructureValidationStatus,
    validate_structure_prediction,
)
from atlas.structure.reconstruct import reconstruct_active_like


SOURCE = Path(__file__).parents[2] / "data" / "23WN.cif"


class _ChainAOnly(Select):
    def accept_chain(self, chain) -> bool:
        return chain.id == "A"


def _reference_inputs(tmp_path: Path) -> tuple[Path, Path, str]:
    reference = tmp_path / "active_like.pdb"
    reconstruct_active_like(SOURCE, reference, tmp_path / "numbering.csv")
    structure = PDBParser(QUIET=True).get_structure("reference", reference)
    sequence = "".join(
        {
            "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
            "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
            "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
            "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        }[residue.resname]
        for residue in structure[0]["A"]
    )
    predicted = tmp_path / "predicted_chain_a.pdb"
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(str(predicted), _ChainAOnly())
    return reference, predicted, sequence


def _write_predictor(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_missing_predictor_is_unavailable_without_a_synthetic_structure(tmp_path: Path) -> None:
    reference, _, sequence = _reference_inputs(tmp_path)
    output = tmp_path / "validation"

    result = validate_structure_prediction(
        candidate_id="ATLAS-MISSING",
        sequence=sequence,
        reference_pdb=reference,
        output_dir=output,
        predictor_command=(str(tmp_path / "does-not-exist"),),
    )

    assert result.status is StructureValidationStatus.UNAVAILABLE
    assert result.predicted_structure is None
    assert result.alignment is None
    assert result.catalytic_site_recovery is None
    report = json.loads((output / "structure_validation.json").read_text())
    assert report["status"] == "unavailable"
    assert report["input"]["sequence"] == sequence
    assert report["tool"]["command"] == [str(tmp_path / "does-not-exist")]
    assert report["tool"]["version"] is None
    assert list(output.glob("*.pdb")) == []


def test_failed_predictor_is_invalid_and_persists_process_evidence(tmp_path: Path) -> None:
    reference, _, sequence = _reference_inputs(tmp_path)
    predictor = _write_predictor(
        tmp_path / "failing_colabfold",
        """import sys
if "--version" in sys.argv:
    print("fake-colabfold 1.0")
    raise SystemExit(0)
print("predictor exploded", file=sys.stderr)
raise SystemExit(17)
""",
    )
    output = tmp_path / "validation"

    result = validate_structure_prediction(
        candidate_id="ATLAS-FAILED",
        sequence=sequence,
        reference_pdb=reference,
        output_dir=output,
        predictor_command=(str(predictor),),
    )

    assert result.status is StructureValidationStatus.INVALID
    assert result.predicted_structure is None
    report = json.loads((output / "structure_validation.json").read_text())
    assert report["process"]["returncode"] == 17
    assert report["tool"]["version"] == "fake-colabfold 1.0"
    assert "predictor exploded" in (output / "predictor.stderr.txt").read_text()


def test_real_command_success_persists_prediction_confidence_and_recovery(
    tmp_path: Path,
) -> None:
    reference, predicted, sequence = _reference_inputs(tmp_path)
    predictor = _write_predictor(
        tmp_path / "working_colabfold",
        f"""import json
from pathlib import Path
import shutil
import sys
if "--version" in sys.argv:
    print("fake-colabfold 2.3")
    raise SystemExit(0)
fasta, output = Path(sys.argv[-2]), Path(sys.argv[-1])
assert fasta.read_text().splitlines()[1] == {sequence!r}
output.mkdir(parents=True, exist_ok=True)
shutil.copyfile({str(predicted)!r}, output / "query_unrelaxed_rank_001_model_1.pdb")
(output / "query_scores_rank_001_model_1.json").write_text(
    json.dumps({{"plddt": [91.0] * {len(sequence)}, "ptm": 0.88}})
)
""",
    )
    output = tmp_path / "validation"

    result = validate_structure_prediction(
        candidate_id="ATLAS-REAL",
        sequence=sequence,
        reference_pdb=reference,
        output_dir=output,
        predictor_command=(sys.executable, str(predictor)),
    )

    assert result.status is StructureValidationStatus.AVAILABLE
    assert result.predicted_structure is not None
    assert result.predicted_structure.is_file()
    assert result.tool_version == "fake-colabfold 2.3"
    assert result.confidence == {"mean_plddt": 91.0, "ptm": 0.88}
    assert result.alignment is not None
    assert result.alignment["aligned_ca_count"] == 215
    assert result.alignment["ca_rmsd_a"] < 1e-5
    assert result.catalytic_site_recovery is not None
    assert result.catalytic_site_recovery["residue_numbers"] == [
        91,
        95,
        96,
        99,
        122,
        126,
        172,
    ]
    assert result.catalytic_site_recovery["matched_sidechain_heavy_atoms"] > 20
    assert result.catalytic_site_recovery["sidechain_heavy_atom_rmsd_a"] < 1e-5
    assert result.catalytic_site_recovery["recovered_by_engineering_threshold"] is True

    report = json.loads((output / "structure_validation.json").read_text())
    assert report["tool"] == {
        "command": [sys.executable, str(predictor)],
        "name": "working_colabfold",
        "version": "fake-colabfold 2.3",
    }
    assert report["input"] == {
        "candidate_id": "ATLAS-REAL",
        "fasta": str(output / "ATLAS-REAL.fasta"),
        "sequence": sequence,
        "sequence_length": 215,
    }
    assert report["claim_limits"]["activity_prediction"] is False
    assert report["claim_limits"]["metal_or_substrate_modeled"] is False
    assert (output / "confidence.json").is_file()
    assert (output / "alignment.json").is_file()
    assert (output / "catalytic_site_recovery.json").is_file()
