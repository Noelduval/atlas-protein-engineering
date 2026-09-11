import json
import re
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_public_portfolio_metadata_and_result_links_are_consistent():
    citation = (ROOT / "CITATION.cff").read_text()
    pyproject = (ROOT / "pyproject.toml").read_text()
    manifest = json.loads(
        (ROOT / "results/gate3/reproducibility_manifest.json").read_text()
    )
    readme = (ROOT / "README.md").read_text()
    final_report = (ROOT / "results/gate3/atlas_final_design_report.md").read_text()

    assert "https://github.com/Noelduval/atlas-protein-engineering" in citation
    assert 'name = "atlas-protein-engineering"' in pyproject
    assert manifest["package_versions"]["atlas-therapeutic-optimization"] == "1.0.0"
    assert "atlas-therapeutic-optimization" not in citation + pyproject

    finalist_rows = re.findall(
        r"\| \[ATLAS-[A-Z0-9]+\]\((results/gate3/dossiers/[^)]+\.md)\).*?"
        r"\[FASTA\]\((results/gate3/fastas/[^)]+\.fasta)\).*?"
        r"\[PDB\]\((results/gate3/structures/[^)]+\.pdb)\)",
        readme,
    )
    assert len(finalist_rows) == 5
    referenced_paths = {path for row in finalist_rows for path in row}
    referenced_paths.update(
        f"results/gate3/dossiers/{candidate}.md"
        for candidate in re.findall(r"ATLAS-[A-Z0-9]+", final_report)
    )
    assert referenced_paths
    assert all((ROOT / path).is_file() for path in referenced_paths)
    assert "(candidates/" not in final_report
    assert "Each finalist is an explicit 215-aa sequence" in readme

    for dossier in (ROOT / "results/gate3/dossiers").glob("ATLAS-*.md"):
        text = dossier.read_text()
        candidate = dossier.stem
        assert f"[FASTA](../fastas/{candidate}.fasta)" in text
        assert f"[PDB](../structures/{candidate}.pdb)" in text


def test_colab_copy_states_the_reviewer_execution_boundary():
    notebook = json.loads(
        (ROOT / "notebooks/Atlas_DP622_Colab.ipynb").read_text()
    )
    text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )

    assert "completed adaptive DP622 computational campaign" in text
    assert "primary Atlas execution path" in text
    assert "first real" not in text
    assert "sole production authority" not in text
