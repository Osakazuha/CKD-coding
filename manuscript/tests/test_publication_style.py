from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pytest

from manuscript.paths import MANUSCRIPT_NAME, SOURCE


FORBIDDEN_PHRASES = [
    "Article type:",
    "Running title:",
    "requirements.txt",
    "results/advanced/",
    "must be confirmed",
    "Competing interests",
    "Author contributions",
    "Supplementary Results Note",
]


def _check_source_text(text: str) -> None:
    assert not re.search(r"[㐀-鿿]", text)
    for phrase in FORBIDDEN_PHRASES:
        assert phrase not in text
    assert "ranked first" not in text
    assert "95% CI" not in text
    assert "ci95" not in text


def test_manuscript_source_avoids_administrative_wording():
    _check_source_text(SOURCE.read_text(encoding="utf-8"))


def test_manuscript_source_states_the_leakage_controls_in_methods():
    text = SOURCE.read_text(encoding="utf-8")
    assert "first standardized" in text and "then imputed" in text
    assert "jointly top-ranked" in text or "jointly ranked" in text
    assert "balanced class weights" in text and "other seven" in text
    assert "grouped" in text
    assert "locked historical" in text
    assert "conditional" in text
    assert "Supplementary Figure S1" in text


@pytest.mark.release_artifact
def test_publication_stage_text_matches_source_and_excludes_administration(publication_stage: Path):
    text = (publication_stage / f"{MANUSCRIPT_NAME}.md").read_text(encoding="utf-8")
    _check_source_text(text)
    manifest = json.loads((publication_stage / "publication_manifest.json").read_text(encoding="utf-8"))
    assert manifest["provenance"]["manuscript_source"] == "manuscript/CKD_Leakage_Safe_Reproduction_Publication_Style.md"


@pytest.mark.release_artifact
def test_publication_docx_uses_real_two_column_section_and_embeds_figures(publication_stage: Path):
    document = publication_stage / f"{MANUSCRIPT_NAME}.docx"
    assert document.is_file() and document.stat().st_size > 20_000
    with zipfile.ZipFile(document) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
        media = [name for name in archive.namelist() if name.startswith("word/media/")]
    assert re.search(r'<w:cols[^>]*w:num="2"', document_xml)
    assert len(media) >= 5


@pytest.mark.release_artifact
def test_publication_pdf_and_vector_figures_exist(publication_stage: Path):
    pdf = publication_stage / f"{MANUSCRIPT_NAME}.pdf"
    assert pdf.is_file() and pdf.stat().st_size > 20_000
    for index in (1, 2, 3, 4, 5, "s1"):
        assert (publication_stage / "publication_figures" / f"figure_{index}.png").stat().st_size > 5_000
        assert (publication_stage / "publication_figures" / f"figure_{index}.pdf").stat().st_size > 1_000


@pytest.mark.release_artifact
def test_figure_five_is_deferred_until_after_principal_findings(publication_stage: Path):
    """Keep the large interpretation figure from stranding space below section 3.7."""
    text = (publication_stage / f"{MANUSCRIPT_NAME}.md").read_text(encoding="utf-8")
    discussion = text.index("## 4. Discussion")
    principal_findings = text.index("### 4.1 Principal findings")
    figure_five = text.index("![Figure 5]")
    comparison = text.index("### 4.2 Comparison with the published study")
    assert discussion < principal_findings < figure_five < comparison


def test_publication_figures_module_reads_artifacts_and_hardcodes_no_performance_values():
    """Figure values must come from result files, never from literals."""
    source = (Path(__file__).resolve().parents[1] / "publication_figures.py").read_text(encoding="utf-8")
    assert not re.search(r"np\.array\(\s*\[\s*[01]\.\d", source)
    assert "read_csv" in source
    for forbidden in ("fit(", "fit_transform(", "PCA(", "StandardScaler(", "load_ckd_data"):
        assert forbidden not in source, forbidden
