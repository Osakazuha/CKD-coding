"""Evidence ledger: derived from the selection manifest and checked section by section."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from manuscript.paths import MANUSCRIPT_NAME
from manuscript.evidence import (
    LEDGER_FIELDS,
    build_evidence_ledger,
    check_manuscript_against_ledger,
    manuscript_sections,
)
from src.model_selection import get_display_representative


def by_id(rows, claim_id):
    return next(row for row in rows if row["claim_id"] == claim_id)


def test_sections_are_split_by_headings_and_figure_captions():
    text = (
        "# Title\n\nintro\n\n## Abstract\n\nvalue 0.9917 here\n\n## 3. Results\n\n### 3.1 Dataset\n\nn = 400\n\n"
        "![Figure 2](x.png)\n\n*Figure 2. 28 components retained.*\n\n### Table 3. Principal results\n\n| a | 1.0000 |\n"
    )
    sections = manuscript_sections(text)
    assert "0.9917" in sections["Abstract"]
    assert "n = 400" in sections["3.1 Dataset"]
    assert "28 components" in sections["Figure 2."]
    assert "1.0000" in sections["Table 3. Principal results"]
    assert "0.9917" not in sections["3.1 Dataset"]


def test_check_reports_missing_values_and_unknown_sections():
    text = "## Abstract\n\naccuracy 0.9917\n"
    rows = [
        {"claim_id": "ok", "value": "0.9917", "manuscript_section": "Abstract"},
        {"claim_id": "missing", "value": "0.5000", "manuscript_section": "Abstract"},
        {"claim_id": "no_section", "value": "0.9917", "manuscript_section": "3.9 Nowhere"},
    ]
    problems = {row["claim_id"]: row["problem"] for row in check_manuscript_against_ledger(text, rows)}
    assert problems == {"missing": "value not in section", "no_section": "section not found"}


@pytest.mark.release_artifact
def test_ledger_rows_follow_selection_manifest_representatives(release_layout, baseline_stage: Path, advanced_stage: Path):
    import json

    rows = build_evidence_ledger(release_layout.artifact_root)
    assert all(list(row) == LEDGER_FIELDS for row in rows)
    assert len({row["claim_id"] for row in rows}) == len(rows)
    selection = json.loads((advanced_stage / "results" / "selection_manifest.json").read_text(encoding="utf-8"))
    primary = get_display_representative(selection, "retained_sample:C_selected_original")
    assert by_id(rows, "primary_pipeline")["value"] == f"{primary['scenario']}|{primary['imputation']}|{primary['model']}"
    complete = get_display_representative(selection, "complete_case_sensitivity")
    assert by_id(rows, "complete_case_pipeline")["value"] == f"{complete['scenario']}|{complete['imputation']}|{complete['model']}"
    assert by_id(rows, "dataset_n")["value"] == "400"
    assert by_id(rows, "dataset_predictors")["value"] == "24"
    assert by_id(rows, "dataset_ckd")["value"] == "250"
    assert by_id(rows, "dataset_non_ckd")["value"] == "150"
    assert by_id(rows, "shifted_rows_repaired")["value"] == "One record"
    split = json.loads((baseline_stage / "results" / "split_manifest.json").read_text(encoding="utf-8"))
    cc = split["development_complete_case_sensitivity"]["class_counts"]
    assert by_id(rows, "complete_case_development_composition")["value"] == f"{cc['0']} non-CKD and {cc['1']} CKD"
    holdout = pd.read_csv(advanced_stage / "results" / "holdout_evaluation.csv")
    cc_hold = holdout[holdout["imputation"] == "complete_case"].iloc[0]
    assert by_id(rows, "complete_case_holdout_composition")["value"] == f"{int(cc_hold['holdout_non_ckd'])} non-CKD and {int(cc_hold['holdout_ckd'])} CKD"
    assert by_id(rows, "main_seed")["value"] == str(split["random_state"])
    assert by_id(rows, "primary_n_development")["value"] == "280"
    assert by_id(rows, "primary_n_holdout")["value"] == "120"


@pytest.mark.release_artifact
def test_every_ledger_value_appears_in_its_manuscript_section(release_layout, publication_stage: Path):
    rows = build_evidence_ledger(release_layout.artifact_root)
    text = (publication_stage / f"{MANUSCRIPT_NAME}.md").read_text(encoding="utf-8")
    mismatches = check_manuscript_against_ledger(text, rows)
    assert mismatches == []
    with (publication_stage / "evidence_ledger.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == LEDGER_FIELDS
        records = list(reader)
    assert len(records) == len(rows)
    assert records == [{field: row[field] for field in LEDGER_FIELDS} for row in rows]
