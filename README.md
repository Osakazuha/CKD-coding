# Leakage-controlled machine-learning reproduction on the UCI Chronic Kidney Disease dataset

Code, persisted outputs and manuscript-rendering scripts for a reproduction and critical re-evaluation of
Islam et al. (2023), *Chronic kidney disease prediction based on machine learning algorithms*, on the public
UCI CKD dataset (400 records, 24 predictors). The analysis re-estimates every learned preprocessing step
(imputation, encoding, scaling, supervised feature selection, PCA) inside each cross-validation training
fold, ranks 81 pipelines with a tie-aware rule, freezes a small set of finalists before the holdout is
touched, evaluates them once on a locked historical holdout, and adds post hoc sensitivity analyses
(nested resampling, missingness, multiple imputation, selection stability, predictor removal, class
weighting, naive-versus-controlled preprocessing).

The released run is `artifacts/run_2026-09-28_seed12/`. Every output file of every stage is listed with
its SHA-256 in that stage's manifest, and each stage records the SHA-256 of the source files that
produced it, so the deposited code can be tied to the deposited results (see *Verifying the released run*).

Manuscript: L. Yang and Y. Zhao, *Reproducibility and leakage-controlled evaluation of machine-learning
models for chronic kidney disease classification* (2026, under review); the full citation will be added on
publication.

## Layout

```text
data/                        UCI CKD workbook (the only input; its SHA-256 is recorded at run time)
src/                         analysis code: preprocessing, feature selection, split registries, fold scoring,
                             tie-aware ranking, the baseline / advanced / revision stages, reports
run_experiment.py            stage 1, baseline: single 5-fold CV of the 81 pipelines (training-CV metrics only)
run_advanced_experiment.py   stage 2, advanced: 5x3 repeated CV, frozen holdout evaluation, bootstrap, calibration,
                             ablation, ordinal-encoding sensitivity, SHAP, seed-sensitivity replications
run_revision_experiment.py   stage 4, revision: post hoc sensitivity analyses on the signed baseline + advanced stages
manuscript/                  publication-style report source (Markdown with {{claim_id}} placeholders), evidence
                             ledger, figure and DOCX/PDF rendering (stage 3, publication), Figure 6 script, tests
tests/                       unit tests and contract tests for a completed run
tools/                       run_main.sh (stages 1-3 + tests) and run_revision.sh (stage 4 + tests)
artifacts/<run-id>/          one immutable root per complete run; only the released run is committed
```

## Installation

Python 3.9 was used for the released run (3.9.6 on macOS, Apple silicon). Exact package versions are in
`requirements.lock.txt`; the runtime versions actually used are recorded in every stage's `config.json`. The
pinned versions have binary wheels for Python 3.9-3.12 (numpy 2.0.2 and numba 0.60.0 have none for 3.13), so
use one of those interpreters for the lock file.

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.lock.txt      # exact versions (reproduction)
# python -m pip install -r requirements.txt         # loose ranges (development)
```

LibreOffice (`soffice` on the PATH) is needed only for the PDF of the publication stage; pass `--skip-pdf` to
`manuscript/render_results.py` to omit it.

## Reproducing the analysis

Each complete run writes to a fresh root `artifacts/<run-id>/`. Stages are written atomically and signed by a
manifest; a signed stage is never overwritten. If any file under `src/` or a `run_*.py` entry point changes,
the whole run has to be repeated under a new root, because every later stage checks the source hashes
recorded by the earlier ones.

```bash
RUN=artifacts/run_$(date +%Y-%m-%d)_seed12
python run_experiment.py          --artifact-root $RUN            # 1. baseline  -> $RUN/baseline/   (--seed N; default 12)
python run_advanced_experiment.py --artifact-root $RUN            # 2. advanced  -> $RUN/advanced/  (--sensitivity-seeds 42 39 55)
python manuscript/render_results.py --artifact-root $RUN --check  # 3a. every ledger value present in its section
python manuscript/render_results.py --artifact-root $RUN          # 3b. figures, ledger, MD/DOCX/PDF -> $RUN/publication/
python -m pytest -q                                                # unit tests
python run_revision_experiment.py --artifact-root $RUN            # 4. revision  -> $RUN/revision/  (about 40 min)
python -m pytest -q --artifact-root $RUN -m release_artifact       # contract tests on the completed run
```

`tools/run_main.sh [RUN_ID]` runs stages 1-3 with the unit and contract tests and writes logs to
`logs/<RUN_ID>/`; `tools/run_revision.sh [RUN_ID]` does the same for stage 4. Both are plain bash scripts; on
macOS they also keep the machine awake and add LibreOffice to the PATH. A shortened non-release check of
stage 4 is available with `run_revision_experiment.py --smoke` (use a scratch root; a written `revision/`
directory is immutable).

If only the manuscript source or the rendering code under `manuscript/` changes, the signed baseline and
advanced stages do not need to be rerun: render a new `publication-vN/` stage
(`tools/run_main.sh <RUN_ID> publication-vN`) and pass `--publication-stage publication-vN` to the contract
tests.

Seeds. The main seed is defined once (`src/data_preprocessing.py: RANDOM_STATE = 12`, overridable with
`run_experiment.py --seed`), written to the baseline `config.json` and `split_manifest.json`, and inherited by
the later stages; the manuscript text and Figure 1 read it from the artifacts. The seed also drives the
mutual-information screening of the selected-feature representation, which is repeated inside every training
fold. The seed-sensitivity replications (42, 39, 55 by default) repeat the partition, both 5x3 registries, the
81-pipeline repeated CV, the tie-aware ranking and a single holdout evaluation of the frozen finalists from
scratch; they are reported (Section 3.8, Supplementary Table S1) but play no role in the main selection.

## What each stage produces

- `baseline/results/`: `config.json` (input and source hashes, runtime versions), `data_audit.json`,
  `split_assignments.csv` + `split_manifest.json` (fixed 70/30 partition), `cv_split_registry.csv`,
  `cv_fold_scores.csv` (81 x 5 folds), `model_comparison.csv` (training CV only), `selection_manifest.json`
  (tie-aware ranking), `reports/reproduction_report.md`.
- `advanced/results/`: `cv_split_registry_full.csv` / `cv_split_registry_complete_case.csv` (5x3 registries for
  the two analytical populations), `repeated_cv_fold_scores.csv` (81 x 15), `model_comparison_repeated_cv.csv`
  (training CV only), `selection_manifest.json` (the single source for every downstream table, figure and
  ledger value), `holdout_evaluation.csv` + `holdout_predictions.csv` (frozen finalists, each evaluated once;
  conditional stratified-bootstrap percentile intervals), `calibration_metrics.csv` + `calibration_bins_quantile.csv`
  + `ece_bins_equal_width.csv`, `pca_explained_variance.csv`, `ablation_results.csv` + `ablation_fold_differences.csv`,
  `ordinal_sensitivity*.csv`, `shap_importance_grouped.csv` + `shap_importance_transformed.csv` +
  `shap_feature_lineage.csv` + `shap_metadata.json`, `seed_sensitivity_summary.csv` + `seed_sensitivity/seed_<n>/`,
  `baseline_integrity.json` (baseline hashes before and after the stage), `figures/`, `reports/advanced_report.md`.
- `publication-v2/`: `publication_figures/figure_{1..5,s1}.{png,pdf}` (drawn from the persisted tables; no
  model is fitted), `evidence_ledger.csv` (source file, row filter and manuscript section of every main-analysis
  value), the rendered publication-style report (`.md`, `.docx`, `.pdf`), `source_hashes.json`. The stage is
  named `publication-v2` because the first rendering of this run was superseded by text-only corrections to the
  report source (author list, two sentences) and is not deposited; the contract tests therefore take
  `--publication-stage publication-v2`.
- `revision/results/`: `config.json` (inherited seeds, parameters), `registry_verification.json` (the
  re-materialised registries match the advanced stage), `missingness/`, `multiple_imputation/`,
  `selection_stability/`, `predictor_removal/`, `class_weighting/`, `naive_comparison/`, `nested/`,
  `figures/`, `reports/revision_report.md`. `class_weighting_policy_check.csv` and
  `controlled_grid_reproduction.csv` show that the stage reproduces the advanced-stage fold scores exactly.

The narrative stage reports (`baseline/reports/reproduction_report.md`, `advanced/reports/advanced_report.md`,
`revision/reports/revision_report.md`) are generated automatically; the baseline report is written in English,
the advanced and revision reports in Chinese. Every table they summarise is also persisted as CSV.

## Manuscript and evidence ledger

`manuscript/CKD_Leakage_Safe_Reproduction_Publication_Style.md` is a publication-style report of the main
analysis (stages 1-3). Its prose is written by hand; every number in it is a `{{claim_id}}` placeholder that
`manuscript/render_results.py` fills from the evidence ledger (`manuscript/evidence.py`), and the manuscript
tests assert that every ledger value appears in the section it belongs to. The submitted journal manuscript
was finalised in a word processor; it contains the same main-analysis values plus the post hoc analyses of
stage 4, whose values were transcribed from the persisted revision tables (Table 5, Supplementary Tables
S2-S8, Section 3.9 and the corresponding discussion). Figure 6 of the manuscript is drawn from `revision/results/predictor_removal/predictor_removal_summary.csv`
by `manuscript/figure6_predictor_removal.py`.

Rank correlations between pipeline rankings (naive versus controlled preprocessing, class-weighting policies)
are Spearman correlations of mean accuracies rounded to 12 decimal places, the quantisation used by the
selection rule, so that pipelines with mathematically equal means receive tied ranks
(`src/revision_common.py: rank_correlation`).

## Design notes

- Numeric k-nearest-neighbour imputation is applied to standardised values (`StandardScaler -> KNNImputer`);
  categorical fields take the training-fold mode.
- The ranking key (ROC area -> F1 -> accuracy) is quantised to 12 decimal places; equal keys are ties
  (`tie_count`), and the display representative of a tied set is chosen by a documented lexicographic order
  and labelled as such.
- Every publication figure reads the persisted CSV/JSON tables; no value is hard-coded.
- SHAP contributions are aggregated per original clinical variable by summing the signed contributions of all
  derived model inputs within each observation and then averaging absolute values. The interpreted model
  (all original fields, k-nearest-neighbour imputation, Extra Trees) is the same configuration as the ablation
  pipeline; ablation refits it within folds, SHAP explains a single development-sample fit on the holdout
  predictors.
- Calibration is displayed with quantile bins (one marker per occupied bin, annotated with its record count,
  plus a probability rug); ECE uses ten equal-width bins with empty bins retained. The two binnings are
  persisted separately.
- Fold-score ranges are the empirical 2.5th-97.5th percentiles of overlapping validation folds and are
  descriptive; the complete-case branch is a different analytical population and is reported separately.
- The feature selector never treats missingness as a category level and uses the same imputation as the
  downstream pipeline.
- `requirements.lock.txt` pins exact versions; `config.json` records the input workbook and source-file
  hashes together with the runtime versions.

## Verifying the released run

```bash
python - <<'EOF'
import hashlib, json, pathlib
root = pathlib.Path("artifacts/run_2026-09-28_seed12")
for stage in ("baseline", "advanced", "revision", "publication-v2"):
    kind = "publication" if stage.startswith("publication") else stage
    manifest = json.loads((root / stage / f"{kind}_manifest.json").read_text())
    bad = [rel for rel, info in manifest["outputs"].items()
           if hashlib.sha256((root / stage / rel).read_bytes()).hexdigest() != info["sha256"]]
    print(stage, "ok" if not bad else bad)
prov = json.loads((root / "revision" / "revision_manifest.json").read_text())["provenance"]
drift = [rel for rel, h in prov["reviewed_source_sha256"].items()
         if hashlib.sha256(pathlib.Path(rel).read_bytes()).hexdigest() != h]
print("source files differing from the signed run:", drift or "none")
EOF
python -m pytest -q --artifact-root artifacts/run_2026-09-28_seed12 --publication-stage publication-v2 -m release_artifact
```

## License and citation

The code and the persisted outputs in this repository are released under the MIT License (see `LICENSE`). The
input workbook under `data/` is the UCI Chronic Kidney Disease dataset (Rubini, Soundarapandian and Eswaran,
UCI Machine Learning Repository, doi:10.24432/C5G020), redistributed under the CC BY 4.0 license stated on its
repository page. If you use this code or these outputs, please cite the manuscript above and the dataset.
