"""Automatic Markdown report generation for the training-CV-only baseline."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from .model_selection import CANDIDATE_KEY_COLUMNS, get_display_representative


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"report input must be a JSON object: {path}")
    return value


def _scope(manifest: Mapping[str, object], scope_id: str) -> Mapping[str, object]:
    scopes = manifest.get("ranked_scopes")
    if not isinstance(scopes, list):
        raise ValueError("selection manifest ranked_scopes must be a list")
    matches = [
        scope for scope in scopes
        if isinstance(scope, Mapping) and scope.get("scope_id") == scope_id
    ]
    if len(matches) != 1:
        raise ValueError(f"selection scope {scope_id!r} must exist exactly once")
    return matches[0]


def _selected_row(
    comparison: pd.DataFrame,
    manifest: Mapping[str, object],
    scope_id: str,
) -> tuple[pd.Series, int]:
    key = get_display_representative(manifest, scope_id)
    mask = pd.Series(True, index=comparison.index)
    for column in CANDIDATE_KEY_COLUMNS:
        mask &= comparison[column] == key[column]
    matches = comparison.loc[mask]
    if len(matches) != 1:
        raise ValueError(f"selection key for {scope_id!r} must match exactly one comparison row")
    scope = _scope(manifest, scope_id)
    tie_count = scope.get("top_tie_count")
    if isinstance(tie_count, bool) or not isinstance(tie_count, int) or tie_count <= 0:
        raise ValueError(f"selection scope {scope_id!r} has an invalid top_tie_count")
    return matches.iloc[0], tie_count


def _metric(row: pd.Series, name: str) -> str:
    return f"{float(row[f'cv_{name}_mean']):.4f}"


def _interval(row: pd.Series, name: str) -> str:
    low = float(row[f"cv_{name}_descriptive_fold_t_interval_low"])
    high = float(row[f"cv_{name}_descriptive_fold_t_interval_high"])
    return f"[{low:.4f}, {high:.4f}]"


def _retained_table(rows: list[tuple[pd.Series, int]]) -> str:
    body = []
    for row, tie_count in rows:
        body.append(
            f"| {row['scenario']} | {row['imputation']} | {row['model']} | "
            f"{int(row['n_population'])} | {_metric(row, 'accuracy')} | "
            f"{_interval(row, 'accuracy')} | {_metric(row, 'f1')} | "
            f"{_metric(row, 'roc_auc')} | {tie_count} |"
        )
    return "\n".join(body)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _distribution_row(split: Mapping[str, object], key: str, label: str) -> str:
    population = _mapping(split.get(key), f"split {key}")
    counts = _mapping(population.get("class_counts"), f"split {key} class_counts")
    if set(counts) != {"0", "1"}:
        raise ValueError(f"split {key} class_counts must have exact keys 0 and 1")
    non_ckd = _positive_int(counts["0"], f"split {key} non-CKD count")
    ckd = _positive_int(counts["1"], f"split {key} CKD count")
    n_rows = _positive_int(population.get("n_rows"), f"split {key} n_rows")
    if non_ckd + ckd != n_rows:
        raise ValueError(f"split {key} class counts must sum to n_rows")
    return f"| {label} | {n_rows} | {non_ckd} | {ckd} |"


def render_reproduction_report(
    *,
    config: Mapping[str, object],
    audit: Mapping[str, object],
    split: Mapping[str, object],
    comparison: pd.DataFrame,
    selection_manifest: Mapping[str, object],
) -> str:
    """Purely render the report from already validated machine evidence."""
    effective = _mapping(config.get("effective_configuration"), "effective configuration")
    split_config = _mapping(effective.get("split"), "effective split configuration")
    cv_config = _mapping(effective.get("cv"), "effective CV configuration")
    grid = _mapping(effective.get("grid"), "effective grid configuration")
    pca = _mapping(effective.get("pca"), "effective PCA configuration")
    class_weights = _mapping(
        effective.get("class_weight_policy"), "effective class-weight policy"
    )
    random_state = effective.get("random_state")
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise ValueError("effective random_state must be an integer")
    test_size = split_config.get("test_size")
    if isinstance(test_size, bool) or not isinstance(test_size, (int, float)):
        raise ValueError("effective test_size must be numeric")
    test_percent = float(test_size) * 100.0
    if not 0.0 < test_percent < 100.0:
        raise ValueError("effective test_size must be between zero and one")
    development_percent = 100.0 - test_percent
    n_splits = _positive_int(cv_config.get("n_splits"), "effective CV n_splits")
    variance_target = pca.get("variance_target")
    if isinstance(variance_target, bool) or not isinstance(variance_target, (int, float)):
        raise ValueError("PCA variance target must be numeric")
    pca_percent = float(variance_target) * 100.0
    pca_name = pca.get("display_name")
    if not isinstance(pca_name, str) or not pca_name:
        raise ValueError("PCA display name must be a nonblank string")
    scenarios = grid.get("scenarios")
    imputations = grid.get("imputations")
    models = grid.get("models")
    if not all(isinstance(value, list) and value for value in (scenarios, imputations, models)):
        raise ValueError("effective grid domains must be nonempty lists")

    manifest = selection_manifest
    required_keys = set(CANDIDATE_KEY_COLUMNS)
    if comparison.empty or not required_keys.issubset(comparison.columns):
        raise ValueError("comparison table is missing candidate identities")

    retained = [
        _selected_row(comparison, manifest, f"retained_sample:{scenario}")
        for scenario in ("A_all_original", "B_pca_95", "C_selected_original")
    ]
    complete_case = _selected_row(comparison, manifest, "complete_case_sensitivity")
    source = audit.get("source")
    if not isinstance(source, str) or not source:
        raise ValueError("data audit source must be a nonblank project-relative path")
    distributions = "\n".join(
        (
            _distribution_row(split, "development", "Development full sample"),
            _distribution_row(
                split,
                "locked_historical_internal_holdout",
                "Locked historical internal holdout",
            ),
            _distribution_row(
                split,
                "development_complete_case_sensitivity",
                "Development complete-case sensitivity",
            ),
        )
    )

    tied_scopes = [
        f"{row['scenario']} ({tie_count})"
        for row, tie_count in [*retained, complete_case]
        if tie_count > 1
    ]
    tie_sentence = (
        "Tied scopes are " + ", ".join(tied_scopes) + "."
        if tied_scopes
        else "Each displayed scope has one top-ranked configuration."
    )
    _complete_row, complete_ties = complete_case
    split_label = f"{development_percent:g}/{test_percent:g}"
    pca_target_label = f"{pca_percent:g}%"
    weight_text = ", ".join(
        f"{name}={value}" for name, value in class_weights.items()
    )
    return f"""# CKD baseline reproduction report

## 1. Source paper method

The source paper compared machine-learning classifiers for CKD classification using a single institutional dataset. Its reported workflow did not provide the same persisted row-level split registry, fold-level evidence, population separation, or atomic provenance controls used here.

## 2. Actual reproduction method

This run loaded `{source}` and fixed a stratified {split_label} row-ID split with random state {random_state}. Candidate selection uses training cross-validation only across {len(scenarios)} feature scenarios, {len(imputations)} missing-data strategies, and {len(models)} registered models. {n_splits}-fold bounds are descriptive fold-based t intervals, not independent-sample or generalization confidence intervals.

| Analysis population | n | non-CKD | CKD |
|---|---:|---:|---:|
{distributions}

For each retained-sample scenario, the table shows the display representative among configurations jointly top-ranked under the prespecified training-CV rule.

| Scenario | Imputation | Model | Development n | CV accuracy | Accuracy descriptive fold-based t interval | CV F1 | CV ROC-AUC | Top-tie count |
|---|---|---|---:|---:|---:|---:|---:|---:|
{_retained_table(retained)}

### Complete-case sensitivity (different analysis population)

Complete-case deletion changes the evaluated development population and is therefore not pooled with retained-sample comparisons. The display representative below is one of {complete_ties} configurations jointly top-ranked under the prespecified training-CV rule.

| Scenario | Imputation | Model | Complete-case n | CV accuracy | Accuracy descriptive fold-based t interval | CV F1 | CV ROC-AUC | Top-tie count |
|---|---|---|---:|---:|---:|---:|---:|---:|
{_retained_table([complete_case])}

{tie_sentence}

## 3. Differences from the paper

This reproduction locks row identities before learned preprocessing, fits preprocessing and feature selection within each registered fold, records every fold score, distinguishes imputed full-sample analyses from complete-case sensitivity, and applies a prespecified tie-aware selection rule. The persisted class-weight policy is {weight_text}.

## 4. Possible leakage risks and controls used here

The principal risks are learning imputation, scaling, PCA, feature selection, or model choice from held-out rows. Controls are a target-only fixed split, development-only materialization for all candidate work, fold-local pipelines, population-specific registries, persisted fold scores, and a selection manifest that contains no holdout metrics. The baseline stage contains no holdout performance result. The reserved branch is a locked historical internal holdout, not fresh confirmation.

## 5. PCA versus original-feature selection

The retained-sample rows above compare PCA with original-feature representations on the same development population. {pca_name} uses fold-fitted expanded features and retains {pca_target_label} variance; selected-original analysis keeps named clinical variables through fold-local selection. Neither representation is declared statistically superior; the rows are technical display representatives under the shared CV rule.

## 6. Limitations

The dataset is small and single-source, the historical holdout was visible during earlier development, and the {n_splits} validation folds are dependent views of one development sample. Their descriptive fold-based t intervals do not quantify external-population generalization. Complete-case results may reflect missingness-related selection and are descriptive only within their smaller population.

## 7. Next improvements

Freeze downstream finalists from the persisted selection manifest, evaluate the locked historical holdout once in its dedicated stage, add repeated-CV sensitivity and calibration diagnostics, and seek genuinely external prospective validation. No later-stage analysis is performed in this baseline report; the advanced stage owns holdout, calibration, ablation and SHAP results.
"""


def generate_reproduction_report(
    *,
    config_path: Path,
    data_audit_path: Path,
    split_manifest_path: Path,
    comparison_path: Path,
    selection_manifest_path: Path,
    output_path: Path,
) -> None:
    """Render the seven-section baseline report only from persisted machine artifacts."""
    config = _load_object(Path(config_path))
    audit = _load_object(Path(data_audit_path))
    split = _load_object(Path(split_manifest_path))
    manifest = _load_object(Path(selection_manifest_path))
    comparison = pd.read_csv(comparison_path)
    report = render_reproduction_report(
        config=config,
        audit=audit,
        split=split,
        comparison=comparison,
        selection_manifest=manifest,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
