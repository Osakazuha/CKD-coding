"""Missingness mechanism and multiple-imputation sensitivity (revision stage).

Three complementary views of the missing data:

1. ``missingness_by_class`` -- for every predictor, the number and share of missing
   observations in CKD and non-CKD records of the analytical dataset, with the ratio
   of the two shares; the number of missing fields per record by class.
2. ``missingness_indicator`` -- classifiers trained on the 24 missingness indicators
   alone (1 = field missing) on the persisted 15 folds of the development sample.
   High discrimination from indicators alone means that *which* measurements were
   ordered is itself diagnostic information, which bears on both the near-perfect AUC
   and on whether predictor measurements precede the diagnosis.
3. ``multiple_imputation`` -- the retained-sample display representatives re-scored
   on the same 15 folds with multiple imputation of the numeric predictors: inside
   each training fold an ``IterativeImputer`` (Bayesian ridge, posterior sampling) is
   fitted ``m`` times with different seeds, the downstream pipeline is fitted on every
   completed training fold, the m predicted probabilities for the validation fold are
   averaged (pooled prediction, 0.5 threshold) and the between-imputation spread of
   the per-imputation metrics is recorded.  Categorical fields keep the within-fold
   mode imputation of the main analysis.  Pooled fold metrics are paired against the
   single-imputation fold scores of the identical pipeline recomputed in this stage.

Nothing here alters the frozen manifest and the holdout is never used.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline

from .advanced_evaluation import ADVANCED_METRICS
from .data_preprocessing import ALL_FEATURES, NUMERIC_FEATURES
from .model_selection import get_display_representative
from .revision_common import (
    PAIRED_METRICS,
    RETAINED_SCOPES,
    RevisionDependencies,
    class_counts,
    describe_fold_metrics,
    key_of,
    outer_splits,
    paired_fold_differences,
    paired_summary,
    score_arrays,
    score_fitted,
)
from .split_registry import FULL_SAMPLE


MISSINGNESS_FILES: Final = (
    "missingness_by_class.csv",
    "missingness_per_record.csv",
    "missingness_indicator_fold_scores.csv",
    "missingness_indicator_summary.csv",
)
MULTIPLE_IMPUTATION_FILES: Final = (
    "mi_imputation_fold_scores.csv",
    "mi_pooled_fold_scores.csv",
    "mi_summary.csv",
)
DEFAULT_IMPUTATIONS: Final = 20
IMPUTER_MAX_ITER: Final = 10
INDICATOR_MODELS: Final = ("Logistic Regression", "Extra Trees")


def missingness_by_class(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Per predictor: missing counts and shares by outcome class."""
    rows = []
    ckd, non = y == 1, y == 0
    for column in ALL_FEATURES:
        missing = X[column].isna()
        share_ckd = float(missing[ckd].mean())
        share_non = float(missing[non].mean())
        rows.append({
            "feature": column,
            "type": "numeric" if column in NUMERIC_FEATURES else "categorical",
            "n_missing_total": int(missing.sum()),
            "share_missing_total": float(missing.mean()),
            "n_missing_ckd": int(missing[ckd].sum()),
            "share_missing_ckd": share_ckd,
            "n_missing_non_ckd": int(missing[non].sum()),
            "share_missing_non_ckd": share_non,
            "missingness_ratio_ckd_vs_non_ckd": float(share_ckd / share_non) if share_non > 0 else float("inf") if share_ckd > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def missingness_per_record(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    counts = X[ALL_FEATURES].isna().sum(axis=1)
    rows = []
    for label, name in ((1, "ckd"), (0, "non_ckd")):
        values = counts[y == label]
        rows.append({
            "class": name, "n_records": int(len(values)),
            "mean_missing_fields": float(values.mean()), "median_missing_fields": float(values.median()),
            "share_complete_records": float((values == 0).mean()),
            "share_with_5_or_more_missing": float((values >= 5).mean()),
        })
    return pd.DataFrame(rows)


def indicator_models(random_state: int) -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000, solver="liblinear", random_state=random_state),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=random_state, n_jobs=-1),
    }


def run_missingness_indicator(
    *,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    full_registry: pd.DataFrame,
    random_state: int,
    results_dir: Path,
    log,
) -> dict[str, pd.DataFrame]:
    """Score indicator-only classifiers on the persisted development folds."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    indicators = X_dev[ALL_FEATURES].isna().astype(int)
    indicators.columns = [f"missing_{column}" for column in ALL_FEATURES]
    rows = []
    for model_name, estimator in indicator_models(random_state).items():
        log.write(f"missingness indicator classifier: {model_name}\n")
        log.flush()
        for split in outer_splits(full_registry, y_dev, population=FULL_SAMPLE):
            train_ids, validation_ids = list(split.train_row_ids), list(split.validation_row_ids)
            fitted = Pipeline([("classifier", clone(estimator))]).fit(indicators.loc[train_ids], y_dev.loc[train_ids])
            values = score_fitted(fitted, indicators.loc[validation_ids], y_dev.loc[validation_ids])
            rows.append({
                "model": model_name, "inputs": "24 missingness indicators only", "cv_scheme": split.cv_scheme,
                "population": split.population, "registry_hash": split.registry_hash,
                "repeat_id": split.repeat_id, "fold_id": split.fold_id,
                "n_train": len(train_ids), "n_validation": len(validation_ids), **values,
            })
    fold_scores = pd.DataFrame(rows)
    fold_scores.to_csv(results_dir / "missingness_indicator_fold_scores.csv", index=False)
    summary = pd.DataFrame([
        {"model": model_name, "inputs": "24 missingness indicators only", **describe_fold_metrics(group, ADVANCED_METRICS, prefix="rcv")}
        for model_name, group in fold_scores.groupby("model", sort=False)
    ])
    summary.to_csv(results_dir / "missingness_indicator_summary.csv", index=False)
    return {"fold_scores": fold_scores, "summary": summary}


def _numeric_imputer(seed: int) -> IterativeImputer:
    return IterativeImputer(
        estimator=BayesianRidge(), sample_posterior=True, max_iter=IMPUTER_MAX_ITER,
        random_state=seed, keep_empty_features=True,
    )


def imputation_seed(random_state: int, imputation_index: int) -> int:
    return int(random_state) * 100 + int(imputation_index)


def _complete_numeric(imputer: IterativeImputer, frame: pd.DataFrame) -> pd.DataFrame:
    completed = frame.copy()
    completed.loc[:, NUMERIC_FEATURES] = imputer.transform(frame.loc[:, NUMERIC_FEATURES].to_numpy(dtype=float))
    return completed


def run_multiple_imputation(
    *,
    deps: RevisionDependencies,
    registered: Mapping[str, object],
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    full_registry: pd.DataFrame,
    main_selection: Mapping[str, Any],
    random_state: int,
    results_dir: Path,
    log,
    n_imputations: int = DEFAULT_IMPUTATIONS,
) -> dict[str, pd.DataFrame]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(n_imputations, bool) or int(n_imputations) < 2:
        raise ValueError("multiple imputation requires at least two imputations")
    n_imputations = int(n_imputations)
    keys = [key_of(get_display_representative(main_selection, scope)) for scope in RETAINED_SCOPES]
    splits = outer_splits(full_registry, y_dev, population=FULL_SAMPLE)
    per_imputation_rows: list[dict[str, object]] = []
    pooled_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for key in keys:
        scenario, imputation, model_name = key
        log.write(f"multiple imputation: {' | '.join(key)} with m={n_imputations}\n")
        log.flush()
        single = deps.fold_score_generator(
            deps.pipeline_builder(scenario, imputation, clone(registered[model_name]), random_state=random_state),
            X_dev, y_dev, full_registry, population=FULL_SAMPLE, scenario=scenario, imputation=imputation,
            model_name=model_name, expected_class_counts=class_counts(y_dev),
        )
        pooled_table = []
        for split in splits:
            train_ids, validation_ids = list(split.train_row_ids), list(split.validation_row_ids)
            X_train, y_train = X_dev.loc[train_ids], y_dev.loc[train_ids]
            X_valid, y_valid = X_dev.loc[validation_ids], y_dev.loc[validation_ids]
            probabilities = np.zeros(len(validation_ids), dtype=float)
            imputation_metrics = []
            for m in range(1, n_imputations + 1):
                imputer = _numeric_imputer(imputation_seed(random_state, m)).fit(X_train.loc[:, NUMERIC_FEATURES].to_numpy(dtype=float))
                completed_train = _complete_numeric(imputer, X_train)
                completed_valid = _complete_numeric(imputer, X_valid)
                pipeline = deps.pipeline_builder(scenario, "median_mode", clone(registered[model_name]), random_state=random_state)
                fitted = clone(pipeline).fit(completed_train, y_train)
                probability = np.asarray(fitted.predict_proba(completed_valid), dtype=float)
                classes = np.asarray(fitted.classes_)
                positive = probability[:, int(np.flatnonzero(classes == 1)[0])]
                prediction = np.asarray(fitted.predict(completed_valid)).astype(int)
                values = score_arrays(y_valid, prediction, positive)
                imputation_metrics.append(values)
                probabilities += positive
                per_imputation_rows.append({
                    "scenario": scenario, "imputation_main": imputation, "model": model_name,
                    "repeat_id": split.repeat_id, "fold_id": split.fold_id, "imputation_index": m,
                    "imputer_random_state": imputation_seed(random_state, m), **values,
                })
            probabilities /= n_imputations
            pooled_prediction = (probabilities >= 0.5).astype(int)
            pooled = score_arrays(y_valid, pooled_prediction, probabilities)
            spread = {
                f"between_imputation_sd_{metric}": float(np.std([v[metric] for v in imputation_metrics], ddof=1))
                for metric in ("accuracy", "roc_auc", "f1", "brier")
            }
            spread.update({
                f"between_imputation_min_{metric}": float(np.min([v[metric] for v in imputation_metrics]))
                for metric in ("accuracy", "roc_auc")
            })
            row = {
                "scenario": scenario, "imputation": imputation, "model": model_name,
                "cv_scheme": split.cv_scheme, "population": split.population, "registry_hash": split.registry_hash,
                "repeat_id": split.repeat_id, "fold_id": split.fold_id,
                "n_train": len(train_ids), "n_validation": len(validation_ids), "n_imputations": n_imputations,
                **pooled, **spread,
            }
            pooled_table.append(row)
        pooled_frame = pd.DataFrame(pooled_table)
        differences = paired_fold_differences(single, pooled_frame, PAIRED_METRICS)
        for column in PAIRED_METRICS:
            pooled_frame[f"single_imputation_{column}"] = differences[f"{column}_reference"].to_numpy()
            pooled_frame[f"delta_{column}_mi_minus_single"] = differences[f"delta_{column}"].to_numpy()
        pooled_rows.extend(pooled_frame.to_dict("records"))
        summary_rows.append({
            "scenario": scenario, "imputation": imputation, "model": model_name, "n_imputations": n_imputations,
            "numeric_imputer": "IterativeImputer(BayesianRidge, sample_posterior=True)",
            "categorical_handling": "within-fold mode imputation (unchanged)",
            **describe_fold_metrics(pooled_frame, ADVANCED_METRICS, prefix="mi_pooled"),
            "mean_between_imputation_sd_accuracy": float(pooled_frame["between_imputation_sd_accuracy"].mean()),
            "mean_between_imputation_sd_roc_auc": float(pooled_frame["between_imputation_sd_roc_auc"].mean()),
            "max_between_imputation_sd_accuracy": float(pooled_frame["between_imputation_sd_accuracy"].max()),
            **paired_summary(differences, PAIRED_METRICS),
        })
    per_imputation = pd.DataFrame(per_imputation_rows)
    per_imputation.to_csv(results_dir / "mi_imputation_fold_scores.csv", index=False)
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(results_dir / "mi_pooled_fold_scores.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(results_dir / "mi_summary.csv", index=False)
    return {"per_imputation": per_imputation, "pooled": pooled, "summary": summary}


def validate_multiple_imputation_outputs(outputs: Mapping[str, Any], *, n_imputations: int, expected_pipelines: int = 3) -> None:
    pooled, summary, per_imputation = outputs["pooled"], outputs["summary"], outputs["per_imputation"]
    if len(summary) != expected_pipelines:
        raise ValueError("multiple imputation must summarize every retained-sample representative")
    if not pooled.groupby(["scenario", "imputation", "model"]).size().eq(15).all():
        raise ValueError("multiple imputation requires 15 pooled folds per representative")
    if not per_imputation.groupby(["scenario", "model", "repeat_id", "fold_id"]).size().eq(n_imputations).all():
        raise ValueError("multiple imputation must score every imputation on every fold")
    values = pooled[[*ADVANCED_METRICS, *[f"delta_{m}_mi_minus_single" for m in PAIRED_METRICS]]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("pooled multiple-imputation scores must be finite")
