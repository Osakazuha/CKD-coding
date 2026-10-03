"""Orchestration for one atomic, training-CV-only CKD baseline stage."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from io import StringIO
import json
import math
from numbers import Integral, Real
from pathlib import Path
import random
import re
from tempfile import TemporaryDirectory
from typing import Final

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from .artifact_layout import (
    ArtifactLayout,
    ArtifactStageError,
    _load_strict_json,
    sha256_file,
)
from .data_preprocessing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    CATEGORY_LEVELS,
    LABEL_MAP,
    MISSING_KEYS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    _compact_key,
    complete_case_subset,
)
from .evaluate_models import (
    BASELINE_FOLD_SCORE_COLUMNS,
    BASELINE_METRICS,
    BASELINE_SUMMARY_COLUMNS,
    summarize_baseline_fold_scores,
)
from .model_selection import build_selection_manifest
from .split_registry import (
    BASELINE_SCHEME,
    COMPLETE_CASE,
    FULL_SAMPLE,
    REGISTRY_COLUMNS,
    compute_registry_hash,
    materialize_baseline_registry,
    read_split_registry_csv,
    validate_split_registry,
    write_split_registry_csv,
)


RELEASE_SCENARIOS: Final = (
    "A_all_original",
    "B_pca_95",
    "C_selected_original",
)
RELEASE_IMPUTATIONS: Final = ("median_mode", "knn", "complete_case")
RELEASE_MODELS: Final = (
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "SVM",
    "KNN",
    "AdaBoost",
    "Gradient Boosting",
    "Extra Trees",
    "XGBoost",
)
BASELINE_SOURCE_ALLOWLIST: Final = (
    "run_experiment.py",
    "src/baseline_experiment.py",
    "src/reproduction_report.py",
    "src/artifact_layout.py",
    "src/split_registry.py",
    "src/data_preprocessing.py",
    "src/evaluate_models.py",
    "src/model_selection.py",
    "src/train_models.py",
    "src/feature_selection.py",
    "src/models/__init__.py",
    "src/models/logistic_regression.py",
    "src/models/decision_tree.py",
    "src/models/random_forest.py",
    "src/models/svm.py",
    "src/models/knn.py",
    "src/models/adaboost.py",
    "src/models/gradient_boosting.py",
    "src/models/extra_trees.py",
    "src/models/xgboost_model.py",
)
RELEASE_OUTPUTS: Final = (
    "results/config.json",
    "results/data_audit.json",
    "results/split_manifest.json",
    "results/split_assignments.csv",
    "results/cv_split_registry.csv",
    "results/model_comparison.csv",
    "results/cv_fold_scores.csv",
    "results/selection_manifest.json",
    "results/figures/model_comparison.png",
    "reports/reproduction_report.md",
    "run.log",
)
_FORBIDDEN_OUTPUT_TOKENS: Final = (
    "test_",
    "holdout",
    "prediction",
    "bootstrap",
    "calibration",
    "confusion",
    "shap",
)
RUNTIME_VERSION_KEYS: Final = (
    "python",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "seaborn",
    "xgboost",
    "openpyxl",
)
MACHINE_OUTPUTS: Final = RELEASE_OUTPUTS[:8]
_AUDIT_KEYS: Final = {
    "source",
    "original_shape",
    "cleaned_shape",
    "shifted_rows_repaired",
    "dropped_extra_columns",
    "numeric_strings_coerced_to_missing",
    "invalid_categories_set_to_missing",
    "invalid_targets_dropped",
    "missing_by_feature",
    "target_distribution",
}


@dataclass(frozen=True)
class ValidatedReleaseEvidence:
    """Strictly reloaded, cross-validated evidence safe for rendering/signing."""

    config: dict[str, object]
    audit: dict[str, object]
    split: dict[str, object]
    assignments: pd.DataFrame
    registry: pd.DataFrame
    fold_scores: pd.DataFrame
    comparison: pd.DataFrame
    selection: dict[str, object]
    registry_hashes: dict[str, str]


@dataclass(frozen=True)
class BaselineDependencies:
    """Injectable slow/scientific boundaries for deterministic runner tests."""

    data_loader: Callable[[Path], tuple[pd.DataFrame, pd.Series, object]]
    model_registry: Callable[[int], Mapping[str, object]]
    pipeline_builder: Callable[..., object]
    fold_score_generator: Callable[..., pd.DataFrame]
    fold_summary: Callable[[pd.DataFrame], pd.DataFrame]
    plotter: Callable[[pd.DataFrame, Path], None]
    report_renderer: Callable[..., None]
    runtime_versions: Callable[[], Mapping[str, str]]


def _default_fold_scores(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    registry: pd.DataFrame,
    *,
    cv_splits: int,
    **kwargs,
) -> pd.DataFrame:
    if cv_splits == 5:
        from .evaluate_models import generate_baseline_fold_scores

        return generate_baseline_fold_scores(pipeline, X, y, registry, **kwargs)
    return _generate_diagnostic_fold_scores(
        pipeline, X, y, registry, cv_splits=cv_splits, **kwargs
    )


def _default_fold_summary(fold_scores: pd.DataFrame) -> pd.DataFrame:
    if set(fold_scores["cv_scheme"]) == {BASELINE_SCHEME}:
        from .evaluate_models import summarize_baseline_fold_scores

        return summarize_baseline_fold_scores(fold_scores)
    return _summarize_diagnostic_fold_scores(fold_scores)


def default_dependencies() -> BaselineDependencies:
    """Load the real scientific boundaries after the runner validates all paths."""
    from importlib import metadata
    import sys

    from .data_preprocessing import load_ckd_data
    from .evaluate_models import plot_model_comparison
    from .reproduction_report import generate_reproduction_report
    from .train_models import build_model_pipeline, get_models

    def versions() -> dict[str, str]:
        result = {"python": sys.version.split()[0]}
        for distribution in (
            "numpy",
            "pandas",
            "scikit-learn",
            "scipy",
            "matplotlib",
            "seaborn",
            "xgboost",
            "openpyxl",
        ):
            result[distribution] = metadata.version(distribution)
        return result

    return BaselineDependencies(
        data_loader=load_ckd_data,
        model_registry=get_models,
        pipeline_builder=build_model_pipeline,
        fold_score_generator=_default_fold_scores,
        fold_summary=_default_fold_summary,
        plotter=plot_model_comparison,
        report_renderer=generate_reproduction_report,
        runtime_versions=versions,
    )


def _json_native(value: object, path: str = "value") -> object:
    if value is None or type(value) in {str, int, bool}:
        return value
    if isinstance(value, np.generic):
        return _json_native(value.item(), path)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _json_native(item, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_native(item, f"{path}[]") for item in value]
    raise TypeError(f"{path} contains a non-JSON-native value: {type(value).__name__}")


def _write_json(path: Path, value: object) -> None:
    native = _json_native(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(native, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@contextmanager
def _run_log(path: Path):
    """Keep every post-open failure in the retained run log and always close it."""
    with path.open("x", encoding="utf-8") as log:
        try:
            yield log
        except Exception as exc:
            log.write(f"error={type(exc).__name__}: {exc}\n")
            log.flush()
            raise


def _source_hashes(layout: ArtifactLayout) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in BASELINE_SOURCE_ALLOWLIST:
        hashes[relative] = sha256_file(layout.project_root, relative)
    return hashes


def _assert_frozen_identity(
    layout: ArtifactLayout,
    *,
    data_relative: str,
    data_hash: str,
    source_hashes: Mapping[str, str],
) -> None:
    """Fail closed when the reviewed input or source tree changed during a run."""
    if sha256_file(layout.project_root, data_relative) != data_hash:
        raise ArtifactStageError("input workbook changed during baseline computation")
    if _source_hashes(layout) != dict(source_hashes):
        raise ArtifactStageError("reviewed baseline source changed during computation")


def _runtime_versions(boundary: Callable[[], Mapping[str, str]]) -> dict[str, str]:
    versions = boundary()
    if not isinstance(versions, Mapping) or not versions:
        raise ValueError("runtime versions must be a nonempty mapping")
    normalized: dict[str, str] = {}
    for package, version in versions.items():
        if not isinstance(package, str) or not package.strip():
            raise ValueError("runtime version package names must be nonblank strings")
        if not isinstance(version, str) or not version.strip():
            raise ValueError("runtime version values must be nonblank strings")
        normalized[package] = version
    if set(normalized) != set(RUNTIME_VERSION_KEYS):
        raise ValueError(
            f"runtime versions require the exact runtime version keys {RUNTIME_VERSION_KEYS}"
        )
    return normalized


def _audit_mapping(audit: object, data_relative: str) -> dict[str, object]:
    if hasattr(audit, "to_dict"):
        raw = audit.to_dict()
    elif is_dataclass(audit):
        raw = asdict(audit)
    elif isinstance(audit, Mapping):
        raw = dict(audit)
    else:
        raise TypeError("loader audit must be a mapping or expose to_dict()")
    raw["source"] = data_relative
    native = _json_native(raw, "data_audit")
    if not isinstance(native, dict):
        raise TypeError("loader audit must serialize to a JSON object")
    return native


def _class_counts(y: pd.Series) -> dict[str, int]:
    return {str(int(key)): int(value) for key, value in y.value_counts().sort_index().items()}


def _typed_row_id(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        token = {"type": "str", "value": value}
    elif not isinstance(value, bool) and isinstance(value, Integral):
        token = {"type": "int", "value": int(value)}
    else:
        raise TypeError("row IDs must be non-boolean integers or strings")
    return json.dumps(token, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_loaded_data(X: pd.DataFrame, y: pd.Series, *, release_eligible: bool) -> None:
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise TypeError("data loader must return a pandas DataFrame and Series")
    if not X.index.is_unique or not y.index.is_unique or not X.index.equals(y.index):
        raise ValueError("loaded features and targets must have exactly matching unique row IDs")
    if X.index.hasnans or y.index.hasnans or y.isna().any():
        raise ValueError("loaded row IDs and targets must not contain missing values")
    targets = [value.item() if isinstance(value, np.generic) else value for value in y]
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in targets
    ):
        raise TypeError("loaded targets must be integer binary outcomes")
    if set(targets) != {0, 1}:
        raise ValueError("loaded targets must be integer binary outcomes with classes 0 and 1")
    if release_eligible and len(y) != 400:
        raise ValueError("a release baseline requires exactly 400 source rows")


def _validate_random_state(value: object) -> int:
    """Return the run seed as a plain int; any seed is legitimate but must be recorded."""
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0 or int(value) > 2**32 - 1:
        raise ValueError("random_state must be an integer between 0 and 2**32 - 1")
    return int(value)


def _split_development(
    X: pd.DataFrame, y: pd.Series, *, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.Series, list[object]]:
    """Fixed stratified 70/30 row-ID split; the seed decides the partition and is recorded."""
    development_ids, holdout_ids = train_test_split(
        list(y.index),
        test_size=0.30,
        random_state=_validate_random_state(random_state),
        stratify=y.to_numpy(),
    )
    return X.loc[development_ids].copy(), y.loc[development_ids].copy(), list(holdout_ids)


def _diagnostic_scheme(cv_splits: int) -> str:
    return f"diagnostic_stratified_{cv_splits}fold"


def _materialize_diagnostic_registry(
    y: pd.Series,
    *,
    population: str,
    cv_splits: int,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    if population not in {FULL_SAMPLE, COMPLETE_CASE}:
        raise ValueError("diagnostic population must be full_sample or complete_case")
    if y.value_counts().min() < cv_splits:
        raise ValueError(f"not enough samples per class for {cv_splits}-fold diagnostic CV")
    rows: list[dict[str, object]] = []
    splitter = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scheme = _diagnostic_scheme(cv_splits)
    for fold_id, (_, validation_positions) in enumerate(splitter.split(y, y), start=1):
        validation = set(int(position) for position in validation_positions)
        for position, (row_id, target) in enumerate(y.items()):
            if isinstance(row_id, np.generic):
                row_id = row_id.item()
            rows.append({
                "cv_scheme": scheme,
                "population": population,
                "repeat_id": 1,
                "fold_id": fold_id,
                "role": "validation" if position in validation else "train",
                "row_id": row_id,
                "target": int(target),
            })
    registry = pd.DataFrame(rows, columns=REGISTRY_COLUMNS[:-1])
    registry["registry_hash"] = compute_registry_hash(registry)
    return registry.loc[:, REGISTRY_COLUMNS]


def _validate_diagnostic_registry(
    registry: pd.DataFrame,
    y: pd.Series,
    *,
    population: str,
    cv_splits: int,
) -> str:
    if list(registry.columns) != REGISTRY_COLUMNS or registry.empty:
        raise ValueError("diagnostic registry must have the exact nonempty registry schema")
    if set(registry["population"]) != {population}:
        raise ValueError("diagnostic registry has the wrong population")
    if set(registry["cv_scheme"]) != {_diagnostic_scheme(cv_splits)}:
        raise ValueError("diagnostic registry has the wrong scheme")
    if set(registry["repeat_id"]) != {1} or set(registry["fold_id"]) != set(range(1, cv_splits + 1)):
        raise ValueError("diagnostic registry has the wrong fold identities")
    if registry.duplicated(["repeat_id", "fold_id", "row_id"]).any():
        raise ValueError("diagnostic registry contains duplicate row assignments")
    known_ids = {_typed_row_id(row_id) for row_id in y.index}
    expected_targets = {
        _typed_row_id(row_id): int(target) for row_id, target in y.items()
    }
    for row_id, target in zip(registry["row_id"], registry["target"]):
        key = _typed_row_id(row_id)
        if key not in known_ids:
            raise ValueError("diagnostic registry contains missing or foreign row IDs")
        if (
            isinstance(target, bool)
            or not isinstance(target, Integral)
            or int(target) != expected_targets[key]
        ):
            raise ValueError("diagnostic registry target does not match the population")
    for fold_id in range(1, cv_splits + 1):
        fold = registry.loc[registry["fold_id"] == fold_id]
        if (
            len(fold) != len(y)
            or {_typed_row_id(value) for value in fold["row_id"]} != known_ids
        ):
            raise ValueError("each diagnostic fold must exhaust its population")
        if set(fold["role"]) != {"train", "validation"}:
            raise ValueError("each diagnostic fold requires train and validation roles")
        validation = fold.loc[fold["role"] == "validation"]
        if validation["target"].value_counts().to_dict() != y.loc[list(validation["row_id"])].value_counts().to_dict():
            raise ValueError("diagnostic registry targets do not match the population")
    if not registry.groupby("row_id")["role"].apply(lambda values: int((values == "validation").sum()) == 1).all():
        raise ValueError("each diagnostic row must be validation exactly once")
    digests = set(registry["registry_hash"])
    if len(digests) != 1 or next(iter(digests)) != compute_registry_hash(registry):
        raise ValueError("diagnostic registry digest does not match its assignments")
    return str(next(iter(digests)))


def _positive_probability(fitted, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.asarray(fitted.predict(X))
    probabilities = np.asarray(fitted.predict_proba(X), dtype=float)
    classes = np.asarray(getattr(fitted, "classes_", []))
    if predictions.shape != (len(X),) or set(predictions.tolist()) - {0, 1}:
        raise ValueError("diagnostic predictions must be one-dimensional binary values")
    if probabilities.shape != (len(X), 2) or classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
        raise ValueError("diagnostic scoring requires binary predict_proba output")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("diagnostic probabilities must be finite values in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("diagnostic probabilities must sum to one")
    positive = int(np.flatnonzero(classes == 1)[0])
    return predictions.astype(int), probabilities[:, positive]


def _generate_diagnostic_fold_scores(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    registry: pd.DataFrame,
    *,
    population: str,
    scenario: str,
    imputation: str,
    model_name: str,
    cv_splits: int,
    **_kwargs,
) -> pd.DataFrame:
    digest = _validate_diagnostic_registry(
        registry, y, population=population, cv_splits=cv_splits
    )
    rows: list[dict[str, object]] = []
    for fold_id in range(1, cv_splits + 1):
        fold = registry.loc[registry["fold_id"] == fold_id]
        train_ids = [row_id for row_id in y.index if row_id in set(fold.loc[fold["role"] == "train", "row_id"])]
        validation_ids = [row_id for row_id in y.index if row_id in set(fold.loc[fold["role"] == "validation", "row_id"])]
        fitted = clone(pipeline).fit(X.loc[train_ids], y.loc[train_ids])
        prediction, probability = _positive_probability(fitted, X.loc[validation_ids])
        y_validation = y.loc[validation_ids]
        tn, fp, _, _ = confusion_matrix(y_validation, prediction, labels=[0, 1]).ravel()
        metrics = {
            "accuracy": float(accuracy_score(y_validation, prediction)),
            "precision": float(precision_score(y_validation, prediction, zero_division=0)),
            "recall": float(recall_score(y_validation, prediction, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "f1": float(f1_score(y_validation, prediction, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_validation, probability)),
            "pr_auc": float(average_precision_score(y_validation, probability)),
        }
        _validate_metric_columns(pd.DataFrame([metrics]), BASELINE_METRICS)
        if scenario == "B_pca_95":
            components = getattr(getattr(fitted, "named_steps", {}).get("pca"), "n_components_", None)
            if isinstance(components, bool) or not isinstance(components, Integral) or int(components) <= 0:
                raise ValueError("diagnostic PCA folds require positive fitted component counts")
            pca_components: int | float = int(components)
        else:
            pca_components = float("nan")
        rows.append({
            "scenario": scenario,
            "imputation": imputation,
            "model": model_name,
            "cv_scheme": _diagnostic_scheme(cv_splits),
            "population": population,
            "registry_hash": digest,
            "repeat_id": 1,
            "fold_id": fold_id,
            "n_train": len(train_ids),
            "n_validation": len(validation_ids),
            "fold_pca_components": pca_components,
            **metrics,
        })
    return pd.DataFrame(rows, columns=BASELINE_FOLD_SCORE_COLUMNS)


def _summarize_diagnostic_fold_scores(fold_scores: pd.DataFrame) -> pd.DataFrame:
    if list(fold_scores.columns) != BASELINE_FOLD_SCORE_COLUMNS or fold_scores.empty:
        raise ValueError("diagnostic fold scores must have the exact baseline fold-score schema")
    if set(fold_scores["cv_scheme"]) == {BASELINE_SCHEME}:
        raise ValueError("fixed baseline scores must use the Task-7 summary boundary")
    _validate_metric_columns(fold_scores, BASELINE_METRICS)
    identity = ["scenario", "imputation", "model", "cv_scheme", "population", "registry_hash"]
    rows: list[dict[str, object]] = []
    for values, group in fold_scores.groupby(identity, sort=True, dropna=False):
        fold_ids = sorted(int(value) for value in group["fold_id"])
        if fold_ids != list(range(1, len(group) + 1)) or set(group["repeat_id"]) != {1}:
            raise ValueError("diagnostic scores must contain one row for each declared fold")
        totals = set((group["n_train"] + group["n_validation"]).astype(int))
        if len(totals) != 1:
            raise ValueError("diagnostic population size must be constant within a candidate")
        row = dict(zip(identity, values))
        row["n_population"] = totals.pop()
        row["n_validation_folds"] = len(group)
        for metric in BASELINE_METRICS:
            metric_values = group[metric].to_numpy(dtype=float)
            mean = float(metric_values.mean())
            std = float(metric_values.std(ddof=1))
            half = float(t.ppf(0.975, len(metric_values) - 1) * std / np.sqrt(len(metric_values)))
            row[f"cv_{metric}_mean"] = mean
            row[f"cv_{metric}_std"] = std
            row[f"cv_{metric}_descriptive_fold_t_interval_low"] = max(0.0, mean - half)
            row[f"cv_{metric}_descriptive_fold_t_interval_high"] = min(1.0, mean + half)
        rows.append(row)
    return pd.DataFrame(rows, columns=BASELINE_SUMMARY_COLUMNS).sort_values(
        identity, kind="stable", ignore_index=True
    )


def _write_diagnostic_report(config: Mapping[str, object], comparison: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "# NON-RELEASE baseline diagnostic\n\n"
        "This scratch run is not a completed baseline release and cannot be promoted.\n\n"
        f"- Reasons: {', '.join(str(item) for item in config['diagnostic_reasons'])}\n"
        f"- Candidate rows: {len(comparison)}\n"
        f"- CV folds per candidate: {config['effective_configuration']['cv']['n_splits']}\n"
        "- No holdout performance was computed.\n",
        encoding="utf-8",
    )


def _persist_split(
    *,
    y: pd.Series,
    development_y: pd.Series,
    complete_case_y: pd.Series,
    holdout_ids: Sequence[object],
    results_dir: Path,
    random_state: int,
) -> None:
    development_ids = set(development_y.index)
    holdout_id_set = set(holdout_ids)
    if development_ids & holdout_id_set or development_ids | holdout_id_set != set(y.index):
        raise ValueError("fixed split must be an exhaustive disjoint row-ID partition")
    assignments = pd.DataFrame(
        {
            "row_id": [_typed_row_id(row_id) for row_id in y.index],
            "target": [int(y.loc[row_id]) for row_id in y.index],
            "assignment": [
                "development" if row_id in development_ids else "locked_historical_internal_holdout"
                for row_id in y.index
            ],
        }
    )
    assignments.to_csv(results_dir / "split_assignments.csv", index=False)

    def assigned_targets(assignment: str | None = None) -> pd.Series:
        if assignment is None:
            return assignments["target"]
        return assignments.loc[assignments["assignment"] == assignment, "target"]

    source_targets = assigned_targets()
    development_targets = assigned_targets("development")
    holdout_targets = assigned_targets("locked_historical_internal_holdout")
    split_manifest = {
        "schema_version": "baseline_split_v1",
        "random_state": random_state,
        "test_size": 0.30,
        "stratification": "target_only",
        "source_population": {
            "n_rows": len(source_targets),
            "class_counts": _class_counts(source_targets),
        },
        "development": {
            "n_rows": len(development_targets),
            "class_counts": _class_counts(development_targets),
        },
        "development_complete_case_sensitivity": {
            "n_rows": len(complete_case_y),
            "class_counts": _class_counts(complete_case_y),
        },
        "locked_historical_internal_holdout": {
            "n_rows": len(holdout_targets),
            "class_counts": _class_counts(holdout_targets),
            "status": "locked_historical_internal_holdout_not_evaluated",
        },
        "assignments_file": "results/split_assignments.csv",
    }
    _write_json(results_dir / "split_manifest.json", split_manifest)


def _diagnostic_reasons(*, filtered: bool, cv_splits: int) -> list[str]:
    return [
        reason
        for condition, reason in (
            (filtered, "model_filter_present"),
            (cv_splits != 5, "cv_splits_changed"),
        )
        if condition
    ]


def _config(
    *,
    release_eligible: bool,
    data_relative: str,
    data_hash: str,
    source_hashes: Mapping[str, str],
    runtime_versions: Mapping[str, str],
    cv_splits: int,
    model_names: Sequence[str],
    filtered: bool,
    random_state: int,
) -> dict[str, object]:
    diagnostic_reasons = _diagnostic_reasons(filtered=filtered, cv_splits=cv_splits)
    return {
        "schema_version": "baseline_config_v1",
        "stage": "baseline",
        "run_kind": "release" if release_eligible else "diagnostic",
        "release_eligible": release_eligible,
        "diagnostic_reasons": [] if release_eligible else diagnostic_reasons,
        "input": {"path": data_relative, "sha256": data_hash},
        "reviewed_source_sha256": dict(source_hashes),
        "runtime_versions": dict(runtime_versions),
        "effective_configuration": {
            "random_state": random_state,
            "split": {
                "test_size": 0.30,
                "random_state": random_state,
                "stratification": "target_only",
            },
            "cv": {
                "scheme": BASELINE_SCHEME if cv_splits == 5 else f"diagnostic_stratified_{cv_splits}fold",
                "n_splits": cv_splits,
                "shuffle": True,
                "random_state": random_state,
            },
            "grid": {
                "scenarios": list(RELEASE_SCENARIOS),
                "imputations": list(RELEASE_IMPUTATIONS),
                "models": list(model_names),
                "candidate_count": len(RELEASE_SCENARIOS) * len(RELEASE_IMPUTATIONS) * len(model_names),
                "model_filter_present": filtered,
            },
            "selection": {
                "basis": "training_cross_validation_only",
                "selection_uses_holdout": False,
                "metric_order": ["roc_auc", "f1", "accuracy"],
                "quantization_decimal_places": 12,
                "tie_label": "jointly_top_ranked",
                "technical_example_label": "display_representative",
            },
            "pca": {
                "representation": "level_weighted_expanded_feature_pca",
                "display_name": "Level-weighted expanded-feature PCA",
                "input_basis": "full_nominal_one_hot_blocks",
                "feature_weighting": "unit_variance_per_expanded_feature",
                "variance_target": 0.95,
                "svd_solver": "full",
            },
            "class_weight_policy": {
                "Random Forest": "balanced",
                "Extra Trees": "balanced",
                "other_models": "registered_factory_defaults",
            },
        },
    }


def _forbidden_columns(frame: pd.DataFrame) -> list[str]:
    return [
        str(column) for column in frame.columns
        if any(token in str(column).lower() for token in _FORBIDDEN_OUTPUT_TOKENS)
    ]


def _validate_metric_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        raw = frame[column].to_numpy()
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in raw):
            raise ValueError(f"{column} must contain finite real values in [0, 1]")
        values = np.asarray(raw, dtype=float)
        if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
            raise ValueError(f"{column} must contain finite real values in [0, 1]")


def _validate_release_tables(
    comparison: pd.DataFrame,
    fold_scores: pd.DataFrame,
    registry_hashes: Mapping[str, str],
) -> None:
    if list(comparison.columns) != BASELINE_SUMMARY_COLUMNS:
        raise ValueError("release comparison must have the exact Task-7 baseline summary schema")
    if list(fold_scores.columns) != BASELINE_FOLD_SCORE_COLUMNS:
        raise ValueError("release fold scores must have the exact Task-7 baseline score schema")
    if _forbidden_columns(comparison) or _forbidden_columns(fold_scores):
        raise ValueError("baseline tables contain holdout or post-baseline columns")
    keys = ["scenario", "imputation", "model"]
    expected_keys = {
        (scenario, imputation, model)
        for scenario in RELEASE_SCENARIOS
        for imputation in RELEASE_IMPUTATIONS
        for model in RELEASE_MODELS
    }
    observed_keys = set(comparison[keys].itertuples(index=False, name=None))
    if len(comparison) != 81 or observed_keys != expected_keys:
        raise ValueError("release comparison requires exactly 81 unique declared candidates")
    if int((comparison["model"] == "XGBoost").sum()) != 9:
        raise ValueError("release comparison requires exactly nine XGBoost rows")
    fold_keys = list(fold_scores[keys].itertuples(index=False, name=None))
    if len(fold_scores) != 405 or set(fold_keys) != expected_keys:
        raise ValueError("release fold scores require exactly 405 declared candidate-fold rows")
    if not fold_scores.groupby(keys, sort=False).size().eq(5).all():
        raise ValueError("each release candidate requires five validation-fold rows")
    if set(fold_scores["fold_id"]) != {1, 2, 3, 4, 5} or set(fold_scores["repeat_id"]) != {1}:
        raise ValueError("release fold identities must be exactly one repeat of five folds")
    expected_population = comparison["imputation"].map(
        lambda value: COMPLETE_CASE if value == "complete_case" else FULL_SAMPLE
    )
    if not comparison["population"].equals(expected_population):
        raise ValueError("release candidate population does not match its imputation")
    for population, digest in registry_hashes.items():
        candidate_digests = set(comparison.loc[comparison["population"] == population, "registry_hash"])
        fold_digests = set(fold_scores.loc[fold_scores["population"] == population, "registry_hash"])
        if candidate_digests != {digest} or fold_digests != {digest}:
            raise ValueError("release candidates do not use the validated population registry")
    _validate_metric_columns(fold_scores, BASELINE_METRICS)
    summary_metrics = [
        column for column in comparison.columns
        if column.startswith("cv_") and column not in {"cv_scheme"}
    ]
    _validate_metric_columns(comparison, summary_metrics)


def _validate_diagnostic_tables(
    comparison: pd.DataFrame,
    fold_scores: pd.DataFrame,
    *,
    candidate_count: int,
    cv_splits: int,
) -> None:
    keys = ["scenario", "imputation", "model"]
    if comparison[keys].drop_duplicates().shape[0] != candidate_count or len(comparison) != candidate_count:
        raise ValueError("diagnostic comparison must contain one row per requested candidate")
    if len(fold_scores) != candidate_count * cv_splits:
        raise ValueError("diagnostic scores must contain the configured folds for every candidate")
    if not fold_scores.groupby(keys, sort=False).size().eq(cv_splits).all():
        raise ValueError("diagnostic candidate fold counts do not match the configured CV")
    if _forbidden_columns(comparison) or _forbidden_columns(fold_scores):
        raise ValueError("diagnostic tables contain holdout or post-baseline columns")
    _validate_metric_columns(fold_scores, BASELINE_METRICS)
    _validate_metric_columns(
        comparison,
        [
            column for column in comparison.columns
            if column.startswith("cv_") and column != "cv_scheme"
        ],
    )


def _validate_exact_release_outputs(root: Path) -> None:
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    expected = set(RELEASE_OUTPUTS)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ArtifactStageError(f"release output contract mismatch; missing={missing}, extra={extra}")
    for relative in RELEASE_OUTPUTS:
        path = root / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise ArtifactStageError(f"release output must be a nonempty regular file: {relative}")


def _expected_release_log() -> str:
    lines = ["baseline experiment started", "mode=release"]
    counter = 0
    total = len(RELEASE_SCENARIOS) * len(RELEASE_IMPUTATIONS) * len(RELEASE_MODELS)
    for imputation in RELEASE_IMPUTATIONS:
        for scenario in RELEASE_SCENARIOS:
            for model_name in RELEASE_MODELS:
                counter += 1
                lines.append(
                    f"candidate {counter}/{total}: {scenario} | {imputation} | {model_name}"
                )
    lines.append("outputs validated; ready for atomic completion")
    return "\n".join(lines) + "\n"


def _require_matching_sha256(
    root: Path,
    relative: str,
    expected: object,
    name: str,
) -> None:
    if (
        type(expected) is not str
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError(f"{name} requires a captured lowercase SHA-256 digest")
    if sha256_file(root, relative) != expected:
        raise ValueError(f"{name} changed after its render boundary")


def _decode_rendered_png(path: Path) -> np.ndarray:
    if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("rendered comparison figure must be a PNG image")
    try:
        from matplotlib import image as matplotlib_image

        pixels = np.asarray(matplotlib_image.imread(path))
    except Exception as exc:
        raise ValueError("rendered comparison figure must be a decodable PNG image") from exc
    if pixels.ndim not in {2, 3} or pixels.size == 0 or min(pixels.shape[:2]) <= 0:
        raise ValueError("rendered comparison PNG must have nonzero image dimensions")
    return pixels


def _validate_rendered_png(path: Path) -> None:
    _decode_rendered_png(path)


def _validate_release_figure_binding(
    comparison: pd.DataFrame,
    rendered_path: Path,
) -> None:
    """Require release pixels to equal an independent reviewed-renderer result."""
    try:
        from .evaluate_models import plot_model_comparison

        with TemporaryDirectory(
            prefix="ckd-baseline-canonical-render-",
            dir=rendered_path.parent,
        ) as directory:
            reference_path = Path(directory) / "model_comparison.png"
            plot_model_comparison(comparison.copy(deep=True), reference_path)
            rendered_pixels = _decode_rendered_png(rendered_path)
            reference_pixels = _decode_rendered_png(reference_path)
            if not np.array_equal(rendered_pixels, reference_pixels):
                raise ValueError(
                    "rendered comparison heatmap pixels do not match the canonical "
                    "persisted-comparison render"
                )
    except Exception as exc:
        raise ArtifactStageError(
            f"release figure semantic validation failed: {exc}"
        ) from exc


def _require_exact_keys(value: object, expected: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} must have the exact schema {sorted(expected)}")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _class_count_mapping(value: object, name: str) -> dict[int, int]:
    mapping = _require_exact_keys(value, {"0", "1"}, name)
    result = {
        0: _nonnegative_int(mapping["0"], f"{name}.0"),
        1: _nonnegative_int(mapping["1"], f"{name}.1"),
    }
    if min(result.values()) <= 0:
        raise ValueError(f"{name} must contain both observed classes")
    return result


def _decode_typed_row_id(token: object) -> object:
    if not isinstance(token, str):
        raise TypeError("split assignment row_id tokens must be text")

    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("split assignment row_id token contains duplicate keys")
            value[key] = item
        return value

    try:
        decoded = json.loads(token, object_pairs_hook=no_duplicates)
    except json.JSONDecodeError as exc:
        raise ValueError("split assignment row_id token must be valid JSON") from exc
    if not isinstance(decoded, dict) or set(decoded) != {"type", "value"}:
        raise ValueError("split assignment row_id token has an invalid schema")
    if decoded["type"] == "str" and isinstance(decoded["value"], str):
        return decoded["value"]
    if (
        decoded["type"] == "int"
        and not isinstance(decoded["value"], bool)
        and isinstance(decoded["value"], Integral)
    ):
        return int(decoded["value"])
    raise ValueError("split assignment row_id token has an invalid type/value pair")


def _validate_audit(
    audit: object,
    *,
    data_relative: str,
    source_counts: Mapping[int, int],
) -> dict[str, object]:
    audit = _require_exact_keys(audit, _AUDIT_KEYS, "data audit")
    if audit["source"] != data_relative:
        raise ValueError("data audit source must match the signed config input path")
    for name in ("original_shape", "cleaned_shape"):
        shape = audit[name]
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) <= 0
                for value in shape
            )
        ):
            raise ValueError(f"data audit {name} must be two positive integers")
    original_rows, original_columns = (int(value) for value in audit["original_shape"])
    cleaned_rows, cleaned_columns = (int(value) for value in audit["cleaned_shape"])
    expected_cleaned_columns = len(ALL_FEATURES) + 1
    if [cleaned_rows, cleaned_columns] != [400, expected_cleaned_columns]:
        raise ValueError("data audit cleaned_shape must match the release population")
    if original_rows < cleaned_rows:
        raise ValueError("data audit original_shape cannot be smaller than cleaned_shape")
    if _nonnegative_int(audit["shifted_rows_repaired"], "shifted_rows_repaired") > original_rows:
        raise ValueError("shifted_rows_repaired cannot exceed the original population")

    dropped = audit["dropped_extra_columns"]
    if (
        not isinstance(dropped, list)
        or any(type(value) is not str or not value.strip() for value in dropped)
        or len(dropped) != len(set(dropped))
    ):
        raise ValueError("dropped_extra_columns must contain unique nonblank strings")
    required_columns = set(ALL_FEATURES) | {TARGET_COLUMN}
    if any(value.strip() in required_columns for value in dropped):
        raise ValueError("dropped_extra_columns cannot contain a required CKD column")
    if original_columns != cleaned_columns + len(dropped):
        raise ValueError("data audit original columns must equal cleaned columns plus dropped extras")

    numeric = _require_exact_keys(
        audit["numeric_strings_coerced_to_missing"],
        set(NUMERIC_FEATURES),
        "numeric_strings_coerced_to_missing",
    )
    missing = _require_exact_keys(
        audit["missing_by_feature"], set(ALL_FEATURES), "missing_by_feature"
    )
    audited_counts: dict[str, dict[str, int]] = {}
    for name, mapping in (("numeric_strings_coerced_to_missing", numeric), ("missing_by_feature", missing)):
        upper_bound = original_rows if name == "numeric_strings_coerced_to_missing" else cleaned_rows
        audited_counts[name] = {}
        for key, value in mapping.items():
            count = _nonnegative_int(value, f"{name}.{key}")
            audited_counts[name][key] = count
            if count > upper_bound:
                raise ValueError(f"{name}.{key} cannot exceed its audited population")
    dropped_rows = original_rows - cleaned_rows
    for feature, count in audited_counts["numeric_strings_coerced_to_missing"].items():
        if count > audited_counts["missing_by_feature"][feature] + dropped_rows:
            raise ValueError(
                f"numeric_strings_coerced_to_missing.{feature} cannot reconcile "
                "with cleaned missingness and dropped rows"
            )

    invalid_categories = audit["invalid_categories_set_to_missing"]
    if not isinstance(invalid_categories, dict):
        raise ValueError("data audit invalid_categories_set_to_missing must be a mapping")
    if not set(invalid_categories).issubset(CATEGORICAL_FEATURES):
        raise ValueError("invalid category audit keys must be categorical CKD features")
    for feature, invalid_values in invalid_categories.items():
        if not isinstance(invalid_values, dict) or not invalid_values:
            raise ValueError(
                f"invalid category audit for {feature} must be a nonempty nested mapping"
            )
        feature_total = 0
        for label, count in invalid_values.items():
            if type(label) is not str or not label.strip():
                raise ValueError(f"invalid category labels for {feature} must be nonblank strings")
            canonical_label = _compact_key(label)
            if feature in {"sg", "al", "su"} and re.fullmatch(
                r"[-+]?\d*\.?\d+", canonical_label
            ):
                canonical_label = format(float(canonical_label), "g")
            if label != canonical_label:
                raise ValueError(
                    f"invalid category audit label {label!r} for {feature} is not canonical"
                )
            if canonical_label in MISSING_KEYS:
                raise ValueError(
                    f"invalid category audit for {feature} contains missing marker {label!r}"
                )
            if canonical_label in CATEGORY_LEVELS[feature]:
                raise ValueError(
                    f"invalid category audit for {feature} contains canonical valid label {label!r}"
                )
            count = _nonnegative_int(count, f"invalid_categories_set_to_missing.{feature}.{label}")
            if count == 0:
                raise ValueError("invalid category audit counts must be positive")
            feature_total += count
        if feature_total > original_rows:
            raise ValueError(f"invalid category counts for {feature} exceed the original population")
        if feature_total > audited_counts["missing_by_feature"][feature] + dropped_rows:
            raise ValueError(
                f"invalid category counts for {feature} cannot reconcile with cleaned "
                "missingness and dropped rows"
            )

    invalid_targets = audit["invalid_targets_dropped"]
    if not isinstance(invalid_targets, dict):
        raise ValueError("data audit invalid_targets_dropped must be a mapping")
    invalid_target_total = 0
    for label, count in invalid_targets.items():
        if type(label) is not str or not label.strip():
            raise ValueError("invalid target labels must be nonblank strings")
        canonical_label = _compact_key(label)
        if label != canonical_label:
            raise ValueError(f"invalid target audit label {label!r} is not canonical")
        if canonical_label in MISSING_KEYS:
            raise ValueError(f"invalid target audit contains missing marker {label!r}")
        if canonical_label in LABEL_MAP:
            raise ValueError(
                f"invalid target audit contains canonical valid label {label!r}"
            )
        count = _nonnegative_int(count, f"invalid_targets_dropped.{label}")
        if count == 0:
            raise ValueError("invalid target audit counts must be positive")
        invalid_target_total += count
    if invalid_target_total > original_rows - cleaned_rows:
        raise ValueError("invalid target counts exceed the original-to-cleaned row reduction")

    target_counts = _class_count_mapping(audit["target_distribution"], "target_distribution")
    if sum(target_counts.values()) != cleaned_rows:
        raise ValueError("data audit target distribution must sum to the cleaned population")
    expected_source_counts = {int(key): int(value) for key, value in source_counts.items()}
    if target_counts != expected_source_counts:
        raise ValueError("data audit target distribution must match split assignments")
    return audit


def _validate_split_and_registry(
    split: object,
    assignments_path: Path,
    registry_path: Path,
    *,
    random_state: int,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, dict[str, str]]:
    split = _require_exact_keys(
        split,
        {
            "schema_version",
            "random_state",
            "test_size",
            "stratification",
            "source_population",
            "development",
            "development_complete_case_sensitivity",
            "locked_historical_internal_holdout",
            "assignments_file",
        },
        "split manifest",
    )
    if (
        split["schema_version"] != "baseline_split_v1"
        or isinstance(split["random_state"], bool)
        or split["random_state"] != random_state
        or split["test_size"] != 0.30
        or split["stratification"] != "target_only"
        or split["assignments_file"] != "results/split_assignments.csv"
    ):
        raise ValueError("split manifest metadata does not match the fixed release split")

    assignments = pd.read_csv(assignments_path, dtype={"row_id": str})
    if list(assignments.columns) != ["row_id", "target", "assignment"] or len(assignments) != 400:
        raise ValueError("split assignments must have the exact 400-row schema")
    assignments["decoded_row_id"] = assignments["row_id"].map(_decode_typed_row_id)
    if assignments["decoded_row_id"].map(_typed_row_id).duplicated().any():
        raise ValueError("split assignment row IDs must be unique")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or int(value) not in {0, 1}
        for value in assignments["target"]
    ):
        raise ValueError("split assignment targets must be exact binary integers")
    roles = {
        "development",
        "locked_historical_internal_holdout",
    }
    if set(assignments["assignment"]) != roles:
        raise ValueError("split assignments contain an invalid branch label")
    role_counts = assignments["assignment"].value_counts().to_dict()
    if role_counts != {
        "development": 280,
        "locked_historical_internal_holdout": 120,
    }:
        raise ValueError("split assignments must contain the fixed 280/120 partition")

    source_counts = {
        int(key): int(value)
        for key, value in assignments["target"].value_counts().sort_index().items()
    }
    branch_counts: dict[str, dict[int, int]] = {}
    for role in roles:
        branch = assignments.loc[assignments["assignment"] == role, "target"]
        branch_counts[role] = {
            int(key): int(value) for key, value in branch.value_counts().sort_index().items()
        }

    population_specs = {
        "source_population": (400, source_counts),
        "development": (280, branch_counts["development"]),
        "locked_historical_internal_holdout": (
            120,
            branch_counts["locked_historical_internal_holdout"],
        ),
    }
    for key, (expected_n, expected_counts) in population_specs.items():
        expected_keys = {"n_rows", "class_counts"}
        if key == "locked_historical_internal_holdout":
            expected_keys.add("status")
        record = _require_exact_keys(split[key], expected_keys, f"split {key}")
        if _nonnegative_int(record["n_rows"], f"split {key} n_rows") != expected_n:
            raise ValueError(f"split {key} row count does not match assignments")
        if _class_count_mapping(record["class_counts"], f"split {key} class_counts") != expected_counts:
            raise ValueError(f"split {key} class counts do not match assignments")
        if key == "locked_historical_internal_holdout" and record["status"] != (
            "locked_historical_internal_holdout_not_evaluated"
        ):
            raise ValueError("locked holdout status is invalid")

    registry = read_split_registry_csv(registry_path)
    if set(registry["population"]) != {FULL_SAMPLE, COMPLETE_CASE}:
        raise ValueError("persisted registry must contain exactly two populations")
    assignment_targets = {
        _typed_row_id(row.decoded_row_id): int(row.target)
        for row in assignments.itertuples(index=False)
    }
    development_keys = {
        _typed_row_id(value)
        for value in assignments.loc[
            assignments["assignment"] == "development", "decoded_row_id"
        ]
    }
    for row in registry.itertuples(index=False):
        key = _typed_row_id(row.row_id)
        if key not in development_keys or int(row.target) != assignment_targets[key]:
            raise ValueError("registry row identity/target must match development assignments")

    registry_hashes: dict[str, str] = {}
    population_counts: dict[str, dict[int, int]] = {}
    for population in (FULL_SAMPLE, COMPLETE_CASE):
        population_registry = registry.loc[
            registry["population"] == population
        ].reset_index(drop=True)
        identity_rows = population_registry.drop_duplicates("row_id")
        identity_keys = {_typed_row_id(value) for value in identity_rows["row_id"]}
        ordered_ids = [
            value
            for value in assignments.loc[
                assignments["assignment"] == "development", "decoded_row_id"
            ]
            if _typed_row_id(value) in identity_keys
        ]
        population_y = pd.Series(
            [assignment_targets[_typed_row_id(value)] for value in ordered_ids],
            index=ordered_ids,
            dtype=int,
        )
        counts = {
            int(key): int(value)
            for key, value in population_y.value_counts().sort_index().items()
        }
        population_counts[population] = counts
        registry_hashes[population] = validate_split_registry(
            population_registry,
            population_y,
            expected_population=population,
            expected_scheme=BASELINE_SCHEME,
            expected_class_counts=counts,
        )
    if len(registry.loc[registry["population"] == FULL_SAMPLE].drop_duplicates("row_id")) != 280:
        raise ValueError("full-sample registry must contain all 280 development rows")

    complete_record = _require_exact_keys(
        split["development_complete_case_sensitivity"],
        {"n_rows", "class_counts"},
        "development complete-case sensitivity",
    )
    complete_n = len(
        registry.loc[registry["population"] == COMPLETE_CASE].drop_duplicates("row_id")
    )
    if _nonnegative_int(complete_record["n_rows"], "complete-case n_rows") != complete_n:
        raise ValueError("complete-case split count does not match its registry")
    if _class_count_mapping(
        complete_record["class_counts"], "complete-case class_counts"
    ) != population_counts[COMPLETE_CASE]:
        raise ValueError("complete-case class counts do not match its registry")
    return split, assignments, registry, registry_hashes


def _csv_round_trip(frame: pd.DataFrame) -> pd.DataFrame:
    buffer = StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)
    return pd.read_csv(buffer)


def _validate_fold_registry_counts(
    fold_scores: pd.DataFrame, registry: pd.DataFrame
) -> None:
    expected_counts = {
        (population, int(fold_id)): (
            int((group["role"] == "train").sum()),
            int((group["role"] == "validation").sum()),
        )
        for (population, fold_id), group in registry.groupby(
            ["population", "fold_id"], sort=False
        )
    }
    for row in fold_scores.itertuples(index=False):
        expected = expected_counts[(str(row.population), int(row.fold_id))]
        if (int(row.n_train), int(row.n_validation)) != expected:
            raise ValueError("fold-score train/validation counts do not match its registry fold")


def validate_release_evidence(
    root: Path,
    *,
    expected_config: Mapping[str, object],
    rendered: bool,
    expected_comparison_sha256: str | None = None,
    expected_figure_sha256: str | None = None,
) -> ValidatedReleaseEvidence:
    """Strict-load and cross-validate every machine artifact before signing."""
    try:
        root = Path(root)
        actual_files = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        expected_files = set(RELEASE_OUTPUTS if rendered else MACHINE_OUTPUTS) | {"run.log"}
        if actual_files != expected_files:
            raise ValueError(
                "release evidence tree mismatch; "
                f"missing={sorted(expected_files - actual_files)}, "
                f"extra={sorted(actual_files - expected_files)}"
            )
        config = _load_strict_json(root / "results" / "config.json")
        if not isinstance(config, dict) or config != dict(expected_config):
            raise ValueError("persisted config must exactly match the effective release config")
        _runtime_versions(lambda: config.get("runtime_versions"))
        audit_raw = _load_strict_json(root / "results" / "data_audit.json")
        split_raw = _load_strict_json(root / "results" / "split_manifest.json")
        selection = _load_strict_json(root / "results" / "selection_manifest.json")
        if not isinstance(selection, dict):
            raise ValueError("selection manifest must be a JSON object")

        effective = config.get("effective_configuration")
        if not isinstance(effective, dict):
            raise ValueError("config must record its effective configuration")
        recorded_seed = _validate_random_state(effective.get("random_state"))
        split, assignments, registry, registry_hashes = _validate_split_and_registry(
            split_raw,
            root / "results" / "split_assignments.csv",
            root / "results" / "cv_split_registry.csv",
            random_state=recorded_seed,
        )
        source_counts = {
            int(key): int(value)
            for key, value in assignments["target"].value_counts().sort_index().items()
        }
        audit = _validate_audit(
            audit_raw,
            data_relative=str(config["input"]["path"]),
            source_counts=source_counts,
        )

        fold_scores = pd.read_csv(root / "results" / "cv_fold_scores.csv")
        comparison = pd.read_csv(root / "results" / "model_comparison.csv")
        _validate_release_tables(comparison, fold_scores, registry_hashes)
        _validate_fold_registry_counts(fold_scores, registry)
        population_sizes = {
            population: len(
                registry.loc[registry["population"] == population].drop_duplicates("row_id")
            )
            for population in (FULL_SAMPLE, COMPLETE_CASE)
        }
        if any(
            int(row.n_population) != population_sizes[str(row.population)]
            for row in comparison.itertuples(index=False)
        ):
            raise ValueError("comparison population counts do not match signed registries")

        canonical_comparison = _csv_round_trip(
            summarize_baseline_fold_scores(fold_scores)
        )
        pd.testing.assert_frame_equal(
            comparison.reset_index(drop=True),
            canonical_comparison.reset_index(drop=True),
            check_exact=True,
            check_dtype=True,
        )
        canonical_selection = build_selection_manifest(comparison, stage="baseline")
        if selection != canonical_selection:
            raise ValueError("selection manifest does not match the persisted comparison")

        if rendered:
            _validate_exact_release_outputs(root)
            _require_matching_sha256(
                root,
                "results/model_comparison.csv",
                expected_comparison_sha256,
                "persisted comparison",
            )
            _require_matching_sha256(
                root,
                "results/figures/model_comparison.png",
                expected_figure_sha256,
                "rendered comparison figure",
            )
            _validate_rendered_png(root / "results" / "figures" / "model_comparison.png")
            if (root / "run.log").read_text(encoding="utf-8") != _expected_release_log():
                raise ValueError("release run log does not match the exact successful run grammar")
            from .reproduction_report import render_reproduction_report

            canonical_report = render_reproduction_report(
                config=config,
                audit=audit,
                split=split,
                comparison=comparison,
                selection_manifest=selection,
            )
            report_path = root / "reports" / "reproduction_report.md"
            if report_path.read_text(encoding="utf-8") != canonical_report:
                raise ValueError("reproduction report does not match validated evidence")

        return ValidatedReleaseEvidence(
            config=config,
            audit=audit,
            split=split,
            assignments=assignments,
            registry=registry,
            fold_scores=fold_scores,
            comparison=comparison,
            selection=selection,
            registry_hashes=registry_hashes,
        )
    except ArtifactStageError as exc:
        raise ArtifactStageError(f"release evidence validation failed: {exc}") from exc
    except Exception as exc:
        raise ArtifactStageError(f"release evidence validation failed: {exc}") from exc


def run_baseline_experiment(
    *,
    layout: ArtifactLayout,
    data_path: Path,
    cv_splits: int = 5,
    model_names: Sequence[str] | None = None,
    dependencies: BaselineDependencies | None = None,
    random_state: int = RANDOM_STATE,
) -> Path:
    """Run one release baseline or one visibly non-release diagnostic scratch grid.

    ``random_state`` seeds the fixed 70/30 split, the CV registries, every model
    factory and the Python/NumPy generators; it is recorded in ``config.json`` and in
    ``split_manifest.json`` so the advanced stage can inherit exactly the same seed.
    """
    if not isinstance(layout, ArtifactLayout):
        raise TypeError("layout must be an ArtifactLayout")
    if isinstance(cv_splits, bool) or not isinstance(cv_splits, Integral) or int(cv_splits) < 2:
        raise ValueError("cv_splits must be an integer greater than or equal to 2")
    cv_splits = int(cv_splits)
    random_state = _validate_random_state(random_state)
    filtered = model_names is not None
    if filtered and (isinstance(model_names, (str, bytes)) or not list(model_names)):
        raise ValueError("model_names filter must contain one or more exact model names")
    requested_models = None if model_names is None else list(model_names)
    if requested_models is not None and (
        len(requested_models) != len(set(requested_models))
        or any(not isinstance(name, str) or not name.strip() for name in requested_models)
    ):
        raise ValueError("model_names filter must contain unique nonblank strings")

    data_path = layout.validate_data_path(Path(data_path))
    data_relative = data_path.relative_to(layout.project_root).as_posix()
    if not data_relative.startswith("data/"):
        raise ValueError("data path must be recorded beneath data/")
    initial_data_hash = sha256_file(layout.project_root, data_relative)
    initial_source_hashes = _source_hashes(layout)
    deps = dependencies or default_dependencies()
    _assert_frozen_identity(
        layout,
        data_relative=data_relative,
        data_hash=initial_data_hash,
        source_hashes=initial_source_hashes,
    )

    release_eligible = not filtered and cv_splits == 5
    workspace = layout.begin_stage("baseline") if release_eligible else None
    output_root = (
        workspace.partial_dir
        if workspace is not None
        else layout.create_diagnostic_directory("baseline-diagnostic")
    )
    if not release_eligible:
        _write_json(
            output_root / "NON_RELEASE_DIAGNOSTIC.json",
            {
                "release_eligible": False,
                "message": "This scratch result is not a completed baseline release.",
                "reasons": _diagnostic_reasons(filtered=filtered, cv_splits=cv_splits),
            },
        )

    results_dir = output_root / "results"
    reports_dir = output_root / "reports"
    figures_dir = results_dir / "figures"
    log_path = output_root / "run.log"
    with _run_log(log_path) as log:
        log.write("baseline experiment started\n")
        log.write(f"mode={'release' if release_eligible else 'diagnostic'}\n")
        figures_dir.mkdir(parents=True)
        reports_dir.mkdir(parents=True)
        random.seed(random_state)
        np.random.seed(random_state)
        versions = _runtime_versions(deps.runtime_versions)
        registered = deps.model_registry(random_state)
        if not isinstance(registered, Mapping):
            raise TypeError("model registry must return a mapping")
        registered_names = list(registered)
        if requested_models is None:
            selected_names = registered_names
        else:
            unknown = [name for name in requested_models if name not in registered]
            if unknown:
                raise ValueError(f"unknown model names: {unknown}; available: {registered_names}")
            selected_names = requested_models
        config = _config(
            release_eligible=release_eligible,
            data_relative=data_relative,
            data_hash=initial_data_hash,
            source_hashes=initial_source_hashes,
            runtime_versions=versions,
            cv_splits=cv_splits,
            model_names=selected_names,
            filtered=filtered,
            random_state=random_state,
        )
        _write_json(results_dir / "config.json", config)
        if release_eligible and tuple(registered_names) != RELEASE_MODELS:
            raise ValueError("release model registry must contain the exact nine-model domain and order")

        X, y, audit = deps.data_loader(data_path)
        _validate_loaded_data(X, y, release_eligible=release_eligible)
        _write_json(results_dir / "data_audit.json", _audit_mapping(audit, data_relative))
        development_X, development_y, holdout_ids = _split_development(X, y, random_state=random_state)
        complete_X, complete_y = complete_case_subset(development_X, development_y)
        _persist_split(
            y=y,
            development_y=development_y,
            complete_case_y=complete_y,
            holdout_ids=holdout_ids,
            results_dir=results_dir,
            random_state=random_state,
        )
        del X, y, holdout_ids
        if cv_splits == 5:
            full_registry = materialize_baseline_registry(
                development_y, population=FULL_SAMPLE, random_state=random_state
            )
            complete_registry = materialize_baseline_registry(
                complete_y, population=COMPLETE_CASE, random_state=random_state
            )
            registry_hashes = {
                FULL_SAMPLE: validate_split_registry(
                    full_registry,
                    development_y,
                    expected_population=FULL_SAMPLE,
                    expected_scheme=BASELINE_SCHEME,
                    expected_class_counts={int(k): v for k, v in development_y.value_counts().items()},
                ),
                COMPLETE_CASE: validate_split_registry(
                    complete_registry,
                    complete_y,
                    expected_population=COMPLETE_CASE,
                    expected_scheme=BASELINE_SCHEME,
                    expected_class_counts={int(k): v for k, v in complete_y.value_counts().items()},
                ),
            }
        else:
            full_registry = _materialize_diagnostic_registry(
                development_y, population=FULL_SAMPLE, cv_splits=cv_splits, random_state=random_state
            )
            complete_registry = _materialize_diagnostic_registry(
                complete_y, population=COMPLETE_CASE, cv_splits=cv_splits, random_state=random_state
            )
            registry_hashes = {
                FULL_SAMPLE: _validate_diagnostic_registry(
                    full_registry,
                    development_y,
                    population=FULL_SAMPLE,
                    cv_splits=cv_splits,
                ),
                COMPLETE_CASE: _validate_diagnostic_registry(
                    complete_registry,
                    complete_y,
                    population=COMPLETE_CASE,
                    cv_splits=cv_splits,
                ),
            }
        write_split_registry_csv(
            pd.concat([full_registry, complete_registry], ignore_index=True),
            results_dir / "cv_split_registry.csv",
        )

        score_tables: list[pd.DataFrame] = []
        total = len(RELEASE_SCENARIOS) * len(RELEASE_IMPUTATIONS) * len(selected_names)
        counter = 0
        for imputation in RELEASE_IMPUTATIONS:
            if imputation == "complete_case":
                branch_X, branch_y, registry, population = (
                    complete_X,
                    complete_y,
                    complete_registry,
                    COMPLETE_CASE,
                )
            else:
                branch_X, branch_y, registry, population = (
                    development_X,
                    development_y,
                    full_registry,
                    FULL_SAMPLE,
                )
            if branch_y.value_counts().min() < cv_splits:
                raise ValueError(f"not enough samples per class for {cv_splits}-fold CV")
            for scenario in RELEASE_SCENARIOS:
                for model_name in selected_names:
                    counter += 1
                    log.write(f"candidate {counter}/{total}: {scenario} | {imputation} | {model_name}\n")
                    pipeline = deps.pipeline_builder(
                        scenario, imputation, registered[model_name], random_state=random_state
                    )
                    scores = deps.fold_score_generator(
                        pipeline,
                        branch_X,
                        branch_y,
                        registry,
                        population=population,
                        scenario=scenario,
                        imputation=imputation,
                        model_name=model_name,
                        expected_class_counts={int(k): int(v) for k, v in branch_y.value_counts().items()},
                        cv_splits=cv_splits,
                    )
                    score_tables.append(scores)

        fold_scores = pd.concat(score_tables, ignore_index=True)
        fold_scores.to_csv(results_dir / "cv_fold_scores.csv", index=False)
        persisted_scores = pd.read_csv(results_dir / "cv_fold_scores.csv")
        comparison = deps.fold_summary(persisted_scores)
        comparison.to_csv(results_dir / "model_comparison.csv", index=False)
        persisted_comparison = pd.read_csv(results_dir / "model_comparison.csv")
        selection_path = results_dir / "selection_manifest.json"
        if release_eligible:
            selection_manifest = build_selection_manifest(persisted_comparison, stage="baseline")
            _write_json(selection_path, selection_manifest)
        elif cv_splits == 5:
            try:
                selection_manifest = build_selection_manifest(persisted_comparison, stage="baseline")
            except ValueError:
                selection_manifest = None
            if selection_manifest is not None:
                _write_json(selection_path, selection_manifest)
        if release_eligible:
            validate_release_evidence(
                output_root,
                expected_config=config,
                rendered=False,
            )
        rendered_comparison_sha256 = sha256_file(
            output_root, "results/model_comparison.csv"
        )
        deps.plotter(persisted_comparison.copy(deep=True), figures_dir / "model_comparison.png")
        if release_eligible:
            _validate_release_figure_binding(
                persisted_comparison,
                figures_dir / "model_comparison.png",
            )
        rendered_figure_sha256 = sha256_file(
            output_root, "results/figures/model_comparison.png"
        )
        if release_eligible:
            deps.report_renderer(
                config_path=results_dir / "config.json",
                data_audit_path=results_dir / "data_audit.json",
                split_manifest_path=results_dir / "split_manifest.json",
                comparison_path=results_dir / "model_comparison.csv",
                selection_manifest_path=selection_path,
                output_path=reports_dir / "reproduction_report.md",
            )
        else:
            _write_diagnostic_report(
                config,
                persisted_comparison,
                reports_dir / "reproduction_report.md",
            )

        if release_eligible:
            log.flush()
            _validate_release_tables(persisted_comparison, persisted_scores, registry_hashes)
            _validate_exact_release_outputs(output_root)
        else:
            _validate_diagnostic_tables(
                persisted_comparison,
                persisted_scores,
                candidate_count=total,
                cv_splits=cv_splits,
            )
        _assert_frozen_identity(
            layout,
            data_relative=data_relative,
            data_hash=initial_data_hash,
            source_hashes=initial_source_hashes,
        )
        log.write("outputs validated; ready for atomic completion\n")

    if workspace is None:
        return output_root
    try:
        final_evidence = validate_release_evidence(
            output_root,
            expected_config=config,
            rendered=True,
            expected_comparison_sha256=rendered_comparison_sha256,
            expected_figure_sha256=rendered_figure_sha256,
        )
        _assert_frozen_identity(
            layout,
            data_relative=data_relative,
            data_hash=initial_data_hash,
            source_hashes=initial_source_hashes,
        )
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as failure_log:
            failure_log.write(f"post_close_error={type(exc).__name__}: {exc}\n")
        raise
    return workspace.complete(
        required_files=RELEASE_OUTPUTS,
        provenance=final_evidence.config,
    )
