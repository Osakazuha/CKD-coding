"""Stability of the supervised feature selection (revision stage).

Scenario C retains eight of the 24 named clinical fields by training-fold mutual
information.  The main manuscript reports the panel fitted on the full development
sample; this module quantifies how stable that panel is when the selector is refitted
on every persisted training fold of the main analysis (15 folds) and of each seed
replication (15 folds per seed, on that seed's own development sample), separately
for each missing-data strategy.  Reported per strategy: the selection frequency of
each field, the mean pairwise Jaccard similarity of the selected panels, and the
stability index of Nogueira, Sechidis and Brown (JMLR 2018) with a nonparametric
bootstrap interval over the selection sets.  The selector is fitted exactly as inside
the modeling pipeline (same imputation-specific screening, same random state); no
classifier is trained and the holdout is never touched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from .data_preprocessing import ALL_FEATURES
from .feature_selection import ClinicalFeatureSelector
from .revision_common import class_counts, outer_splits
from .split_registry import COMPLETE_CASE, FULL_SAMPLE


STABILITY_FILES: Final = (
    "selection_stability_selections.csv",
    "selection_stability_frequency.csv",
    "selection_stability_summary.csv",
)
SELECTION_FRACTION: Final = 0.30
BOOTSTRAP_RESAMPLES: Final = 2000


def nogueira_stability(membership: np.ndarray) -> float:
    """Stability index of Nogueira et al. (2018) for an M x d binary selection matrix."""
    Z = np.asarray(membership, dtype=float)
    if Z.ndim != 2 or Z.shape[0] < 2:
        raise ValueError("stability requires at least two selection sets")
    M, d = Z.shape
    p = Z.mean(axis=0)
    k_bar = Z.sum(axis=1).mean()
    sample_variances = (M / (M - 1.0)) * p * (1.0 - p)
    denominator = (k_bar / d) * (1.0 - k_bar / d)
    if denominator <= 0.0:
        return float("nan")
    return float(1.0 - sample_variances.mean() / denominator)


def mean_pairwise_jaccard(sets: Sequence[frozenset[str]]) -> float:
    pairs = list(combinations(sets, 2))
    if not pairs:
        return float("nan")
    return float(np.mean([len(a & b) / len(a | b) if (a | b) else 1.0 for a, b in pairs]))


def bootstrap_stability_interval(membership: np.ndarray, *, resamples: int, random_state: int) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    M = membership.shape[0]
    values = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sample = membership[rng.integers(0, M, size=M)]
        values[index] = nogueira_stability(sample)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def _selector_on_fold(X_train: pd.DataFrame, y_train: pd.Series, *, imputation: str, random_state: int) -> ClinicalFeatureSelector:
    selector = ClinicalFeatureSelector(fraction=SELECTION_FRACTION, imputation=imputation, random_state=random_state)
    return selector.fit(X_train.loc[:, ALL_FEATURES], y_train)


def run_selection_stability(
    *,
    seed_branches: Mapping[int, Mapping[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]]],
    main_seed: int,
    results_dir: Path,
    log,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, pd.DataFrame]:
    """Refit the fold-level selector for every seed, strategy and training fold."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    selection_rows: list[dict[str, object]] = []
    for seed, branches in seed_branches.items():
        for imputation in ("median_mode", "knn", "complete_case"):
            population = COMPLETE_CASE if imputation == "complete_case" else FULL_SAMPLE
            X, y, registry = branches[population]
            log.write(f"selection stability: seed={seed} imputation={imputation} population={population} n={len(y)} {class_counts(y)}\n")
            log.flush()
            for split in outer_splits(registry, y, population=population):
                train_ids = list(split.train_row_ids)
                selector = _selector_on_fold(X.loc[train_ids], y.loc[train_ids], imputation=imputation, random_state=int(seed))
                ranked = sorted(ALL_FEATURES, key=lambda column: (-selector.feature_scores_[column], ALL_FEATURES.index(column)))
                selection_rows.append({
                    "seed": int(seed), "role": "main_analysis" if int(seed) == int(main_seed) else "sensitivity_replication",
                    "imputation": imputation, "population": population, "registry_hash": split.registry_hash,
                    "repeat_id": split.repeat_id, "fold_id": split.fold_id, "n_train": len(train_ids),
                    "n_selected": len(selector.selected_features_),
                    "selected_features": ";".join(selector.selected_features_),
                    "mutual_information_rank_order": ";".join(ranked),
                    **{f"score_{column}": float(selector.feature_scores_[column]) for column in ALL_FEATURES},
                })
        for imputation in ("median_mode", "knn"):
            X, y, _registry = branches[FULL_SAMPLE]
            selector = _selector_on_fold(X, y, imputation=imputation, random_state=int(seed))
            selection_rows.append({
                "seed": int(seed), "role": "full_development_fit",
                "imputation": imputation, "population": FULL_SAMPLE, "registry_hash": "",
                "repeat_id": 0, "fold_id": 0, "n_train": len(y),
                "n_selected": len(selector.selected_features_),
                "selected_features": ";".join(selector.selected_features_),
                "mutual_information_rank_order": ";".join(sorted(ALL_FEATURES, key=lambda c: (-selector.feature_scores_[c], ALL_FEATURES.index(c)))),
                **{f"score_{column}": float(selector.feature_scores_[column]) for column in ALL_FEATURES},
            })
    selections = pd.DataFrame(selection_rows)
    selections.to_csv(results_dir / "selection_stability_selections.csv", index=False)
    frequency = summarize_selection_frequency(selections)
    frequency.to_csv(results_dir / "selection_stability_frequency.csv", index=False)
    summary = summarize_stability(selections, bootstrap_resamples=bootstrap_resamples, random_state=main_seed)
    summary.to_csv(results_dir / "selection_stability_summary.csv", index=False)
    return {"selections": selections, "frequency": frequency, "summary": summary}


def _fold_level(selections: pd.DataFrame) -> pd.DataFrame:
    return selections[selections["role"] != "full_development_fit"]


def _membership(selections: pd.DataFrame) -> np.ndarray:
    return np.array([
        [1 if column in set(str(value).split(";")) else 0 for column in ALL_FEATURES]
        for value in selections["selected_features"]
    ], dtype=int)


def summarize_selection_frequency(selections: pd.DataFrame) -> pd.DataFrame:
    """Per strategy and field: selection frequency across all folds, per seed, and mean MI rank."""
    rows = []
    fold_level = _fold_level(selections)
    for imputation, group in fold_level.groupby("imputation", sort=False):
        membership = _membership(group)
        rank_matrix = np.array([
            [str(order).split(";").index(column) + 1 for column in ALL_FEATURES]
            for order in group["mutual_information_rank_order"]
        ], dtype=float)
        seeds = sorted(set(group["seed"].astype(int)))
        for position, column in enumerate(ALL_FEATURES):
            row: dict[str, object] = {
                "imputation": imputation, "feature": column,
                "n_selection_sets": int(len(group)),
                "selection_frequency": float(membership[:, position].mean()),
                "mean_mutual_information_rank": float(rank_matrix[:, position].mean()),
                "mean_mutual_information_score": float(group[f"score_{column}"].mean()),
            }
            for seed in seeds:
                seed_mask = (group["seed"].astype(int) == seed).to_numpy()
                row[f"selection_frequency_seed_{seed}"] = float(membership[seed_mask, position].mean())
            rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.sort_values(["imputation", "selection_frequency", "mean_mutual_information_rank"], ascending=[True, False, True], kind="stable", ignore_index=True)


def summarize_stability(selections: pd.DataFrame, *, bootstrap_resamples: int, random_state: int) -> pd.DataFrame:
    """Per strategy: Nogueira stability with bootstrap interval, mean pairwise Jaccard, core panel."""
    rows = []
    fold_level = _fold_level(selections)
    for imputation, group in fold_level.groupby("imputation", sort=False):
        membership = _membership(group)
        sets = [frozenset(str(value).split(";")) for value in group["selected_features"]]
        frequency = membership.mean(axis=0)
        always = [column for column, value in zip(ALL_FEATURES, frequency) if value == 1.0]
        never = [column for column, value in zip(ALL_FEATURES, frequency) if value == 0.0]
        majority = [column for column, value in zip(ALL_FEATURES, frequency) if value >= 0.5]
        low, high = bootstrap_stability_interval(membership, resamples=bootstrap_resamples, random_state=random_state)
        main_group = group[group["role"] == "main_analysis"]
        main_sets = [frozenset(str(value).split(";")) for value in main_group["selected_features"]]
        full_fit = selections[(selections["role"] == "full_development_fit") & (selections["imputation"] == imputation)
                              & (selections["seed"].astype(int) == int(random_state))]
        reference_panel = frozenset(str(full_fit["selected_features"].iat[0]).split(";")) if len(full_fit) else frozenset()
        rows.append({
            "imputation": imputation,
            "n_selection_sets": int(len(group)),
            "n_seeds": int(group["seed"].nunique()),
            "panel_size": int(round(float(membership.sum(axis=1).mean()))),
            "nogueira_stability": nogueira_stability(membership),
            "nogueira_stability_bootstrap_p2_5": low,
            "nogueira_stability_bootstrap_p97_5": high,
            "bootstrap_resamples": int(bootstrap_resamples),
            "mean_pairwise_jaccard": mean_pairwise_jaccard(sets),
            "main_seed_nogueira_stability": nogueira_stability(_membership(main_group)) if len(main_group) >= 2 else float("nan"),
            "main_seed_mean_pairwise_jaccard": mean_pairwise_jaccard(main_sets),
            "n_distinct_panels": int(len(set(sets))),
            "always_selected": ";".join(always),
            "n_always_selected": len(always),
            "selected_in_majority": ";".join(majority),
            "never_selected": ";".join(never),
            "main_full_development_panel": ";".join(sorted(reference_panel, key=ALL_FEATURES.index)),
            "mean_jaccard_with_main_full_development_panel": float(np.mean([
                len(s & reference_panel) / len(s | reference_panel) for s in sets
            ])) if reference_panel else float("nan"),
        })
    return pd.DataFrame(rows)


def validate_stability_outputs(outputs: Mapping[str, Any], *, expected_seeds: Sequence[int]) -> None:
    selections, summary = outputs["selections"], outputs["summary"]
    if set(selections["seed"].astype(int)) != set(int(seed) for seed in expected_seeds):
        raise ValueError("selection stability must cover every seed")
    if set(summary["imputation"]) != {"median_mode", "knn", "complete_case"}:
        raise ValueError("selection stability summary must cover every strategy")
    fold_level = _fold_level(selections)
    per_seed = fold_level.groupby(["seed", "imputation"]).size()
    if not per_seed.eq(15).all():
        raise ValueError("selection stability requires 15 fold-level selections per seed and strategy")
    if (fold_level["n_selected"] != 8).any():
        raise ValueError("the fold-level selector must retain eight fields")
    values = summary[["nogueira_stability", "mean_pairwise_jaccard"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("stability summary metrics must be finite")
