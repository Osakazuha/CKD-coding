"""Signed release-artifact checks for the advanced stage contract."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from src.advanced_evaluation import (
    ADVANCED_METRICS,
    BOOTSTRAP_INTERVAL_LABEL,
    REPEATED_CV_FOLD_SCORE_COLUMNS,
    REPEATED_CV_SUMMARY_COLUMNS,
)
from src.advanced_experiment import RELEASE_OUTPUTS, release_outputs
from src.data_preprocessing import ALL_FEATURES
from src.model_selection import CANDIDATE_KEY_COLUMNS, get_display_representative, get_selection_set
from src.seed_sensitivity import (
    ROLE_MAIN,
    ROLE_SENSITIVITY,
    SEED_OUTPUT_FILES,
    SENSITIVITY_SCOPES,
    SUMMARY_COLUMNS,
    validate_seed_sensitivity_summary,
)
from src.split_registry import (
    ADVANCED_SCHEME,
    COMPLETE_CASE,
    FULL_SAMPLE,
    read_split_registry_csv,
    validate_split_registry,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key(row) -> tuple[str, str, str]:
    return tuple(str(row[column]) for column in CANDIDATE_KEY_COLUMNS)


def _config(stage: Path) -> dict:
    return json.loads((stage / "results" / "config.json").read_text(encoding="utf-8"))


def _sensitivity_seeds(advanced_stage: Path) -> list[int]:
    return [int(seed) for seed in _config(advanced_stage)["effective_configuration"]["seed_sensitivity"]["seeds"]]


@pytest.mark.release_artifact
def test_signed_advanced_stage_has_exact_tree_and_cv_only_grid(advanced_stage: Path):
    manifest = json.loads((advanced_stage / "advanced_manifest.json").read_text(encoding="utf-8"))
    actual_files = {
        path.relative_to(advanced_stage).as_posix()
        for path in advanced_stage.rglob("*")
        if path.is_file()
    }
    assert actual_files == set(release_outputs(_sensitivity_seeds(advanced_stage))) | {"advanced_manifest.json"}
    for relative, record in manifest["outputs"].items():
        path = advanced_stage / relative
        assert record == {"sha256": _sha256(path), "size_bytes": path.stat().st_size}

    comparison = pd.read_csv(advanced_stage / "results" / "model_comparison_repeated_cv.csv")
    folds = pd.read_csv(advanced_stage / "results" / "repeated_cv_fold_scores.csv")
    assert list(comparison.columns) == REPEATED_CV_SUMMARY_COLUMNS
    assert list(folds.columns) == REPEATED_CV_FOLD_SCORE_COLUMNS
    assert len(comparison) == 81
    assert comparison[list(CANDIDATE_KEY_COLUMNS)].drop_duplicates().shape[0] == 81
    assert int((comparison["model"] == "XGBoost").sum()) == 9
    assert len(folds) == 81 * 15
    assert folds.groupby(list(CANDIDATE_KEY_COLUMNS)).size().eq(15).all()
    assert set(folds["repeat_id"]) == {1, 2, 3} and set(folds["fold_id"]) == {1, 2, 3, 4, 5}
    assert set(folds["cv_scheme"]) == {ADVANCED_SCHEME}
    assert (comparison["n_validation_folds"] == 15).all()
    for metric in ADVANCED_METRICS:
        assert folds[metric].map(lambda value: math.isfinite(value) and 0 <= value <= 1).all()
        assert (comparison[f"rcv_{metric}_fold_p2_5"] <= comparison[f"rcv_{metric}_fold_p97_5"]).all()
    forbidden = ("test_", "holdout", "prediction", "bootstrap", "calibration", "confusion", "shap", "ci95")
    assert not any(any(token in column.lower() for token in forbidden) for column in [*comparison, *folds])
    pca_folds = folds[folds["scenario"] == "B_pca_95"]
    assert pca_folds["fold_pca_components"].notna().all()
    assert folds[folds["scenario"] != "B_pca_95"]["fold_pca_components"].isna().all()


@pytest.mark.release_artifact
def test_signed_advanced_registries_are_population_specific(baseline_stage: Path, advanced_stage: Path):
    split = json.loads((baseline_stage / "results" / "split_manifest.json").read_text(encoding="utf-8"))
    expected_rows = {FULL_SAMPLE: int(split["development"]["n_rows"]), COMPLETE_CASE: int(split["development_complete_case_sensitivity"]["n_rows"])}
    assert expected_rows[FULL_SAMPLE] == 280 and expected_rows[COMPLETE_CASE] < 280
    hashes = {}
    for population, name in ((FULL_SAMPLE, "cv_split_registry_full.csv"), (COMPLETE_CASE, "cv_split_registry_complete_case.csv")):
        registry = read_split_registry_csv(advanced_stage / "results" / name)
        identities = registry.drop_duplicates("row_id").set_index("row_id")["target"]
        hashes[population] = validate_split_registry(
            registry, identities, expected_population=population, expected_scheme=ADVANCED_SCHEME,
            expected_class_counts={int(key): int(value) for key, value in identities.value_counts().items()},
        )
        assert len(identities) == expected_rows[population]
    assert hashes[FULL_SAMPLE] != hashes[COMPLETE_CASE]
    comparison = pd.read_csv(advanced_stage / "results" / "model_comparison_repeated_cv.csv")
    assert set(comparison.loc[comparison["imputation"] == "complete_case", "registry_hash"]) == {hashes[COMPLETE_CASE]}
    assert set(comparison.loc[comparison["imputation"] != "complete_case", "registry_hash"]) == {hashes[FULL_SAMPLE]}


@pytest.mark.release_artifact
def test_frozen_finalists_are_evaluated_once_with_conditional_bootstrap(baseline_stage: Path, advanced_stage: Path):
    split = json.loads((baseline_stage / "results" / "split_manifest.json").read_text(encoding="utf-8"))
    complete_development = int(split["development_complete_case_sensitivity"]["n_rows"])
    selection = json.loads((advanced_stage / "results" / "selection_manifest.json").read_text(encoding="utf-8"))
    assert selection["stage"] == "advanced"
    assert selection["selection_uses_holdout"] is False
    finalists = {_key(member) for member in get_selection_set(selection, "frozen_holdout_finalists")}
    holdout = pd.read_csv(advanced_stage / "results" / "holdout_evaluation.csv")
    assert {_key(row) for _, row in holdout.iterrows()} == finalists
    assert len(holdout) == len(finalists)
    assert (holdout["n_bootstrap"] == 1000).all()
    full = holdout[holdout["imputation"] != "complete_case"]
    complete = holdout[holdout["imputation"] == "complete_case"]
    assert (full["n_development"] == 280).all() and (full["n_holdout"] == 120).all()
    assert (complete["n_development"] == complete_development).all()
    assert complete["n_holdout"].nunique() == 1 and (complete["n_holdout"] < 120).all()
    assert (complete["holdout_non_ckd"] + complete["holdout_ckd"] == complete["n_holdout"]).all()
    assert (complete["holdout_non_ckd"] > complete["holdout_ckd"]).all()
    for metric in ADVANCED_METRICS:
        low = holdout[f"holdout_{metric}_{BOOTSTRAP_INTERVAL_LABEL}_p2_5"]
        high = holdout[f"holdout_{metric}_{BOOTSTRAP_INTERVAL_LABEL}_p97_5"]
        observed = holdout[f"holdout_{metric}_observed"]
        assert low.between(0, 1).all() and high.between(0, 1).all() and (low <= high).all()
        assert observed.between(0, 1).all()
    for scenario, model in (("A_all_original", "XGBoost"), ("B_pca_95", "XGBoost")):
        assert (scenario, "median_mode", model) in finalists
    predictions = pd.read_csv(advanced_stage / "results" / "holdout_predictions.csv")
    sizes = predictions.groupby(list(CANDIDATE_KEY_COLUMNS)).size()
    assert set(sizes.index) == finalists
    assert predictions["probability_ckd"].between(0, 1).all()


@pytest.mark.release_artifact
def test_calibration_tables_use_two_explicit_binnings(advanced_stage: Path):
    metrics = pd.read_csv(advanced_stage / "results" / "calibration_metrics.csv")
    quantile = pd.read_csv(advanced_stage / "results" / "calibration_bins_quantile.csv")
    ece = pd.read_csv(advanced_stage / "results" / "ece_bins_equal_width.csv")
    assert len(metrics) == 3 and set(metrics["scenario"]) == {"A_all_original", "B_pca_95", "C_selected_original"}
    assert (metrics["n_holdout"] == 120).all()
    assert metrics["brier_score"].between(0, 1).all() and metrics["ece_equal_width_10"].between(0, 1).all()
    assert (metrics["calibration_display_bins_realized"] <= metrics["calibration_display_bins_requested"]).all()
    for _, row in metrics.iterrows():
        key = _key(row)
        q = quantile[(quantile["scenario"] == key[0]) & (quantile["imputation"] == key[1]) & (quantile["model"] == key[2])]
        e = ece[(ece["scenario"] == key[0]) & (ece["imputation"] == key[1]) & (ece["model"] == key[2])]
        assert len(q) == int(row["calibration_display_bins_realized"]) and q["n"].sum() == 120
        assert len(e) == 10 and e["n"].sum() == 120
        assert "quantile" in q["edge_convention"].iloc[0] and "equal-width" in e["edge_convention"].iloc[0]


@pytest.mark.release_artifact
def test_ablation_is_paired_on_the_full_sample_registry_and_complete(advanced_stage: Path):
    ablation = pd.read_csv(advanced_stage / "results" / "ablation_results.csv")
    differences = pd.read_csv(advanced_stage / "results" / "ablation_fold_differences.csv")
    selection = json.loads((advanced_stage / "results" / "selection_manifest.json").read_text(encoding="utf-8"))
    reference = get_display_representative(selection, "retained_sample:A_all_original")
    assert len(ablation) == 25 and set(ablation["dropped_feature"]) == {"none", *ALL_FEATURES}
    assert set(ablation["model"]) == {reference["model"]} and set(ablation["imputation"]) == {reference["imputation"]}
    assert (ablation["population"] == FULL_SAMPLE).all() and (ablation["n_paired_folds"] == 15).all()
    assert differences.groupby("dropped_feature").size().eq(15).all()
    none = ablation[ablation["dropped_feature"] == "none"].iloc[0]
    assert none["delta_accuracy_mean"] == 0.0
    assert (ablation["delta_accuracy_fold_p2_5"] <= ablation["delta_accuracy_mean"]).all()
    assert (ablation["delta_accuracy_mean"] <= ablation["delta_accuracy_fold_p97_5"]).all()


@pytest.mark.release_artifact
def test_ordinal_sensitivity_and_grouped_shap_are_persisted(advanced_stage: Path):
    ordinal = pd.read_csv(advanced_stage / "results" / "ordinal_sensitivity.csv")
    fold_scores = pd.read_csv(advanced_stage / "results" / "ordinal_sensitivity_fold_scores.csv")
    assert len(ordinal) == 1 and ordinal.iloc[0]["ordinal_features"] == "sg;al;su"
    assert bool(ordinal.iloc[0]["changes_selection"]) is False
    assert set(fold_scores["encoding"]) == {"nominal", "ordinal"} and len(fold_scores) == 30
    grouped = pd.read_csv(advanced_stage / "results" / "shap_importance_grouped.csv")
    transformed = pd.read_csv(advanced_stage / "results" / "shap_importance_transformed.csv")
    lineage = pd.read_csv(advanced_stage / "results" / "shap_feature_lineage.csv")
    metadata = json.loads((advanced_stage / "results" / "shap_metadata.json").read_text(encoding="utf-8"))
    assert metadata["grouping"] == "signed_sum_per_observation_then_mean_absolute"
    assert metadata["scenario"] in {"A_all_original", "C_selected_original"}
    assert grouped["n_transformed_columns"].sum() == len(lineage) == len(transformed)
    assert list(grouped["rank"]) == list(range(1, len(grouped) + 1))
    assert grouped["mean_abs_grouped_shap"].is_monotonic_decreasing
    assert (grouped["mean_abs_grouped_shap"] <= grouped["sum_of_level_mean_abs_shap"] + 1e-12).all()
    pca = pd.read_csv(advanced_stage / "results" / "pca_explained_variance.csv")
    assert pca["cumulative_explained_variance"].iloc[int(pca["n_components_retained"].iloc[0]) - 1] >= 0.95


@pytest.mark.release_artifact
def test_baseline_hashes_still_match_the_protected_manifest(baseline_stage: Path, advanced_stage: Path):
    integrity = json.loads((advanced_stage / "results" / "baseline_integrity.json").read_text(encoding="utf-8"))
    assert integrity["status"] == "verified_unchanged"
    assert integrity["before"] == integrity["after"]
    manifest = json.loads((baseline_stage / "baseline_manifest.json").read_text(encoding="utf-8"))
    expected = {f"baseline/{relative}": _sha256(baseline_stage / relative) for relative in manifest["outputs"]}
    expected["baseline/baseline_manifest.json"] = _sha256(baseline_stage / "baseline_manifest.json")
    assert integrity["after"] == expected
    assert integrity["baseline_manifest_sha256"] == expected["baseline/baseline_manifest.json"]
    config = json.loads((advanced_stage / "results" / "config.json").read_text(encoding="utf-8"))
    assert config["baseline"]["manifest_sha256"] == expected["baseline/baseline_manifest.json"]
    baseline_config = json.loads((baseline_stage / "results" / "config.json").read_text(encoding="utf-8"))
    assert config["input"] == baseline_config["input"]
    for relative, digest in baseline_config["reviewed_source_sha256"].items():
        assert config["reviewed_source_sha256"][relative] == digest


@pytest.mark.release_artifact
def test_advanced_report_and_figures_exist(advanced_stage: Path):
    report = (advanced_stage / "reports" / "advanced_report.md").read_text(encoding="utf-8")
    for section in ["缺失值处理对比", "PCA 与原始特征选择", "校准与 Brier score", "消融实验", "SHAP 解释", "序数编码敏感性", "并列"]:
        assert section in report
    assert "winner" not in report.lower()
    for relative in RELEASE_OUTPUTS:
        if relative.endswith(".png"):
            assert (advanced_stage / relative).stat().st_size > 1_000


@pytest.mark.release_artifact
def test_seed_is_inherited_from_the_baseline_and_recorded_everywhere(baseline_stage: Path, advanced_stage: Path):
    baseline = _config(baseline_stage)["effective_configuration"]
    advanced = _config(advanced_stage)
    effective = advanced["effective_configuration"]
    seed = baseline["random_state"]
    assert isinstance(seed, int) and not isinstance(seed, bool)
    assert baseline["split"]["random_state"] == seed and baseline["cv"]["random_state"] == seed
    split = json.loads((baseline_stage / "results" / "split_manifest.json").read_text(encoding="utf-8"))
    assert split["random_state"] == seed
    assert advanced["baseline"]["random_state_inherited"] is True
    assert effective["random_state"] == seed
    assert effective["cv"]["random_state"] == seed
    assert effective["holdout"]["bootstrap"]["random_state"] == seed
    sensitivity = effective["seed_sensitivity"]
    assert sensitivity["main_seed"] == seed
    assert sensitivity["design"] == "full_replication_per_seed"
    assert sensitivity["changes_main_selection"] is False
    seeds = sensitivity["seeds"]
    assert seeds and len(set(seeds)) == len(seeds) and seed not in seeds
    assert all(isinstance(value, int) and not isinstance(value, bool) for value in seeds)
    assert sensitivity["scopes"] == list(SENSITIVITY_SCOPES)


@pytest.mark.release_artifact
def test_seed_sensitivity_replications_are_from_scratch_and_summarized(baseline_stage: Path, advanced_stage: Path):
    seed = _config(baseline_stage)["effective_configuration"]["random_state"]
    seeds = _sensitivity_seeds(advanced_stage)
    results = advanced_stage / "results"
    summary = pd.read_csv(results / "seed_sensitivity_summary.csv")
    assert list(summary.columns) == SUMMARY_COLUMNS
    validate_seed_sensitivity_summary(summary, main_seed=seed, seeds=seeds)

    main_selection = json.loads((results / "selection_manifest.json").read_text(encoding="utf-8"))
    main_comparison = pd.read_csv(results / "model_comparison_repeated_cv.csv")
    main_holdout = pd.read_csv(results / "holdout_evaluation.csv")
    main_split = pd.read_csv(baseline_stage / "results" / "split_assignments.csv")
    for scope_id in SENSITIVITY_SCOPES:
        row = summary[(summary["seed"] == seed) & (summary["scope_id"] == scope_id)].iloc[0]
        representative = get_display_representative(main_selection, scope_id)
        assert row["role"] == ROLE_MAIN and _key(row) == _key(representative)
        cv = main_comparison[[_key(r) == _key(representative) for _, r in main_comparison.iterrows()]].iloc[0]
        hold = main_holdout[[_key(r) == _key(representative) for _, r in main_holdout.iterrows()]].iloc[0]
        assert row["rcv_accuracy_mean"] == pytest.approx(float(cv["rcv_accuracy_mean"]))
        assert row["holdout_accuracy_observed"] == pytest.approx(float(hold["holdout_accuracy_observed"]))
        assert int(row["n_holdout"]) == int(hold["n_holdout"])

    registry_hashes = {digest for digest in main_comparison["registry_hash"]}
    for other in seeds:
        seed_dir = results / "seed_sensitivity" / f"seed_{other}"
        assert {path.name for path in seed_dir.iterdir()} == set(SEED_OUTPUT_FILES)
        split = json.loads((seed_dir / "split_manifest.json").read_text(encoding="utf-8"))
        assert split["random_state"] == other and split["test_size"] == 0.30
        assignments = pd.read_csv(seed_dir / "split_assignments.csv")
        assert len(assignments) == len(main_split) == 400
        assert set(assignments["row_id"]) == set(main_split["row_id"])
        assert not assignments.set_index("row_id")["assignment"].equals(main_split.set_index("row_id")["assignment"])
        assert (assignments["assignment"] == "development").sum() == 280
        for population, name in ((FULL_SAMPLE, "cv_split_registry_full.csv"), (COMPLETE_CASE, "cv_split_registry_complete_case.csv")):
            registry = read_split_registry_csv(seed_dir / name)
            identities = registry.drop_duplicates("row_id").set_index("row_id")["target"]
            digest = validate_split_registry(
                registry, identities, expected_population=population, expected_scheme=ADVANCED_SCHEME,
                expected_class_counts={int(key): int(value) for key, value in identities.value_counts().items()},
            )
            assert digest not in registry_hashes
            registry_hashes.add(digest)
        comparison = pd.read_csv(seed_dir / "model_comparison_repeated_cv.csv")
        folds = pd.read_csv(seed_dir / "repeated_cv_fold_scores.csv")
        assert len(comparison) == 81 and len(folds) == 81 * 15
        assert list(comparison.columns) == REPEATED_CV_SUMMARY_COLUMNS and list(folds.columns) == REPEATED_CV_FOLD_SCORE_COLUMNS
        selection = json.loads((seed_dir / "selection_manifest.json").read_text(encoding="utf-8"))
        assert selection["selection_uses_holdout"] is False and selection["source_candidate_count"] == 81
        holdout = pd.read_csv(seed_dir / "holdout_evaluation.csv")
        finalists = {_key(member) for member in get_selection_set(selection, "frozen_holdout_finalists")}
        assert {_key(row) for _, row in holdout.iterrows()} == finalists and len(holdout) == len(finalists)
        assert set(holdout["bootstrap"]) == {"not_performed_in_sensitivity_replication"}
        assert "n_bootstrap" not in holdout.columns
        for scope_id in SENSITIVITY_SCOPES:
            row = summary[(summary["seed"] == other) & (summary["scope_id"] == scope_id)].iloc[0]
            assert row["role"] == ROLE_SENSITIVITY
            assert _key(row) == _key(get_display_representative(selection, scope_id))
            main_key = _key(get_display_representative(main_selection, scope_id))
            scope = next(s for s in selection["ranked_scopes"] if s["scope_id"] == scope_id)
            record = next(r for r in scope["ranked_candidates"] if _key(r) == main_key)
            assert int(row["main_representative_dense_rank"]) == int(record["dense_rank"])
            assert bool(row["main_representative_jointly_top_ranked"]) == bool(record["jointly_top_ranked"])
            assert bool(row["same_representative_as_main"]) == (_key(row) == main_key)
