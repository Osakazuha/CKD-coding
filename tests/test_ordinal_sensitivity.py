"""The single ordinal-encoding sensitivity branch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from src.data_preprocessing import (
    ORDINAL_SENSITIVITY_CATEGORIES,
    ORDINAL_SENSITIVITY_FEATURES,
    OrdinalSensitivityBranch,
    build_ordinal_sensitivity_encoder,
    build_preprocessor,
    load_ckd_data,
)
from src.train_models import build_model_pipeline


def test_encoder_uses_the_literal_canonical_lists_and_unknown_value():
    encoder = build_ordinal_sensitivity_encoder(["sg", "al", "su"])
    assert encoder.categories == [
        ["1.005", "1.01", "1.015", "1.02", "1.025"],
        ["0", "1", "2", "3", "4", "5"],
        ["0", "1", "2", "3", "4", "5"],
    ]
    assert encoder.handle_unknown == "use_encoded_value"
    assert encoder.unknown_value == -1
    assert ORDINAL_SENSITIVITY_FEATURES == ["sg", "al", "su"]
    assert ORDINAL_SENSITIVITY_CATEGORIES[0][-1] == "1.025"
    with pytest.raises(ValueError):
        build_ordinal_sensitivity_encoder(["sg", "htn"])


def test_absent_level_keeps_its_code_and_out_of_schema_maps_to_minus_one():
    frame = pd.DataFrame({"sg": ["1.01", "1.02", np.nan], "al": ["0", "3", "3"]}, dtype=object)
    branch = OrdinalSensitivityBranch(imputation="median_mode", scale=False).fit(frame)
    coded = branch.transform(pd.DataFrame({"sg": ["1.025", "1.005", "zzz"], "al": ["5", "0", "1"]}, dtype=object))
    assert coded[:, 0].tolist() == [4.0, 0.0, -1.0]
    assert coded[:, 1].tolist() == [5.0, 0.0, 1.0]
    assert branch.get_feature_names_out().tolist() == ["sg_ordinal", "al_ordinal"]


def test_ordinal_preprocessor_keeps_other_categoricals_nominal():
    X, y, _ = load_ckd_data("data/chronic_kidney_disease_full.xlsx")
    nominal = build_preprocessor("median_mode").fit(X.iloc[:200], y.iloc[:200])
    ordinal = build_preprocessor("median_mode", ordinal_sensitivity=True).fit(X.iloc[:200], y.iloc[:200])
    nominal_names = list(nominal.get_feature_names_out())
    ordinal_names = list(ordinal.get_feature_names_out())
    assert [n for n in ordinal_names if n.startswith("ord__")] == ["ord__sg_ordinal", "ord__al_ordinal", "ord__su_ordinal"]
    assert not any(n.startswith("cat__sg_") or n.startswith("cat__al_") or n.startswith("cat__su_") for n in ordinal_names)
    kept_nominal = [n for n in nominal_names if n.startswith("cat__") and not n.startswith(("cat__sg_", "cat__al_", "cat__su_"))]
    assert [n for n in ordinal_names if n.startswith("cat__")] == kept_nominal
    assert ordinal_names[: len([n for n in nominal_names if n.startswith("num__")])] == [
        n for n in nominal_names if n.startswith("num__")
    ]


def test_ordinal_pipeline_variant_fits_every_scenario_with_selector_active():
    X, y, _ = load_ckd_data("data/chronic_kidney_disease_full.xlsx")
    for scenario in ("A_all_original", "B_pca_95", "C_selected_original"):
        pipeline = build_model_pipeline(scenario, "knn", LogisticRegression(max_iter=2000), ordinal_sensitivity=True)
        fitted = clone(pipeline).fit(X.iloc[:280], y.iloc[:280])
        probabilities = fitted.predict_proba(X.iloc[280:])
        assert probabilities.shape == (120, 2) and np.isfinite(probabilities).all()
        if scenario == "C_selected_original":
            assert "feature_selection" in fitted.named_steps
            preprocessor = fitted.named_steps["preprocessor"]
            ordinal_columns = [columns for name, _, columns in preprocessor.transformers_ if name == "ord"][0]
            assert set(ordinal_columns) <= set(ORDINAL_SENSITIVITY_FEATURES)
            assert set(ordinal_columns) == set(fitted.named_steps["feature_selection"].selected_features_) & set(ORDINAL_SENSITIVITY_FEATURES)
    plain = clone(build_model_pipeline("A_all_original", "knn", LogisticRegression(max_iter=2000)))
    plain.fit(X.iloc[:280], y.iloc[:280])
    assert "ord" not in {name for name, _, _ in plain.named_steps["preprocessor"].transformers_}
