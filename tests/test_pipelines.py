from sklearn.base import clone
from sklearn.model_selection import train_test_split

from src.data_preprocessing import load_ckd_data
from src.train_models import build_model_pipeline, get_models


def test_all_three_scenarios_fit_without_leakage_breakage():
    X, y, _ = load_ckd_data("data/chronic_kidney_disease_full.xlsx")
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    estimator = get_models()["Logistic Regression"]
    for scenario in ["A_all_original", "B_pca_95", "C_selected_original"]:
        model = build_model_pipeline(scenario, "median_mode", clone(estimator))
        model.fit(X_train, y_train)
        assert model.predict(X_test).shape == (120,)
    assert len(model.named_steps["feature_selection"].selected_features_) == 8


def test_nine_models_are_exposed_from_separate_factories():
    models = get_models()
    assert list(models) == [
        "Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN",
        "AdaBoost", "Gradient Boosting", "Extra Trees", "XGBoost",
    ]
    assert len({type(model).__name__ for model in models.values()}) == 9


def test_scenario_c_selector_follows_the_run_seed():
    from sklearn.linear_model import LogisticRegression

    from src.data_preprocessing import RANDOM_STATE

    default = build_model_pipeline("C_selected_original", "median_mode", LogisticRegression())
    seeded = build_model_pipeline("C_selected_original", "knn", LogisticRegression(), random_state=42)
    assert default.named_steps["feature_selection"].random_state == RANDOM_STATE
    assert seeded.named_steps["feature_selection"].random_state == 42
    assert "feature_selection" not in build_model_pipeline(
        "A_all_original", "median_mode", LogisticRegression(), random_state=42
    ).named_steps
