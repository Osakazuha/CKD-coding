"""XGBoost model definition."""

from xgboost import XGBClassifier

from ..data_preprocessing import RANDOM_STATE


def build_model(random_state: int = RANDOM_STATE) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=1,
    )

