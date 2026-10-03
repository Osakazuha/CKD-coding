"""Gradient Boosting model definition."""

from sklearn.ensemble import GradientBoostingClassifier

from ..data_preprocessing import RANDOM_STATE


def build_model(random_state: int = RANDOM_STATE) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        random_state=random_state,
    )

