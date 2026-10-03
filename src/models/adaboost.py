"""AdaBoost model definition."""

from sklearn.ensemble import AdaBoostClassifier

from ..data_preprocessing import RANDOM_STATE


def build_model(random_state: int = RANDOM_STATE) -> AdaBoostClassifier:
    return AdaBoostClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=random_state,
    )

