"""Logistic Regression model definition."""

from sklearn.linear_model import LogisticRegression

from ..data_preprocessing import RANDOM_STATE


def build_model(random_state: int = RANDOM_STATE) -> LogisticRegression:
    return LogisticRegression(
        max_iter=5000,
        solver="liblinear",
        random_state=random_state,
    )

