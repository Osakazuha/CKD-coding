"""Support Vector Machine model definition."""

from sklearn.svm import SVC

from ..data_preprocessing import RANDOM_STATE


def build_model(random_state: int = RANDOM_STATE) -> SVC:
    return SVC(
        C=2.0,
        kernel="rbf",
        probability=True,
        random_state=random_state,
    )

