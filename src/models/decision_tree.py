"""Decision Tree model definition."""

from sklearn.tree import DecisionTreeClassifier

from ..data_preprocessing import RANDOM_STATE


def build_model(random_state: int = RANDOM_STATE) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=2,
        random_state=random_state,
    )

