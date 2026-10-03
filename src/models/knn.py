"""K-Nearest Neighbors model definition."""

from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier


def build_model(random_state: int | None = None) -> KNeighborsClassifier:
    del random_state
    return KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
    )
