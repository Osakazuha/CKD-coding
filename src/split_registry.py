"""Train-population deterministic cross-validation split registries."""

from __future__ import annotations

from hashlib import sha256
import json
from numbers import Integral
from typing import Final
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from .data_preprocessing import RANDOM_STATE


FULL_SAMPLE: Final = "full_sample"
COMPLETE_CASE: Final = "complete_case"
BASELINE_SCHEME: Final = "baseline_stratified_5fold"
ADVANCED_SCHEME: Final = "advanced_repeated_stratified_5x3"
REGISTRY_COLUMNS: Final = [
    "cv_scheme", "population", "repeat_id", "fold_id", "role", "row_id", "target", "registry_hash",
]
_ASSIGNMENT_COLUMNS: Final = REGISTRY_COLUMNS[:-1]
_SCHEMA_VERSION: Final = "split_registry_v1"
_POPULATIONS: Final = {FULL_SAMPLE, COMPLETE_CASE}


@dataclass(frozen=True)
class RegistrySplit:
    cv_scheme: str
    population: str
    registry_hash: str
    repeat_id: int
    fold_id: int
    train_row_ids: tuple[object, ...]
    validation_row_ids: tuple[object, ...]


def _python(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def _typed_id(value: object) -> dict[str, object]:
    value = _python(value)
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("row IDs must be non-boolean integers or strings")
    return {"type": "int", "value": int(value)}


def _validate_y(y: pd.Series, population: str) -> None:
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series")
    if isinstance(y.index, pd.MultiIndex):
        raise TypeError("row IDs must be scalar non-boolean integers or strings")
    if not y.index.is_unique:
        raise ValueError("y must have a unique index")
    if y.index.hasnans:
        raise ValueError("y index must not contain null row IDs")
    for row_id in y.index:
        _typed_id(row_id)
    if y.isna().any():
        raise ValueError("y must not contain missing targets")
    targets = [_python(value) for value in y]
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in targets):
        raise TypeError("targets must be integer binary outcomes")
    if set(targets) != {0, 1}:
        raise ValueError("observed target classes must be exactly {0, 1}")
    if population not in _POPULATIONS:
        raise ValueError("population must be full_sample or complete_case")
    if min(targets.count(0), targets.count(1)) < 5:
        raise ValueError("each target class must contain at least five rows")


def _id_key(value: object) -> tuple[str, object]:
    token = _typed_id(value)
    return str(token["type"]), token["value"]


def _expected_counts(y: pd.Series, supplied: Mapping[int, int] | None) -> dict[int, int]:
    observed = {0: int((y == 0).sum()), 1: int((y == 1).sum())}
    if supplied is None:
        return observed
    if not isinstance(supplied, Mapping):
        raise TypeError("expected class counts must be a mapping")
    if any(
        isinstance(key, bool) or not isinstance(key, Integral)
        or isinstance(value, bool) or not isinstance(value, Integral)
        for key, value in supplied.items()
    ):
        raise TypeError("expected class counts require non-boolean integral keys and values")
    if set(supplied.keys()) != {0, 1}:
        raise ValueError("expected class counts must have exactly keys {0, 1}")
    exact = {0: supplied[0], 1: supplied[1]}
    if exact != observed:
        raise ValueError("expected class counts do not match the supplied population")
    return observed


def _validate_split_id_cells(registry: pd.DataFrame) -> None:
    for column in ("repeat_id", "fold_id"):
        if any(
            isinstance(_python(value), bool) or not isinstance(_python(value), Integral)
            for value in registry[column]
        ):
            raise TypeError(f"registry {column} values must be non-boolean integers")


def _record(row: dict[str, object]) -> dict[str, object]:
    return {
        "cv_scheme": str(row["cv_scheme"]), "population": str(row["population"]),
        "repeat_id": int(_python(row["repeat_id"])), "fold_id": int(_python(row["fold_id"])),
        "role": str(row["role"]), "row_id": _typed_id(row["row_id"]),
        "target": int(_python(row["target"])),
    }


def compute_registry_hash(registry: pd.DataFrame) -> str:
    """Compute the canonical row-order-independent registry identity digest."""
    if not isinstance(registry, pd.DataFrame):
        raise TypeError("registry must be a pandas DataFrame")
    if not set(_ASSIGNMENT_COLUMNS).issubset(registry.columns):
        raise ValueError("registry is missing assignment columns required for hashing")
    records = [_record(row) for row in registry[_ASSIGNMENT_COLUMNS].to_dict("records")]
    records.sort(key=lambda item: (
        item["cv_scheme"], item["population"], item["repeat_id"], item["fold_id"], item["role"],
        item["row_id"]["type"], str(item["row_id"]["value"]), item["target"],
    ))
    payload = {"schema_version": _SCHEMA_VERSION, "records": records}
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return sha256(text.encode("utf-8")).hexdigest()


def materialize_baseline_registry(
    y: pd.Series, *, population: str, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Materialize the declared shuffled five-fold assignments."""
    _validate_y(y, population)
    rows: list[dict[str, object]] = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    for fold_id, (_, validation_positions) in enumerate(splitter.split(y, y), start=1):
        validation = set(validation_positions)
        for position, (row_id, target) in enumerate(y.items()):
            rows.append({
                "cv_scheme": BASELINE_SCHEME, "population": population, "repeat_id": 1,
                "fold_id": fold_id, "role": "validation" if position in validation else "train",
                "row_id": _python(row_id), "target": int(_python(target)),
            })
    registry = pd.DataFrame(rows, columns=_ASSIGNMENT_COLUMNS)
    registry["registry_hash"] = compute_registry_hash(registry)
    return registry.loc[:, REGISTRY_COLUMNS]


def materialize_advanced_registry(
    y: pd.Series, *, population: str, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Materialize the declared three-repeat five-fold assignments."""
    _validate_y(y, population)
    rows: list[dict[str, object]] = []
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=random_state)
    for split_position, (_, validation_positions) in enumerate(splitter.split(y, y)):
        repeat_offset, fold_offset = divmod(split_position, 5)
        validation = set(validation_positions)
        for position, (row_id, target) in enumerate(y.items()):
            rows.append({
                "cv_scheme": ADVANCED_SCHEME, "population": population, "repeat_id": repeat_offset + 1,
                "fold_id": fold_offset + 1, "role": "validation" if position in validation else "train",
                "row_id": _python(row_id), "target": int(_python(target)),
            })
    registry = pd.DataFrame(rows, columns=_ASSIGNMENT_COLUMNS)
    registry["registry_hash"] = compute_registry_hash(registry)
    return registry.loc[:, REGISTRY_COLUMNS]


def validate_split_registry(
    registry: pd.DataFrame,
    y: pd.Series,
    *,
    expected_population: str,
    expected_scheme: str,
    expected_class_counts: Mapping[int, int] | None = None,
) -> str:
    """Validate a registry before a consumer fits a fold-specific estimator."""
    if not isinstance(registry, pd.DataFrame):
        raise TypeError("registry must be a pandas DataFrame")
    if list(registry.columns) != REGISTRY_COLUMNS:
        raise ValueError("registry must have the exact required schema and column order")
    if registry.empty:
        raise ValueError("registry must not be empty")
    _validate_y(y, expected_population)
    class_counts = _expected_counts(y, expected_class_counts)
    if expected_scheme not in {BASELINE_SCHEME, ADVANCED_SCHEME}:
        raise ValueError("expected scheme is not supported")
    populations = registry["population"].unique()
    schemes = registry["cv_scheme"].unique()
    if len(populations) != 1 or _python(populations[0]) != expected_population:
        raise ValueError("registry population does not match expected population")
    if len(schemes) != 1 or _python(schemes[0]) != expected_scheme:
        raise ValueError("registry scheme does not match expected scheme")
    _validate_split_id_cells(registry)
    expected_repeats = {1} if expected_scheme == BASELINE_SCHEME else {1, 2, 3}
    if set(registry["repeat_id"]) != expected_repeats:
        raise ValueError("registry repeat IDs do not match its scheme")
    if set(registry["fold_id"]) != {1, 2, 3, 4, 5}:
        raise ValueError("registry fold IDs must be exactly 1 through 5")
    known_ids = {_id_key(row_id) for row_id in y.index}
    targets = {_id_key(row_id): int(_python(target)) for row_id, target in y.items()}
    for row_id, target in zip(registry["row_id"], registry["target"]):
        key = _id_key(row_id)
        if key not in known_ids:
            raise ValueError("registry contains missing or foreign row IDs")
        target = _python(target)
        if isinstance(target, bool) or not isinstance(target, Integral) or int(target) != targets[key]:
            raise ValueError("registry target does not match y.loc[row_id]")
    if registry.duplicated(["repeat_id", "fold_id", "row_id"]).any():
        raise ValueError("registry contains duplicate repeat/fold/row assignments")
    if not set(registry["role"]).issubset({"train", "validation"}):
        raise ValueError("registry roles must be train or validation")
    for repeat_id in expected_repeats:
        for fold_id in range(1, 6):
            partition = registry.loc[
                (registry["repeat_id"] == repeat_id) & (registry["fold_id"] == fold_id)
            ]
            if len(partition) != len(y) or {_id_key(value) for value in partition["row_id"]} != known_ids:
                raise ValueError("each repeat/fold must be an exhaustive row-ID partition")
            train = {_id_key(value) for value in partition.loc[partition["role"] == "train", "row_id"]}
            validation = {_id_key(value) for value in partition.loc[partition["role"] == "validation", "row_id"]}
            if not train or not validation or train & validation or train | validation != known_ids:
                raise ValueError("each repeat/fold must be a disjoint train/validation partition")
        repeat = registry.loc[registry["repeat_id"] == repeat_id]
        for row_id in y.index:
            roles = repeat.loc[repeat["row_id"] == row_id, "role"]
            if int((roles == "validation").sum()) != 1 or int((roles == "train").sum()) != 4:
                raise ValueError("each row must be validation once and train four times per repeat")
        for target, total in class_counts.items():
            validation_counts = [
                int(((repeat["fold_id"] == fold_id) & (repeat["role"] == "validation") & (repeat["target"] == target)).sum())
                for fold_id in range(1, 6)
            ]
            lower, upper = total // 5, (total + 4) // 5
            if sum(validation_counts) != total or any(value not in {lower, upper} for value in validation_counts):
                raise ValueError("validation class counts are not stratified floor/ceiling allocations")
            train_counts = [
                int(((repeat["fold_id"] == fold_id) & (repeat["role"] == "train") & (repeat["target"] == target)).sum())
                for fold_id in range(1, 6)
            ]
            if train_counts != [total - value for value in validation_counts]:
                raise ValueError("train class counts are not complementary to validation counts")
    digests = registry["registry_hash"].unique()
    if len(digests) != 1:
        raise ValueError("registry digest must contain exactly one value")
    digest = _python(digests[0])
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("registry digest must be one lowercase SHA-256 string")
    if digest != compute_registry_hash(registry):
        raise ValueError("registry digest does not match its assignments")
    return digest


def iter_registry_splits(
    registry: pd.DataFrame,
    y: pd.Series,
    *,
    expected_population: str,
    expected_scheme: str,
    expected_class_counts: Mapping[int, int] | None = None,
) -> Iterator[RegistrySplit]:
    """Validate then yield train/validation IDs in repeat-major, fold-minor order."""
    digest = validate_split_registry(
        registry, y, expected_population=expected_population, expected_scheme=expected_scheme,
        expected_class_counts=expected_class_counts,
    )
    repeats = [1] if expected_scheme == BASELINE_SCHEME else [1, 2, 3]
    for repeat_id in repeats:
        for fold_id in range(1, 6):
            fold = registry.loc[(registry["repeat_id"] == repeat_id) & (registry["fold_id"] == fold_id)]
            roles = dict(zip(fold["row_id"], fold["role"]))
            yield RegistrySplit(
                cv_scheme=expected_scheme, population=expected_population, registry_hash=digest,
                repeat_id=repeat_id, fold_id=fold_id,
                train_row_ids=tuple(row_id for row_id in y.index if roles[row_id] == "train"),
                validation_row_ids=tuple(row_id for row_id in y.index if roles[row_id] == "validation"),
            )


def _same_split_population(registry: pd.DataFrame) -> object:
    if not isinstance(registry, pd.DataFrame) or list(registry.columns) != REGISTRY_COLUMNS or registry.empty:
        raise ValueError("registry must have the exact required schema and be nonempty")
    populations = registry["population"].unique()
    if len(populations) != 1:
        raise ValueError("registry must contain one population")
    return _python(populations[0])


def _same_split_scheme(registry: pd.DataFrame) -> object:
    schemes = registry["cv_scheme"].unique()
    if len(schemes) != 1:
        raise ValueError("registry must contain one scheme")
    return _python(schemes[0])


def _same_split_digest(registry: pd.DataFrame, scheme: object) -> str:
    _validate_split_id_cells(registry)
    expected_repeats = {1} if scheme == BASELINE_SCHEME else {1, 2, 3}
    if set(registry["repeat_id"]) != expected_repeats or set(registry["fold_id"]) != {1, 2, 3, 4, 5}:
        raise ValueError("registry repeat/fold IDs do not match its scheme")
    digests = registry["registry_hash"].unique()
    if len(digests) != 1:
        raise ValueError("registry must contain one digest")
    digest = _python(digests[0])
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("registry digest must be one lowercase SHA-256 string")
    if digest != compute_registry_hash(registry):
        raise ValueError("registry digest does not match its assignments")
    return digest


def assert_same_split_registry(left: pd.DataFrame, right: pd.DataFrame) -> str:
    """Return a common digest only for compatible, integrity-checked registries."""
    left_population = _same_split_population(left)
    right_population = _same_split_population(right)
    if left_population != right_population:
        raise ValueError("registries have different populations and cannot use the same splits")
    if left_population not in _POPULATIONS:
        raise ValueError("registry population is not supported")
    left_scheme = _same_split_scheme(left)
    right_scheme = _same_split_scheme(right)
    if left_scheme != right_scheme:
        raise ValueError("registries have different schemes")
    if left_scheme not in {BASELINE_SCHEME, ADVANCED_SCHEME}:
        raise ValueError("registry scheme is not supported")
    left_digest = _same_split_digest(left, left_scheme)
    right_digest = _same_split_digest(right, right_scheme)
    if left_digest != right_digest:
        raise ValueError("registries have different assignments or digests")
    return left_digest


def write_split_registry_csv(registry: pd.DataFrame, path) -> None:
    """Write registry IDs as reversible type-tagged JSON tokens in the row_id CSV cell."""
    if not isinstance(registry, pd.DataFrame) or list(registry.columns) != REGISTRY_COLUMNS:
        raise ValueError("registry must have the exact required schema and column order")
    encoded = registry.copy()
    encoded["row_id"] = encoded["row_id"].map(
        lambda value: json.dumps(_typed_id(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    encoded.to_csv(path, index=False)


def read_split_registry_csv(path) -> pd.DataFrame:
    """Read a typed split-registry CSV without letting CSV inference change real row IDs."""
    registry = pd.read_csv(path, dtype={"row_id": str})
    if list(registry.columns) != REGISTRY_COLUMNS:
        raise ValueError("registry CSV must have the exact required schema and column order")

    def decode(token: object) -> object:
        if not isinstance(token, str):
            raise TypeError("registry CSV row_id token must be text")
        try:
            decoded = json.loads(token)
        except json.JSONDecodeError as error:
            raise ValueError("registry CSV row_id token is not valid JSON") from error
        if not isinstance(decoded, dict) or set(decoded) != {"type", "value"}:
            raise ValueError("registry CSV row_id token has an invalid schema")
        if decoded["type"] == "str" and isinstance(decoded["value"], str):
            return decoded["value"]
        if decoded["type"] == "int" and not isinstance(decoded["value"], bool) and isinstance(decoded["value"], Integral):
            return int(decoded["value"])
        raise ValueError("registry CSV row_id token has an invalid type/value pair")

    registry["row_id"] = registry["row_id"].map(decode)
    return registry.loc[:, REGISTRY_COLUMNS]
