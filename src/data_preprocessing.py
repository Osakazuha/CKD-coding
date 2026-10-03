"""Data loading, validation, cleaning, and preprocessing for the UCI CKD data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


RANDOM_STATE = 12
TARGET_COLUMN = "class"

ORDINAL_SENSITIVITY_FEATURES = ["sg", "al", "su"]
ORDINAL_SENSITIVITY_CATEGORIES = [
    ["1.005", "1.01", "1.015", "1.02", "1.025"],
    ["0", "1", "2", "3", "4", "5"],
    ["0", "1", "2", "3", "4", "5"],
]

NUMERIC_FEATURES = [
    "age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"
]
CATEGORICAL_FEATURES = [
    "sg", "al", "su", "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CATEGORY_LEVELS = {
    "sg": {"1.005", "1.01", "1.015", "1.02", "1.025"},
    "al": {"0", "1", "2", "3", "4", "5"},
    "su": {"0", "1", "2", "3", "4", "5"},
    "rbc": {"normal", "abnormal"},
    "pc": {"normal", "abnormal"},
    "pcc": {"present", "notpresent"},
    "ba": {"present", "notpresent"},
    "htn": {"yes", "no"},
    "dm": {"yes", "no"},
    "cad": {"yes", "no"},
    "appet": {"good", "poor"},
    "pe": {"yes", "no"},
    "ane": {"yes", "no"},
}

HEADER_ALIASES = {
    "wc": "wbcc",
    "wbc": "wbcc",
    "whitebloodcellcount": "wbcc",
    "rc": "rbcc",
    "rbc_count": "rbcc",
    "redbloodcellcount": "rbcc",
    "classification": "class",
    "target": "class",
}
LABEL_MAP = {"ckd": 1, "notckd": 0, "nonckd": 0}
MISSING_KEYS = {"", "?", "??", "nan", "na", "n/a", "none", "null", "missing"}


class MissingValueValidator(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Reject missing values while preserving the input representation and schema."""

    def _validate_missing_values(self, X) -> None:
        if np.asarray(pd.isna(X)).any():
            raise ValueError("missing values are not allowed for complete_case preprocessing")

    def fit(self, X, y=None):
        self._validate_missing_values(X)
        self.n_features_in_ = np.asarray(X).shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        self._validate_missing_values(X)
        return X


@dataclass
class DataAudit:
    source: str
    original_shape: tuple[int, int]
    cleaned_shape: tuple[int, int] = (0, 0)
    shifted_rows_repaired: int = 0
    dropped_extra_columns: list[str] = field(default_factory=list)
    numeric_strings_coerced_to_missing: dict[str, int] = field(default_factory=dict)
    invalid_categories_set_to_missing: dict[str, dict[str, int]] = field(default_factory=dict)
    invalid_targets_dropped: dict[str, int] = field(default_factory=dict)
    missing_by_feature: dict[str, int] = field(default_factory=dict)
    target_distribution: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _normalise_header(value: object) -> str:
    header = unicodedata.normalize("NFKC", str(value)).strip().lower()
    header = re.sub(r"[\s\-]+", "_", header)
    header = re.sub(r"[^a-z0-9_]", "", header).strip("_")
    return HEADER_ALIASES.get(header, header)


def _compact_key(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return re.sub(r"[\s_\-]+", "", text)


def _normalise_cell(value: object) -> object:
    if pd.isna(value):
        return np.nan
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str):
        return value
    text = unicodedata.normalize("NFKC", value).replace("\t", " ").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return np.nan if text in MISSING_KEYS else text


def find_data_file(data_dir: str | Path = "data") -> Path:
    """Find the CKD data deterministically without downloading anything."""
    data_dir = Path(data_dir)
    candidates: list[Path] = []
    for pattern in ("*.xlsx", "*.xls", "*.csv", "*.arff"):
        candidates.extend(sorted(data_dir.glob(pattern)))
    if not candidates:
        raise FileNotFoundError(
            f"No .xlsx/.xls/.csv/.arff dataset found in {data_dir.resolve()}"
        )
    preferred = [p for p in candidates if "kidney" in p.name.lower() or "ckd" in p.name.lower()]
    return (preferred or candidates)[0]


def _read_source(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        book = pd.ExcelFile(path)
        sheet = "CKD_Data" if "CKD_Data" in book.sheet_names else book.sheet_names[0]
        return pd.read_excel(path, sheet_name=sheet, dtype=object)
    if suffix == ".csv":
        return pd.read_csv(path, dtype=object)
    if suffix == ".arff":
        values, _ = arff.loadarff(path)
        return pd.DataFrame(values)
    raise ValueError(f"Unsupported data format: {path.suffix}")


def _repair_shifted_rows(df: pd.DataFrame, audit: DataAudit) -> pd.DataFrame:
    """Repair rows whose label was shifted into an Excel 'Unnamed' overflow column."""
    extras = [c for c in df.columns if c.startswith("unnamed") or c not in ALL_FEATURES + [TARGET_COLUMN]]
    if not extras:
        return df

    repaired = df.copy()
    ordered = [c for c in ALL_FEATURES + [TARGET_COLUMN] if c in repaired.columns]
    for extra in extras:
        extra_labels = repaired[extra].map(_compact_key)
        current_labels = repaired[TARGET_COLUMN].map(_compact_key)
        needs_repair = extra_labels.isin(LABEL_MAP) & ~current_labels.isin(LABEL_MAP)
        for idx in repaired.index[needs_repair]:
            missing_positions = [i for i, c in enumerate(ordered) if pd.isna(repaired.at[idx, c])]
            if not missing_positions:
                continue
            start = missing_positions[0]
            sequence = ordered[start:] + [extra]
            for left, right in zip(sequence[:-1], sequence[1:]):
                repaired.at[idx, left] = repaired.at[idx, right]
            audit.shifted_rows_repaired += 1

    audit.dropped_extra_columns = extras
    return repaired.drop(columns=extras)


def load_ckd_data(path: str | Path | None = None) -> tuple[pd.DataFrame, pd.Series, DataAudit]:
    """Load and clean CKD data, returning X, binary y (CKD=1), and an audit record."""
    source = Path(path) if path is not None else find_data_file()
    raw = _read_source(source)
    audit = DataAudit(source=str(source.resolve()), original_shape=raw.shape)

    df = raw.copy()
    df.columns = [_normalise_header(c) for c in df.columns]
    for col in df.columns:
        df[col] = df[col].map(_normalise_cell)
    df = _repair_shifted_rows(df, audit)

    missing_columns = sorted(set(ALL_FEATURES + [TARGET_COLUMN]) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    df = df[ALL_FEATURES + [TARGET_COLUMN]].copy()

    for col in NUMERIC_FEATURES:
        before = df[col].notna()
        converted = pd.to_numeric(df[col], errors="coerce")
        audit.numeric_strings_coerced_to_missing[col] = int((before & converted.isna()).sum())
        df[col] = converted.astype(float)

    for col in CATEGORICAL_FEATURES:
        compact = df[col].map(lambda v: np.nan if pd.isna(v) else _compact_key(v))
        if col in {"sg", "al", "su"}:
            compact = compact.map(
                lambda v: np.nan if pd.isna(v) else format(float(v), "g")
                if re.fullmatch(r"[-+]?\d*\.?\d+", str(v)) else v
            )
        invalid = compact.notna() & ~compact.isin(CATEGORY_LEVELS[col])
        if invalid.any():
            audit.invalid_categories_set_to_missing[col] = (
                compact[invalid].value_counts().astype(int).to_dict()
            )
            compact.loc[invalid] = np.nan
        df[col] = compact.astype(object).where(compact.notna(), np.nan)

    label_keys = df[TARGET_COLUMN].map(
        lambda v: np.nan if pd.isna(v) else _compact_key(v)
    )
    invalid_target = label_keys.notna() & ~label_keys.isin(LABEL_MAP)
    if invalid_target.any():
        audit.invalid_targets_dropped = label_keys[invalid_target].value_counts().astype(int).to_dict()
    valid = label_keys.isin(LABEL_MAP)
    df = df.loc[valid].copy()
    y = label_keys.loc[valid].map(LABEL_MAP).astype(int).rename("ckd")
    X = df[ALL_FEATURES].copy()

    audit.cleaned_shape = (len(X), len(ALL_FEATURES) + 1)
    audit.missing_by_feature = X.isna().sum().astype(int).to_dict()
    audit.target_distribution = y.value_counts().sort_index().astype(int).to_dict()
    return X, y, audit


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in NUMERIC_FEATURES if c in frame.columns]


def _categorical_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in CATEGORICAL_FEATURES if c in frame.columns]


def _nominal_columns(frame: pd.DataFrame) -> list[str]:
    return [
        c for c in CATEGORICAL_FEATURES
        if c in frame.columns and c not in ORDINAL_SENSITIVITY_FEATURES
    ]


def _ordinal_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in ORDINAL_SENSITIVITY_FEATURES if c in frame.columns]


def build_ordinal_sensitivity_encoder(columns: list[str]) -> OrdinalEncoder:
    """Return the literal ordered encoder used by the ordinal sensitivity analysis."""
    unknown = [column for column in columns if column not in ORDINAL_SENSITIVITY_FEATURES]
    if unknown:
        raise ValueError(f"ordinal sensitivity only covers {ORDINAL_SENSITIVITY_FEATURES}: {unknown}")
    lookup = dict(zip(ORDINAL_SENSITIVITY_FEATURES, ORDINAL_SENSITIVITY_CATEGORIES))
    return OrdinalEncoder(
        categories=[list(lookup[column]) for column in columns],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )


class OrdinalSensitivityBranch(BaseEstimator, TransformerMixin):
    """Fold-fitted ordered coding of sg/al/su (mode-imputed first), then scaling.

    The encoder categories are the loader's literal canonical strings, so a level
    absent from one training fold keeps its prescribed integer code; ``-1`` is
    reserved for out-of-schema values only.  Integer coding is an equal-spacing
    assumption for linear/distance models and an order constraint for trees.
    """

    def __init__(self, imputation: str = "median_mode", scale: bool = True):
        self.imputation = imputation
        self.scale = scale

    def fit(self, X, y=None):
        frame = pd.DataFrame(X).copy()
        columns = [str(column) for column in frame.columns]
        unknown = [column for column in columns if column not in ORDINAL_SENSITIVITY_FEATURES]
        if unknown:
            raise ValueError(f"ordinal sensitivity branch received non-ordinal columns: {unknown}")
        steps: list[tuple[str, object]] = []
        if self.imputation == "complete_case":
            steps.append(("missing_validator", MissingValueValidator()))
        else:
            steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
        steps.append(("ordinal", build_ordinal_sensitivity_encoder(columns)))
        if self.scale:
            steps.append(("scaler", StandardScaler()))
        self.columns_ = columns
        self.pipeline_ = Pipeline(steps).fit(frame.astype(object), y)
        self.n_features_in_ = len(columns)
        self.feature_names_in_ = np.asarray(columns, dtype=object)
        return self

    def transform(self, X):
        frame = pd.DataFrame(X)
        if [str(column) for column in frame.columns] != list(self.columns_):
            raise ValueError("ordinal sensitivity branch received different columns at transform")
        return self.pipeline_.transform(frame.astype(object))

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{column}_ordinal" for column in self.columns_], dtype=object)


def build_preprocessor(
    imputation: str,
    for_pca: bool = False,
    *,
    ordinal_sensitivity: bool = False,
) -> ColumnTransformer:
    """Build a train-fitted ColumnTransformer for one missing-data strategy.

    ``ordinal_sensitivity=True`` is reserved for the single prespecified ordinal
    sensitivity analysis: sg/al/su are mode-imputed inside the fold and then coded
    with their literal clinical order, while every other categorical predictor keeps
    the nominal one-hot treatment of the main analysis.
    """
    if imputation not in {"median_mode", "knn", "complete_case"}:
        raise ValueError(f"Unknown imputation strategy: {imputation}")

    if imputation == "median_mode":
        numeric_steps: list[tuple[str, object]] = [
            ("imputer", SimpleImputer(strategy="median")),
        ]
        if not for_pca:
            numeric_steps.append(("scaler", StandardScaler()))
        categorical_steps: list[tuple[str, object]] = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, drop=None
                ),
            ),
        ]
    elif imputation == "knn":
        numeric_steps = [
            ("scaler", StandardScaler()),
            ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
        ]
        categorical_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, drop=None
                ),
            ),
        ]
    else:
        numeric_steps = [("missing_validator", MissingValueValidator())]
        if not for_pca:
            numeric_steps.append(("scaler", StandardScaler()))
        categorical_steps = [
            ("missing_validator", MissingValueValidator()),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, drop=None
                ),
            ),
        ]

    numeric = Pipeline(numeric_steps)
    categorical = Pipeline(categorical_steps)
    if not ordinal_sensitivity:
        return ColumnTransformer(
            [("num", numeric, _numeric_columns), ("cat", categorical, _categorical_columns)],
            remainder="drop",
            verbose_feature_names_out=True,
        )
    ordinal = OrdinalSensitivityBranch(imputation=imputation, scale=not for_pca)
    return ColumnTransformer(
        [
            ("num", numeric, _numeric_columns),
            ("cat", categorical, _nominal_columns),
            ("ord", ordinal, _ordinal_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def complete_case_subset(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Delete rows with any missing input after the split, preserving index alignment."""
    mask = X[ALL_FEATURES].notna().all(axis=1)
    return X.loc[mask].copy(), y.loc[mask].copy()
