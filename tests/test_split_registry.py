import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.data_preprocessing import complete_case_subset, load_ckd_data


ASSIGNMENT_COLUMNS = [
    "cv_scheme", "population", "repeat_id", "fold_id", "role", "row_id", "target",
]
REGISTRY_COLUMNS = ASSIGNMENT_COLUMNS + ["registry_hash"]


def test_canonical_hash_is_row_order_independent_and_distinguishes_id_types():
    """Breaks if the hash follows DataFrame order or treats integer 7 as string '7'."""
    from src import split_registry

    assignments = pd.DataFrame(
        [
            ["baseline_stratified_5fold", "full_sample", 1, 1, "train", 7, 0],
            ["baseline_stratified_5fold", "full_sample", 1, 1, "validation", "7", 1],
        ],
        columns=ASSIGNMENT_COLUMNS,
    )
    assert split_registry.compute_registry_hash(assignments) == split_registry.compute_registry_hash(
        assignments.iloc[::-1].reset_index(drop=True)
    )
    changed = assignments.copy()
    changed.loc[1, "row_id"] = 7
    assert split_registry.compute_registry_hash(assignments) != split_registry.compute_registry_hash(changed)
    assert (
        split_registry.FULL_SAMPLE,
        split_registry.COMPLETE_CASE,
        split_registry.BASELINE_SCHEME,
        split_registry.ADVANCED_SCHEME,
    ) == (
        "full_sample",
        "complete_case",
        "baseline_stratified_5fold",
        "advanced_repeated_stratified_5x3",
    )


def test_baseline_registry_has_literal_fixed_five_fold_validation_assignments():
    """Breaks if baseline stops emitting the declared fixed five validation folds."""
    from src import split_registry

    y = pd.Series([0] * 5 + [1] * 5, index=["id-7", 41, "id-2", 19, "id-9", "id-1", 88, "id-8", 73, "id-4"])
    registry = split_registry.materialize_baseline_registry(y, population="full_sample", random_state=17)
    expected_validation = [(41, 73), ("id-9", "id-4"), ("id-2", "id-1"), (19, 88), ("id-7", "id-8")]

    assert list(registry.columns) == REGISTRY_COLUMNS
    assert len(registry) == 50
    assert registry["registry_hash"].str.fullmatch(r"[0-9a-f]{64}").all()
    for fold_id, expected_ids in enumerate(expected_validation, start=1):
        fold = registry.loc[registry["fold_id"] == fold_id]
        assert tuple(fold.loc[fold["role"] == "validation", "row_id"]) == expected_ids
        assert tuple(fold["row_id"]) == tuple(y.index)
        assert set(fold["role"]) == {"train", "validation"}
        assert set(fold["repeat_id"]) == {1}


def test_advanced_registry_has_literal_fixed_three_by_five_validation_assignments():
    """Breaks if repeated CV stops emitting the declared 15 validation folds."""
    from src import split_registry

    y = pd.Series([0] * 5 + [1] * 5, index=["id-7", 41, "id-2", 19, "id-9", "id-1", 88, "id-8", 73, "id-4"])
    registry = split_registry.materialize_advanced_registry(y, population="full_sample", random_state=23)
    expected_validation = [
        ("id-2", "id-4"), (41, "id-8"), (19, "id-1"), ("id-9", 73), ("id-7", 88),
        ("id-7", 88), (41, "id-8"), ("id-2", 73), ("id-9", "id-4"), (19, "id-1"),
        ("id-7", "id-4"), ("id-9", "id-8"), (19, "id-1"), (41, 88), ("id-2", 73),
    ]

    assert len(registry) == 150
    for position, expected_ids in enumerate(expected_validation):
        repeat_id, fold_offset = divmod(position, 5)
        fold = registry.loc[(registry["repeat_id"] == repeat_id + 1) & (registry["fold_id"] == fold_offset + 1)]
        assert tuple(fold.loc[fold["role"] == "validation", "row_id"]) == expected_ids
        assert tuple(fold["row_id"]) == tuple(y.index)


@pytest.mark.parametrize(
    ("y", "population", "error"),
    [
        (pd.Series([0] * 5 + [1] * 5, index=[0, 0, 2, 3, 4, 5, 6, 7, 8, 9]), "full_sample", "unique"),
        (pd.Series([0] * 5 + [1] * 5, index=[None, 1, 2, 3, 4, 5, 6, 7, 8, 9]), "full_sample", "null"),
        (pd.Series([0] * 5 + [1] * 5, index=[0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9]), "full_sample", "row IDs"),
        (pd.Series([0] * 5 + [1] * 5, index=[False, 1, 2, 3, 4, 5, 6, 7, 8, 9]), "full_sample", "row IDs"),
        (pd.Series([0] * 5 + [1] * 4 + [None], index=range(10)), "full_sample", "missing"),
        (pd.Series([0] * 5 + [2] * 5, index=range(10)), "full_sample", "classes"),
        (pd.Series([0] * 5 + [1] * 5, index=range(10), dtype=float), "full_sample", "integer"),
        (pd.Series([0] * 4 + [1] * 5, index=range(9)), "full_sample", "five"),
        (pd.Series([0] * 5 + [1] * 5, index=range(10)), "other", "population"),
    ],
)
def test_construction_rejects_each_target_and_population_contract_violation(y, population, error):
    """Breaks if malformed input reaches a splitter instead of its contract check."""
    from src import split_registry

    with pytest.raises((TypeError, ValueError), match=error):
        split_registry.materialize_baseline_registry(y, population=population)


def test_construction_rejects_nonseries_and_composite_row_id_inputs():
    """Regression: the y boundary admits only a scalar-ID pandas Series."""
    from src import split_registry

    with pytest.raises(TypeError, match="Series"):
        split_registry.materialize_baseline_registry([0, 1], population="full_sample")
    composite = pd.Series([0] * 5 + [1] * 5, index=pd.MultiIndex.from_tuples([(0, value) for value in range(10)]))
    with pytest.raises(TypeError, match="row IDs"):
        split_registry.materialize_baseline_registry(composite, population="full_sample")


def _y() -> pd.Series:
    return pd.Series([0] * 5 + [1] * 5, index=["p0", 1, "p2", 3, "p4", 5, "p6", 7, "p8", 9])


def _valid_registry():
    from src import split_registry

    y = _y()
    return y, split_registry.materialize_baseline_registry(y, population="full_sample", random_state=17)


def _rehash(registry: pd.DataFrame) -> pd.DataFrame:
    from src import split_registry

    changed = registry.copy()
    changed.loc[:, "registry_hash"] = split_registry.compute_registry_hash(changed)
    return changed


def test_validator_first_rejects_wrong_schema_order_and_empty_table():
    """Breaks if malformed registry tables reach semantic validation."""
    from src import split_registry

    y, registry = _valid_registry()
    assert split_registry.validate_split_registry(
        registry, y, expected_population="full_sample", expected_scheme="baseline_stratified_5fold"
    ) == registry["registry_hash"].iat[0]
    with pytest.raises(ValueError, match="schema"):
        split_registry.validate_split_registry(
            registry.loc[:, REGISTRY_COLUMNS[::-1]], y,
            expected_population="full_sample", expected_scheme="baseline_stratified_5fold",
        )
    with pytest.raises(ValueError, match="empty"):
        split_registry.validate_split_registry(
            registry.iloc[0:0], y, expected_population="full_sample", expected_scheme="baseline_stratified_5fold"
        )


def test_validator_rejects_missing_or_foreign_real_ids_with_a_fresh_hash():
    """Breaks if a recomputed digest can conceal a row-ID substitution."""
    from src import split_registry

    y, registry = _valid_registry()
    changed = registry.copy()
    changed.loc[0, "row_id"] = "foreign-person"
    changed = _rehash(changed)
    with pytest.raises(ValueError, match="row IDs"):
        split_registry.validate_split_registry(
            changed, y, expected_population="full_sample", expected_scheme="baseline_stratified_5fold"
        )


def test_validator_rejects_duplicate_repeat_fold_row_assignments_with_a_fresh_hash():
    """Breaks if a duplicate triple can hide the missing real row in a fold."""
    from src import split_registry

    y, registry = _valid_registry()
    changed = registry.copy()
    changed.loc[0, "row_id"] = changed.loc[1, "row_id"]
    changed.loc[0, "target"] = changed.loc[1, "target"]
    changed = _rehash(changed)
    with pytest.raises(ValueError, match="duplicate"):
        split_registry.validate_split_registry(
            changed, y, expected_population="full_sample", expected_scheme="baseline_stratified_5fold"
        )


def test_validator_rejects_repeat_fold_domain_and_nonexhaustive_partitions_with_fresh_hashes():
    """Breaks if fold labels or per-fold row coverage are only implied by a digest."""
    from src import split_registry

    y, registry = _valid_registry()
    wrong_repeat = registry.copy()
    wrong_repeat.loc[0, "repeat_id"] = 2
    with pytest.raises(ValueError, match="repeat"):
        split_registry.validate_split_registry(
            _rehash(wrong_repeat), y, expected_population="full_sample", expected_scheme="baseline_stratified_5fold"
        )
    missing_row = registry.drop(index=0).reset_index(drop=True)
    with pytest.raises(ValueError, match="exhaustive"):
        split_registry.validate_split_registry(
            _rehash(missing_row), y, expected_population="full_sample", expected_scheme="baseline_stratified_5fold"
        )


def test_validator_rejects_an_out_of_domain_fold_id_with_a_fresh_hash():
    """Regression: every fold ID must be one of the five declared one-based values."""
    from src import split_registry

    y, registry = _valid_registry()
    changed = registry.copy()
    changed.loc[0, "fold_id"] = 6
    with pytest.raises(ValueError, match="fold"):
        split_registry.validate_split_registry(
            _rehash(changed), y, expected_population="full_sample", expected_scheme="baseline_stratified_5fold"
        )


@pytest.mark.parametrize(
    ("column", "value"),
    [("repeat_id", True), ("repeat_id", 1.0), ("fold_id", True), ("fold_id", 1.0)],
)
def test_validator_rejects_nonintegral_or_boolean_repeat_fold_cells_with_a_fresh_hash(column, value):
    """Breaks if Python equality lets True or 1.0 impersonate an integer split ID."""
    from src import split_registry

    y, registry = _valid_registry()
    changed = registry.copy()
    changed[column] = changed[column].astype(object)
    changed.loc[0, column] = value
    with pytest.raises(TypeError, match=column):
        split_registry.validate_split_registry(
            _rehash(changed), y, expected_population="full_sample", expected_scheme="baseline_stratified_5fold"
        )


def test_validator_rejects_target_mismatch_with_a_fresh_hash():
    """Breaks if a digest-valid row can store an outcome different from y.loc[row_id]."""
    from src import split_registry

    y, registry = _valid_registry()
    changed = registry.copy()
    changed.loc[0, "target"] = 1 - changed.loc[0, "target"]
    with pytest.raises(ValueError, match="target"):
        split_registry.validate_split_registry(
            _rehash(changed), y, expected_population="full_sample", expected_scheme="baseline_stratified_5fold"
        )


@pytest.mark.parametrize(
    "counts",
    [
        {0.9: 5.9, 1.9: 5.9},
        {"0": "5", "1": "5"},
        {False: 5, 1: 5},
        {0: True, 1: 5},
        {0: 5, 1: 6},
    ],
)
def test_validator_requires_exact_integral_binary_expected_class_counts(counts):
    """Breaks if expected counts are silently normalized by int()."""
    from src import split_registry

    y, registry = _valid_registry()
    with pytest.raises((TypeError, ValueError), match="expected class counts"):
        split_registry.validate_split_registry(
            registry, y, expected_population="full_sample", expected_scheme="baseline_stratified_5fold",
            expected_class_counts=counts,
        )


def test_validator_rejects_an_invalid_role_with_a_fresh_hash():
    """Breaks if a digest-valid assignment has a role outside train/validation."""
    from src import split_registry

    y, registry = _valid_registry()
    changed = registry.copy()
    changed.loc[0, "role"] = "test"
    with pytest.raises(ValueError, match="roles"):
        split_registry.validate_split_registry(
            _rehash(changed), y, expected_population="full_sample", expected_scheme="baseline_stratified_5fold"
        )


def test_validator_rejects_validation_more_than_once_with_a_fresh_hash():
    """Breaks if a row can be validation twice within a repeat."""
    from src import split_registry

    y, registry = _valid_registry()
    changed = registry.copy()
    row = changed.index[(changed["fold_id"] == 1) & (changed["role"] == "train")][0]
    changed.loc[row, "role"] = "validation"
    with pytest.raises(ValueError, match="validation once"):
        split_registry.validate_split_registry(
            _rehash(changed), y, expected_population="full_sample", expected_scheme="baseline_stratified_5fold"
        )


def test_validator_rejects_an_exhaustive_frequency_valid_but_nonstratified_allocation():
    """Breaks if preserving row frequency is mistaken for per-class stratification."""
    from src import split_registry

    y, registry = _valid_registry()
    changed = registry.copy()
    fold1_validation_one = changed.index[(changed["fold_id"] == 1) & (changed["role"] == "validation") & (changed["target"] == 1)][0]
    fold1_train_zero = changed.index[(changed["fold_id"] == 1) & (changed["role"] == "train") & (changed["target"] == 0) & (changed["row_id"] == changed.loc[(changed["fold_id"] == 2) & (changed["role"] == "validation") & (changed["target"] == 0), "row_id"].iat[0])][0]
    fold2_validation_zero = changed.index[(changed["fold_id"] == 2) & (changed["role"] == "validation") & (changed["target"] == 0)][0]
    fold2_train_one = changed.index[(changed["fold_id"] == 2) & (changed["role"] == "train") & (changed["target"] == 1) & (changed["row_id"] == changed.loc[(changed["fold_id"] == 1) & (changed["role"] == "validation") & (changed["target"] == 1), "row_id"].iat[0])][0]
    changed.loc[[fold1_validation_one, fold2_validation_zero], "role"] = "train"
    changed.loc[[fold1_train_zero, fold2_train_one], "role"] = "validation"
    with pytest.raises(ValueError, match="stratified"):
        split_registry.validate_split_registry(
            _rehash(changed), y, expected_population="full_sample", expected_scheme="baseline_stratified_5fold"
        )


@pytest.mark.parametrize("replacement", ["0" * 64, "UPPER" * 12 + "ABCD"])
def test_validator_rejects_forged_or_mixed_stored_digests(replacement):
    """Breaks if the validator returns an unchecked stored hash."""
    from src import split_registry

    y, registry = _valid_registry()
    changed = registry.copy()
    changed.loc[0, "registry_hash"] = replacement
    with pytest.raises(ValueError, match="digest"):
        split_registry.validate_split_registry(
            changed, y, expected_population="full_sample", expected_scheme="baseline_stratified_5fold"
        )


def test_validator_rejects_a_single_malformed_digest_value():
    """Regression: an all-row malformed digest must fail before its assignment hash is trusted."""
    from src import split_registry

    y, registry = _valid_registry()
    changed = registry.copy()
    changed.loc[:, "registry_hash"] = "UPPER" * 12 + "ABCD"
    with pytest.raises(ValueError, match="lowercase"):
        split_registry.validate_split_registry(
            changed, y, expected_population="full_sample", expected_scheme="baseline_stratified_5fold"
        )


def test_iteration_validates_then_yields_target_index_order_after_registry_shuffle():
    """Breaks if consumers inherit physical registry order rather than y.index order."""
    from src import split_registry

    y, registry = _valid_registry()
    shuffled = registry.sample(frac=1, random_state=99).reset_index(drop=True)
    splits = list(split_registry.iter_registry_splits(
        shuffled, y, expected_population="full_sample", expected_scheme="baseline_stratified_5fold"
    ))
    assert [(item.repeat_id, item.fold_id) for item in splits] == [(1, fold_id) for fold_id in range(1, 6)]
    first = splits[0]
    assert first.train_row_ids == tuple(row_id for row_id in y.index if row_id in set(first.train_row_ids))
    assert first.validation_row_ids == tuple(row_id for row_id in y.index if row_id in set(first.validation_row_ids))


def test_same_split_registry_requires_population_scheme_and_digest_compatibility():
    """Breaks if different populations can be labelled as using the same splits."""
    from src import split_registry

    y, baseline = _valid_registry()
    shuffled = baseline.sample(frac=1, random_state=3).reset_index(drop=True)
    assert split_registry.assert_same_split_registry(baseline, shuffled) == baseline["registry_hash"].iat[0]
    advanced = split_registry.materialize_advanced_registry(y, population="full_sample", random_state=23)
    with pytest.raises(ValueError, match="scheme"):
        split_registry.assert_same_split_registry(baseline, advanced)
    complete = split_registry.materialize_baseline_registry(y, population="complete_case", random_state=17)
    with pytest.raises(ValueError, match="population"):
        split_registry.assert_same_split_registry(baseline, complete)
    altered = baseline.copy()
    altered.loc[0, "role"] = "validation"
    with pytest.raises(ValueError, match="digest|assignments"):
        split_registry.assert_same_split_registry(baseline, altered)


def test_same_split_population_mismatch_precedes_a_forged_digest():
    """Breaks if digest checking masks the mandatory population-compatibility error."""
    from src import split_registry

    y, full = _valid_registry()
    complete = split_registry.materialize_baseline_registry(y, population="complete_case", random_state=17)
    complete.loc[0, "registry_hash"] = "0" * 64
    with pytest.raises(ValueError, match="populations"):
        split_registry.assert_same_split_registry(full, complete)


def test_same_split_population_mismatch_precedes_other_right_side_metadata_failures():
    """Breaks if scheme inspection occurs before the mandatory population boundary."""
    from src import split_registry

    y, full = _valid_registry()
    complete = split_registry.materialize_baseline_registry(y, population="complete_case", random_state=17)
    complete.loc[0, "cv_scheme"] = "other_scheme"
    with pytest.raises(ValueError, match="populations"):
        split_registry.assert_same_split_registry(full, complete)


def test_same_split_rejects_identically_rehashed_unsupported_population_or_scheme():
    """Breaks if matching arbitrary metadata can be described as a compatible registry."""
    from src import split_registry

    _, baseline = _valid_registry()
    unsupported_population = baseline.copy()
    unsupported_population.loc[:, "population"] = "other_population"
    unsupported_population = _rehash(unsupported_population)
    with pytest.raises(ValueError, match="population.*supported"):
        split_registry.assert_same_split_registry(unsupported_population, unsupported_population.copy())
    unsupported_scheme = baseline.copy()
    unsupported_scheme.loc[:, "cv_scheme"] = "other_scheme"
    unsupported_scheme = _rehash(unsupported_scheme)
    with pytest.raises(ValueError, match="scheme.*supported"):
        split_registry.assert_same_split_registry(unsupported_scheme, unsupported_scheme.copy())


def test_typed_csv_round_trip_preserves_numeric_strings_mixed_ids_schema_and_hash(tmp_path):
    """Breaks if CSV inference turns accepted string ID '0' into integer 0."""
    from src import split_registry

    y = pd.Series([0] * 5 + [1] * 5, index=["0", 1, "2", 3, "4", 5, "6", 7, "8", 9])
    y_before = y.copy(deep=True)
    registry = split_registry.materialize_baseline_registry(y, population="full_sample", random_state=17)
    registry_before = registry.copy(deep=True)
    path = tmp_path / "typed-splits.csv"
    split_registry.write_split_registry_csv(registry, path)
    restored = split_registry.read_split_registry_csv(path)

    pd.testing.assert_series_equal(y, y_before)
    pd.testing.assert_frame_equal(registry, registry_before)
    assert list(restored.columns) == REGISTRY_COLUMNS
    assert list(restored["row_id"]) == list(registry["row_id"])
    assert [type(value) for value in restored["row_id"]] == [type(value) for value in registry["row_id"]]
    assert split_registry.validate_split_registry(
        restored, y, expected_population="full_sample", expected_scheme="baseline_stratified_5fold"
    ) == registry["registry_hash"].iat[0]


def test_cleaned_workbook_has_four_deterministic_distinct_train_only_registries():
    """Regression: both schemes must be deterministic and population-bound after the fixed split."""
    from src import split_registry

    X, y, _ = load_ckd_data("data/chronic_kidney_disease_full.xlsx")
    X_train, _, y_train, y_holdout = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    _, complete_y_train = complete_case_subset(X_train, y_train)
    assert (len(y_train), y_train.value_counts().sort_index().to_dict()) == (280, {0: 105, 1: 175})
    assert (len(complete_y_train), complete_y_train.value_counts().sort_index().to_dict()) == (108, {0: 78, 1: 30})

    specifications = [
        (y_train, "full_sample", "baseline_stratified_5fold", split_registry.materialize_baseline_registry),
        (y_train, "full_sample", "advanced_repeated_stratified_5x3", split_registry.materialize_advanced_registry),
        (complete_y_train, "complete_case", "baseline_stratified_5fold", split_registry.materialize_baseline_registry),
        (complete_y_train, "complete_case", "advanced_repeated_stratified_5x3", split_registry.materialize_advanced_registry),
    ]
    hashes = set()
    for population_y, population, scheme, constructor in specifications:
        first = constructor(population_y, population=population)
        second = constructor(population_y, population=population)
        pd.testing.assert_frame_equal(first, second)
        assert first["registry_hash"].iat[0] == second["registry_hash"].iat[0]
        assert split_registry.validate_split_registry(
            first, population_y, expected_population=population, expected_scheme=scheme,
            expected_class_counts=population_y.value_counts().sort_index().to_dict(),
        ) == first["registry_hash"].iat[0]
        assert set(first["row_id"]).isdisjoint(set(y_holdout.index))
        hashes.add(first["registry_hash"].iat[0])
    assert len(hashes) == 4
