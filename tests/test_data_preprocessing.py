from src.data_preprocessing import ALL_FEATURES, load_ckd_data


def test_actual_workbook_is_cleaned_and_repaired():
    X, y, audit = load_ckd_data("data/chronic_kidney_disease_full.xlsx")
    assert X.shape == (400, 24)
    assert list(X.columns) == ALL_FEATURES
    assert y.value_counts().to_dict() == {1: 250, 0: 150}
    assert audit.shifted_rows_repaired == 1
    assert X.notna().all(axis=1).sum() == 158

