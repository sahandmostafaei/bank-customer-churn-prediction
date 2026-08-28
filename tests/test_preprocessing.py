import pandas as pd

from preprocessing import clean_data, split_features_target


def test_clean_data_removes_duplicates():

    df = pd.DataFrame({
        "Age": [30, 30, 40],
        "Gender": ["Male", "Male", "Female"],
        "Exited": [0, 0, 1],
    })

    cleaned = clean_data(df)

    assert len(cleaned) == 2


def test_split_features_target():

    df = pd.DataFrame({
        "Age": [30, 40],
        "Balance": [1000, 2000],
        "Gender": ["Male", "Female"],
        "Exited": [0, 1],
    })

    X, y, numerical_columns, categorical_columns = (
        split_features_target(df)
    )

    assert "Exited" not in X.columns
    assert len(y) == 2

    assert "Age" in numerical_columns
    assert "Balance" in numerical_columns

    assert "Gender" in categorical_columns
