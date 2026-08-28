import pandas as pd

from preprocessing import (
    clean_data,
    split_features_target,
)

from model import train_models


def test_models_train_successfully():

    df = pd.DataFrame({
        "Age": [
            25, 30, 35, 40, 45,
            50, 55, 60, 28, 33,
            38, 43, 48, 53, 58,
            27, 32, 37, 42, 47,
        ],
        "Balance": [
            1000, 2000, 3000, 4000, 5000,
            6000, 7000, 8000, 1500, 2500,
            3500, 4500, 5500, 6500, 7500,
            1200, 2200, 3200, 4200, 5200,
        ],
        "Gender": [
            "Male", "Female", "Male", "Female", "Male",
            "Female", "Male", "Female", "Male", "Female",
            "Male", "Female", "Male", "Female", "Male",
            "Female", "Male", "Female", "Male", "Female",
        ],
        "Geography": [
            "France", "Germany", "Spain", "France", "Germany",
            "Spain", "France", "Germany", "Spain", "France",
            "Germany", "Spain", "France", "Germany", "Spain",
            "France", "Germany", "Spain", "France", "Germany",
        ],
        "Exited": [
            0, 1, 0, 1, 0,
            1, 0, 1, 0, 1,
            0, 1, 0, 1, 0,
            1, 0, 1, 0, 1,
        ],
    })

    df = clean_data(df)

    X, y, numerical_columns, categorical_columns = (
        split_features_target(df)
    )

    results = train_models(
        X,
        y,
        numerical_columns,
        categorical_columns,
    )

    assert "Logistic Regression" in results
    assert "Random Forest" in results
    assert "Gradient Boosting" in results

    for result in results.values():

        assert 0 <= result["accuracy"] <= 1
        assert 0 <= result["precision"] <= 1
        assert 0 <= result["recall"] <= 1
        assert 0 <= result["f1"] <= 1
        assert 0 <= result["roc_auc"] <= 1

        assert len(result["predictions"]) == len(
            result["y_test"]
        )

        assert len(result["probabilities"]) == len(
            result["y_test"]
        )
