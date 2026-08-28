"""
Data Preprocessing Module
Author: Sahand Mostafaei
"""

import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    df = df.drop_duplicates()
    return df


def split_features_target(df):
    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        exclude=["object", "category"]
    ).columns.tolist()

    return X, y, numerical_columns, categorical_columns
