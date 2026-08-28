"""
Visualization Module
Author: Sahand Mostafaei
"""

import os

import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


def create_output_folder():
    os.makedirs("figures", exist_ok=True)


def plot_target_distribution(df):

    create_output_folder()

    plt.figure(figsize=(6, 4))

    df["Exited"].value_counts().sort_index().plot(
        kind="bar"
    )

    plt.title("Customer Churn Distribution")
    plt.xlabel("Exited")
    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig(
        "figures/churn_distribution.png"
    )

    plt.close()


def plot_confusion_matrix(
    y_test,
    predictions,
    model_name,
):

    create_output_folder()

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions
    )

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        f"figures/{filename}"
    )

    plt.close()


def plot_roc_curve(
    y_test,
    probabilities,
    model_name,
):

    create_output_folder()

    RocCurveDisplay.from_predictions(
        y_test,
        probabilities
    )

    plt.title(
        f"ROC Curve - {model_name}"
    )

    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_roc_curve.png"
    )

    plt.savefig(
        f"figures/{filename}"
    )

    plt.close()
