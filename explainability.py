"""
Model Explainability Module
Author: Sahand Mostafaei
"""

import os

import matplotlib.pyplot as plt


def feature_importance(model_pipeline):
    """
    Create a feature-importance chart for a tree-based model
    inside an sklearn Pipeline.
    """

    model = model_pipeline.named_steps["model"]
    preprocessor = model_pipeline.named_steps["preprocessing"]

    if not hasattr(model, "feature_importances_"):
        print("This model does not provide feature importance.")
        return

    feature_names = preprocessor.get_feature_names_out()

    importance = model.feature_importances_

    if len(feature_names) != len(importance):
        print("Feature names and importance values do not match.")
        return

    importance_data = sorted(
        zip(feature_names, importance),
        key=lambda x: x[1],
        reverse=True,
    )

    top_features = importance_data[:15]

    names = [
        feature.replace("numerical__", "")
        .replace("categorical__", "")
        for feature, _ in top_features
    ]

    values = [
        value
        for _, value in top_features
    ]

    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.barh(
        names[::-1],
        values[::-1],
    )

    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")

    plt.tight_layout()

    plt.savefig(
        "figures/feature_importance.png"
    )

    plt.close()

    print(
        "Feature importance saved to "
        "figures/feature_importance.png"
    )
