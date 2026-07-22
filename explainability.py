"""
Explainability Module
Author: Sahand Mostafaei
"""

import matplotlib.pyplot as plt


def feature_importance(model, X):

    if hasattr(model, "feature_importances_"):

        importance = model.feature_importances_

        plt.figure(figsize=(10,5))

        plt.bar(X.columns, importance)

        plt.xticks(rotation=45)

        plt.title("Feature Importance")

        plt.tight_layout()

        plt.savefig("figures/feature_importance.png")

        plt.close()
