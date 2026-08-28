"""
Bank Customer Churn Prediction
Author: Sahand Mostafaei
"""

from preprocessing import (
    load_data,
    clean_data,
    split_features_target,
)

from model import train_models

from visualization import (
    plot_target_distribution,
    plot_confusion_matrix,
    plot_roc_curve,
)

from explainability import feature_importance


def main():

    print("=" * 60)
    print("BANK CUSTOMER CHURN PREDICTION")
    print("=" * 60)

    # Load data
    df = load_data("data/bank_churn.csv")

    # Clean data
    df = clean_data(df)

    print("\nDataset Shape:")
    print(df.shape)

    print("\nFirst 5 Rows:")
    print(df.head())

    # Split features and target
    X, y, numerical_columns, categorical_columns = (
        split_features_target(df)
    )

    print("\nNumerical Features:")
    print(numerical_columns)

    print("\nCategorical Features:")
    print(categorical_columns)

    # Target visualization
    plot_target_distribution(df)

    print("\nTarget distribution saved.")

    # Train machine learning models
    results = train_models(
        X,
        y,
        numerical_columns,
        categorical_columns,
    )

    # Create model visualizations
    for name, result in results.items():

        plot_confusion_matrix(
            result["y_test"],
            result["predictions"],
            name,
        )

        plot_roc_curve(
            result["y_test"],
            result["probabilities"],
            name,
        )

    # Explain the Random Forest model
    if "Random Forest" in results:

        feature_importance(
            results["Random Forest"]["pipeline"]
        )

    # Model comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    for name, result in results.items():

        print(
            f"{name}: "
            f"Accuracy={result['accuracy']:.4f}, "
            f"Precision={result['precision']:.4f}, "
            f"Recall={result['recall']:.4f}, "
            f"F1={result['f1']:.4f}, "
            f"ROC-AUC={result['roc_auc']:.4f}"
        )

    print("\nVisualizations saved in figures folder.")
    print("\nProject executed successfully.")


if __name__ == "__main__":
    main()
