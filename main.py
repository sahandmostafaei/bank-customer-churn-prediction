"""
Bank Customer Churn Prediction
Author: Sahand Mostafaei
"""

from preprocessing import (
    load_data,
    clean_data,
)

from visualization import (
    plot_target_distribution,
)


def main():

    print("="*60)
    print("BANK CUSTOMER CHURN PREDICTION")
    print("="*60)

    df = load_data("data/bank_churn.csv")

    df = clean_data(df)

    print(df.head())

    print("\nDataset Shape")

    print(df.shape)

    plot_target_distribution(df)

    print("\nVisualization saved in figures folder.")

    print("\nProject executed successfully.")


if __name__ == "__main__":

    main()
