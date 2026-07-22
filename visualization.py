"""
Visualization Module
Author: Sahand Mostafaei
"""

import os
import matplotlib.pyplot as plt


def create_output_folder():
    os.makedirs("figures", exist_ok=True)


def plot_target_distribution(df):

    create_output_folder()

    plt.figure(figsize=(6,4))

    df["Exited"].value_counts().plot(kind="bar")

    plt.title("Customer Churn Distribution")

    plt.xlabel("Exited")

    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig("figures/churn_distribution.png")

    plt.close()
