# Bank Customer Churn Prediction

A machine learning project that predicts customer churn for a banking dataset using multiple classification models and evaluates their performance with business-relevant metrics.

## Project Overview

Customer churn is an important business problem for banks because retaining existing customers is generally more efficient than continuously acquiring new customers.

This project builds an end-to-end machine learning workflow to:

- Load and clean customer data
- Separate features and target variables
- Identify numerical and categorical features
- Apply appropriate preprocessing
- Train multiple classification models
- Evaluate model performance
- Compare models using several metrics
- Visualize classification performance
- Analyze feature importance

The project is implemented in Python using scikit-learn and follows a modular structure.

## Machine Learning Models

Three classification models are evaluated:

1. **Logistic Regression**
2. **Random Forest**
3. **Gradient Boosting**

Logistic Regression provides a simple linear baseline, while Random Forest and Gradient Boosting provide nonlinear tree-based approaches.

## Data Preprocessing

The preprocessing workflow includes:

- Duplicate removal
- Numerical feature identification
- Categorical feature identification
- Standard scaling for numerical variables
- One-hot encoding for categorical variables
- Handling of previously unseen categorical values
- Stratified train/test splitting

Preprocessing is implemented inside an `sklearn` pipeline to prevent data leakage between training and testing data.

## Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

ROC-AUC is calculated using predicted probabilities rather than hard class predictions.

This is particularly useful for churn prediction because the probability of churn can be used to prioritize customers for retention campaigns.

## Visualizations

The project generates:

- Customer churn distribution
- Confusion matrix for each model
- ROC curve for each model
- Random Forest feature importance

Generated figures are stored in the `figures/` directory.

## Explainability

Random Forest feature importance is used to identify the most influential features in the churn prediction model.

The analysis helps connect machine learning results with potential business questions such as:

- Which customer characteristics are associated with churn?
- Which factors may help identify high-risk customers?
- Where could customer-retention efforts be prioritized?

Feature importance should be interpreted as model-based association rather than proof of causality.

## Project Structure

```text
bank-customer-churn-prediction/
│
├── data/
│   └── bank_churn.csv
│
├── figures/
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_model.py
│
├── preprocessing.py
├── model.py
├── visualization.py
├── explainability.py
├── main.py
├── requirements.txt
├── README.md
├── PROJECT.md
├── LICENSE
└── .gitignore
