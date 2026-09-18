# Bank Customer Churn Prediction

A machine-learning project that predicts customer churn using demographic, financial, and banking customer characteristics.

The project demonstrates the application of supervised classification methods to a banking customer-retention problem, with emphasis on probability-based prediction, model comparison, and interpretation.

## Overview

Customer churn is an important analytical problem for banks because customer attrition can affect customer lifetime value, revenue, and retention strategy.

This project develops a classification workflow to estimate the probability that a banking customer will churn.

The analysis covers:

- Data loading and cleaning
- Exploratory analysis
- Numerical and categorical feature handling
- Feature scaling
- Categorical encoding
- Classification modelling
- Model comparison
- ROC-AUC evaluation
- Churn probability estimation
- Feature importance analysis
- Data visualization

## Technology Stack

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib

## Machine Learning Workflow

Customer Data
      ↓
Data Cleaning
      ↓
Feature / Target Separation
      ↓
Train-Test Split
      ↓
Numerical Scaling + Categorical Encoding
      ↓
Classification Models
      ↓
Probability Prediction
      ↓
Model Evaluation
      ↓
Feature Importance
      ↓
Interpretation

## Dataset

The project uses customer-level banking data containing demographic, account, and financial characteristics.

Representative variables include:

- Customer age
- Geography
- Gender
- Credit score
- Tenure
- Account balance
- Number of products
- Estimated salary
- Credit card status
- Active membership status

The target variable is `Exited`, representing whether the customer churned.

## Data Processing

The preprocessing workflow:

- Removes duplicate observations
- Separates explanatory variables from the target
- Identifies numerical variables
- Identifies categorical variables
- Standardizes numerical variables
- One-hot encodes categorical variables
- Preserves consistent transformations between training and prediction

The machine-learning preprocessing is implemented using scikit-learn's `ColumnTransformer` and `Pipeline`.

## Machine Learning Models

The project compares three classification approaches.

### Logistic Regression

Logistic regression provides an interpretable baseline for binary classification and estimates the probability of customer churn.

### Random Forest

Random Forest is used to capture nonlinear relationships and interactions between customer characteristics.

### Gradient Boosting

Gradient Boosting is used as an additional nonlinear ensemble method for comparison with the linear baseline and Random Forest.

## Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC

ROC-AUC is particularly relevant because the model produces probability estimates and the analytical objective is to distinguish customers with different levels of estimated churn probability.

## Churn Probability

The classification models generate estimated churn probabilities for individual observations.

These probabilities can be used analytically to:

- Identify customers with elevated estimated churn risk
- Segment customers by predicted risk
- Examine characteristics associated with higher predicted churn probability
- Support hypothetical customer-retention analysis

The predictions are analytical outputs and should not be interpreted as production banking decisions.

## Explainability

Tree-based models provide feature-importance information that can be used to examine which variables contribute most strongly to model predictions.

This provides an additional interpretive layer beyond predictive performance metrics.

## Visualizations

The project produces visualizations covering areas such as:

- Churn distribution
- Model performance
- ROC curves
- Feature importance
- Customer-level prediction analysis

Generated figures are stored in the `figures/` directory.

## Project Structure

bank-customer-churn-prediction/
│
├── data/
│   └── dataset files
│
├── tests/
│   └── test_model.py
│
├── figures/
│   └── generated figures
│
├── preprocessing.py
├── model.py
├── visualization.py
├── explainability.py
├── main.py
├── requirements.txt
├── PROJECT.md
├── RESULTS.md
├── ROADMAP.md
└── README.md

## Source Modules

| Module | Purpose |
|---|---|
| `preprocessing.py` | Data loading, cleaning, and feature/target separation |
| `model.py` | Preprocessing pipelines, model training, prediction, and evaluation |
| `visualization.py` | Data and model visualizations |
| `explainability.py` | Feature-importance analysis |
| `main.py` | Main analytical workflow |

## Key Skills Demonstrated

- Python
- pandas
- NumPy
- scikit-learn
- Supervised machine learning
- Binary classification
- Probability estimation
- Data preprocessing
- Model evaluation
- Feature analysis
- Data visualization
- Financial and banking analytics

## Banking Application

The project illustrates how machine-learning methods can be applied to customer analytics in a banking context.

A hypothetical banking application could use predicted churn probabilities to:

1. Identify customers with elevated estimated churn risk.
2. Segment customers according to predicted risk.
3. Examine characteristics associated with customer attrition.
4. Develop hypothetical retention strategies.
5. Monitor model performance over time.

## Limitations

This project is an analytical and educational implementation rather than a production banking system.

Important practical considerations such as probability calibration, model monitoring, fairness assessment, temporal validation, economic cost functions, and deployment controls would require additional development before operational use.

## Disclaimer

This project is intended for educational, research, and portfolio purposes.

The model outputs should not be interpreted as financial advice or as a production banking decision system.
