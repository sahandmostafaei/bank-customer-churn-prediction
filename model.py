"""
Machine Learning Models
Author: Sahand Mostafaei
"""

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)


def train_models(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    logistic = LogisticRegression(max_iter=1000)

    logistic.fit(X_train, y_train)

    rf = RandomForestClassifier(
        random_state=42
    )

    rf.fit(X_train, y_train)

    for model in [logistic, rf]:

        pred = model.predict(X_test)

        print(model.__class__.__name__)

        print("Accuracy:", accuracy_score(y_test, pred))

        print("ROC AUC:", roc_auc_score(y_test, pred))

        print(classification_report(y_test, pred))
