from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from .data import process_data
import numpy as np
import pandas as pd


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a Random Forest classifiers model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(random_state = 0).fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(X, model):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """

    preds = model.predict(X)

    return preds


def test_model_slices(test, X_test, y_test, feature, model):
    """ Function for calculating model performance metrics on slices of categorical variables."""
    
    performance = {}

    for cls in test[feature].unique():
        # Rows of data
        rows = test[test[feature] == cls].index.to_list()

        X = X_test.take(rows, axis=0)
        y = y_test.take(rows, axis=0)
        preds = inference(X, model)

        precision, recall, f1 = compute_model_metrics(y, preds)

        performance[cls] = {
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }

    return performance