import pytest
import joblib
import numpy as np
import pandas as pd
from src.ml.model import compute_model_metrics, inference
from src.ml.data import process_data

# Setting-up data and model objects
@pytest.fixture
def data():
    df = pd.read_csv("./data/census.csv", nrows = 50)
    df.replace("?", np.nan, inplace = True)
    df.dropna(inplace = True)
    return df

@pytest.fixture
def model():
    return joblib.load("./model/model.pkl")

@pytest.fixture
def encoder():
    return joblib.load("./model/encoder.pkl")

@pytest.fixture
def lb():
    return joblib.load("./model/label_binarizer.pkl")

# Tests for functions
def test_compute_model_metrics():
    y = np.array([0, 1, 1, 0, 1])
    preds = np.array([1, 1, 1, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    # Testing data types of metrics
    assert type(precision) == np.float64
    assert type(recall) == np.float64
    assert type(fbeta) == np.float64

def test_process_data(data):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, y, enc, lb = process_data(
        data,
        cat_features,
        label = "salary",
        )
    # Testing size of data outputs
    assert data.shape[0] == X.shape[0]
    assert data.shape[1] - 1 <= X.shape[1]
    assert data["salary"].shape[0] == y.shape[0]

def test_inference(data, model, encoder, lb):
    predictions = inference(data.drop(columns = ["salary"]), model, encoder, lb)
    # Testing type and size of inference outputs
    assert predictions.shape[0] == data.shape[0]
    assert type(predictions) == np.ndarray