import pytest
import joblib
import numpy as np
import pandas as pd
from src.ml.model import compute_model_metrics, inference
from src.ml.data import process_data

from fastapi.testclient import TestClient
import json
from main import app

# Instantiate the testing client with our app
client = TestClient(app)

# Setting-up data and model objects
@pytest.fixture
def raw_data():
    df = pd.read_csv("./data/census.csv", nrows = 50)
    df.replace("?", np.nan, inplace = True)
    df.dropna(inplace = True)
    return df

@pytest.fixture
def model():
    return joblib.load("./output/model.pkl")

@pytest.fixture
def encoder():
    return joblib.load("./output/encoder.pkl")

@pytest.fixture
def lb():
    return joblib.load("./output/label_binarizer.pkl")

@pytest.fixture
def processed_data(raw_data, encoder, lb):
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
    X, y = process_data(
        raw_data,
        cat_features,
        label = "salary",
        training = False,
        encoder = encoder,
        lb = lb,
        )
    
    return X, y


# Tests for functions
def test_compute_model_metrics():
    y = np.array([0, 1, 1, 0, 1])
    preds = np.array([1, 1, 1, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    # Testing data types of metrics
    assert type(precision) == np.float64
    assert type(recall) == np.float64
    assert type(fbeta) == np.float64

def test_process_data(raw_data, processed_data):
    
    # Testing size of data outputs
    assert raw_data.shape[0] == processed_data[0].shape[0]
    assert raw_data.shape[1] - 1 <= processed_data[0].shape[1]
    assert raw_data["salary"].shape[0] == processed_data[1].shape[0]

def test_inference(processed_data, model):
    predictions = inference(processed_data[0], model)
    # Testing type and size of inference outputs
    assert predictions.shape[0] == processed_data[0].shape[0]
    assert type(predictions) == np.ndarray

# Tests for API
def test_get_greeting():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to The Census Salary Classifier API"}

def test_post_inference_0():
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    r = client.post("/inference", data = json.dumps(data))
    assert r.status_code == 200
    assert r.json() == {"prediction": 0}

def test_post_inference_1():
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 217400,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    r = client.post("/inference", data = json.dumps(data))
    assert r.status_code == 200
    assert r.json() == {"prediction": 1}