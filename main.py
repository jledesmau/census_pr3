from pydantic import BaseModel, Field
from fastapi import FastAPI, Body, Query, Path
import pandas as pd

import os
import joblib
from src.ml.model import inference
from src.ml.data import process_data

app = FastAPI()

# Models
class Person(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias = "education-num")
    marital_status: str = Field(alias = "marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias = "capital-gain")
    capital_loss: int = Field(alias = "capital-loss")
    hours_per_week: int = Field(alias = "hours-per-week")
    native_country: str = Field(alias = "native-country")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
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
        }

# Greeting function
@app.get("/")
def greeting():
    return {"message": "Welcome to The Census Salary Classifier API"}

# Inference function
@app.post("/inference")
def make_inference(person: Person = Body()):
    # Loading artifacts from serialized .pkl file
    path_base = os.path.dirname(os.path.abspath(__file__)) + "/output/"
    model = joblib.load(path_base + "model.pkl")
    encoder = joblib.load(path_base + "encoder.pkl")
    lb = joblib.load(path_base + "label_binarizer.pkl")

    # Converting JSON to Pandas DataFrame
    person_data = pd.DataFrame([person.dict(by_alias = True)])

    # Processing the data
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
    processed_data = process_data(
        X = person_data,
        categorical_features = cat_features,
        training = False,
        encoder = encoder,
        lb = lb
    )

    # Getting the prediction from the model
    prediction = inference(processed_data[0], model)
    return {"prediction": float(prediction)}
