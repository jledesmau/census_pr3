from pydantic import BaseModel, Field
from fastapi import FastAPI, Body, Query, Path

import os
from src.ml.model import inference

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
    native_country: int = Field(alias = "native-country")

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


@app.get("/")
def greeting():
    return {"Welcome to": "The Census Salary Classifier API"}


@app.post("/inference/new")
def make_inference(person: Person = Body(...)):
    path_base = os.path.dirname(os.path.abspath(__file__))
    model_path = path_base + "output/model.pkl"
    return person
    # return inference(person, model_path)