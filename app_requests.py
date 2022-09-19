import requests
import json

url = "https://census-pr3.herokuapp.com/inference"

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

response = requests.post(url, data = json.dumps(data))
print("Querying live API")
print("Status code: " + str(response.status_code))
print("Response: " + str(response.json()))