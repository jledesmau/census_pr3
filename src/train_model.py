# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import train_model
import joblib
import os

path_base = os.path.dirname(os.path.abspath(__file__))
path_base = path_base.replace("src", "")

# Loading the data
data = pd.read_csv(path_base + "data/census.csv")
data.replace("?", np.nan, inplace = True)
data.dropna(inplace = True)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size = 0.20)

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

# Processing the data
X_train, y_train, encoder, lb = process_data(
    X = train,
    categorical_features = cat_features,
    label = "salary",
)

# Training the model
model = train_model(X_train, y_train)

# Saving model and data processing objects
outcomes = {"model": model,
            "encoder": encoder,
            "label_binarizer": lb}

for name, outcome in outcomes.items():
    joblib.dump(outcome, path_base + "model/" + name + ".pkl")

