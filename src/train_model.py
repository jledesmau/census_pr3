# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# Loading the data
data = pd.read_csv("../data/census.csv")
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

# Train and save a model
model = train_model(X_train, y_train)


