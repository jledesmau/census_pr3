# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import train_model, test_model_slices, compute_model_metrics, inference
import joblib
import os

def main():
    path_base = os.path.dirname(os.path.abspath(__file__))
    path_base = path_base.replace("src", "")

    # Loading data
    data = load_data(path_base + "data/census.csv")

    # Train-test split
    train, test = train_test_split(data, test_size = 0.20,random_state = 0)
    
    # Identifying categorical features names
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

    # Training and saving the model, encoder and label binarizer
    model, encoder, lb = train_objects(train, cat_features)
    save_artifacts(path_base + "output/", model, encoder, lb)

    # Processing the test data
    test.reset_index(drop = True, inplace = True)
    X_test, y_test = process_data(
        X = test,
        categorical_features = cat_features,
        label = "salary",
        training = False,
        encoder = encoder,
        lb = lb
    )

    # Testing the model
    overall_performance = test_overall_model(X_test, y_test, model)

    slice_performances = {}
    for feature in cat_features:
        slice_performances[feature] = test_model_slices(
            test, X_test, y_test, feature, model)

    slice_metrics = pd.DataFrame.from_dict({(i,j): slice_performances[i][j]
                                                for i in slice_performances.keys()
                                                for j in slice_performances[i].keys()},
                                                orient = 'index').rename_axis(('Features', 'Slices'))

    slice_metrics.to_csv(path_base + "output/" + "slice_metrics.csv")

    # Printing and saving model performance
    performance_text = "OVERALL PERFORMANCE\n"
    performance_text += f"Precision: {overall_performance['Precision']}\n"
    performance_text += f"Recall: {overall_performance['Recall']}\n"
    performance_text += f"F1: {overall_performance['F1']}\n\n"

    performance_text += "PERFORMANCE PER SLICES\n"
    performance_text += slice_metrics.to_string()
    
    print(performance_text)

    with open(path_base + "output/" + "slice_output.txt", "w") as text_file:
        print(performance_text, file = text_file)

def load_data(data_path):
    """
    Loads the data from the given path (csv file) as a dataframe
    """
    data = pd.read_csv(data_path)
    data.replace("?", np.nan, inplace = True)
    data.dropna(inplace = True)
    return data

def train_objects(train, cat_features):
    """
    Trains model, categorical encodfer and label binarizer
    """
    # Processing the training data
    X_train, y_train, encoder, lb = process_data(
        X = train,
        categorical_features = cat_features,
        label = "salary",
    )

    # Training the model
    model = train_model(X_train, y_train)

    return model, encoder, lb

def save_artifacts(path, model, encoder, lb):
    """
    Saves model and data processing objects
    """
    outcomes = {"model": model,
                "encoder": encoder,
                "label_binarizer": lb}

    for name, outcome in outcomes.items():
        joblib.dump(outcome, path + name + ".pkl")

def test_overall_model(X_test, y_test, model):
    """
    Tests overall model performance
    """
    # Making predictions
    preds = inference(X_test, model)

    # Computing overall performance metrics
    precision, recall, f1 = compute_model_metrics(y_test, preds)

    performance = {
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

    return performance


if __name__ == "__main__":
    main()