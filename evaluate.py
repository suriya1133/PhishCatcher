# evaluate_models.py

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
import numpy as np

def load_test_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["CLASS_LABEL"])
    y = df["CLASS_LABEL"]
    return X, y

def preprocess(X):
    imputer = joblib.load("imputer.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    X = X.reindex(columns=feature_columns, fill_value=0)
    X = imputer.transform(X)
    X_scaled = scaler.transform(X)
    return X_scaled

def evaluate_ml_models(X, y):
    models = {
        "Random Forest": joblib.load("rf_model.pkl"),
        "SVM": joblib.load("svm_model.pkl"),
        "XGBoost": joblib.load("xgb_model.pkl"),
    }

    results = {}
    for name, model in models.items():
        y_pred = model.predict(X)
        results[name] = get_metrics(y, y_pred)

    return results

def evaluate_dl_models(X, y):
    X_dl = X.reshape(X.shape[0], X.shape[1], 1)
    models = {
        "CNN": load_model("cnn_model.h5"),
        "LSTM": load_model("lstm_model.h5"),
    }

    results = {}
    for name, model in models.items():
        y_proba = model.predict(X_dl, verbose=0).flatten()
        y_pred = (y_proba > 0.5).astype(int)
        results[name] = get_metrics(y, y_pred)

    return results

def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
    }

def evaluate_all(csv_path="Phishing_Legitimate_full.csv"):
    X, y = load_test_data(csv_path)
    X_scaled = preprocess(X)
    ml_results = evaluate_ml_models(X_scaled, y)
    dl_results = evaluate_dl_models(X_scaled, y)
    return ml_results, dl_results
