import os
import csv
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from check import extract_features_from_url

from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# Load models and preprocessors
cnn_model = load_model("cnn_model.h5")
lstm_model = load_model("lstm_model.h5")
rf_model = joblib.load("rf_model.pkl")  # Random Forest model
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ---- Heuristic Checks ----
def contains_suspicious_keywords(url):
    keywords = ['login', 'verify', 'secure', 'account', 'update', 'signin', 'bank', 'confirm']
    return int(any(k in url.lower() for k in keywords))

def is_shortened_url(url):
    shortening_services = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly']
    return int(any(service in url for service in shortening_services))

def unusual_tld(url):
    uncommon_tlds = ['.xyz', '.top', '.tk', '.gq', '.ml', '.cf']
    return int(any(url.endswith(tld) for tld in uncommon_tlds))

def heuristic_warning(url):
    reasons = []
    if contains_suspicious_keywords(url):
        reasons.append("Suspicious keyword")
    if is_shortened_url(url):
        reasons.append("Shortened URL")
    if unusual_tld(url):
        reasons.append("Unusual TLD")
    return reasons if reasons else None

# ---- URL Reachability ----
def is_url_reachable(url):
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except:
        return False

# ---- Logging ----
def log_prediction(url, cnn_pred, lstm_pred, final_label, heuristics):
    log_file = "prediction_log.csv"
    headers = ["Timestamp", "URL", "CNN_Pred", "LSTM_Pred", "Final_Label", "Heuristics"]
    row = [datetime.now().isoformat(), url, cnn_pred, lstm_pred, final_label, "; ".join(heuristics) if heuristics else "None"]

    file_exists = os.path.isfile(log_file)
    with open(log_file, "a", newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)

# ---- Main Prediction ----
def predict_url(url):
    print(f"\n🔍 Checking: {url}")

    reachable = is_url_reachable(url)
    print("🌐 URL is reachable." if reachable else "❌ URL not reachable. Flagged as potential phishing.")

    if not reachable:
        print("🚨 Final Verdict → PHISHING (unreachable URL)")
        log_prediction(url, "N/A", "N/A", "PHISHING", ["Unreachable"])
        return "PHISHING", 1.0  # ← Always return (label, confidence) even if unreachable

    try:
        # Feature Extraction
        features = extract_features_from_url(url)
        features_df = pd.DataFrame([features])
        features_df = features_df.reindex(columns=feature_columns, fill_value=0)

        X = imputer.transform(features_df)
        X_scaled = scaler.transform(X)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        # Deep Learning Predictions
        cnn_proba = cnn_model.predict(X_reshaped, verbose=0)[0][0]
        lstm_proba = lstm_model.predict(X_reshaped, verbose=0)[0][0]

        # Machine Learning Prediction
        rf_proba = rf_model.predict_proba(X_scaled)[0][1]

        print(f"🧠 CNN Probability: {cnn_proba:.2f}")
        print(f"🧠 LSTM Probability: {lstm_proba:.2f}")
        print(f"🧠 Random Forest Probability: {rf_proba:.2f}")

        # Smarter Ensemble
        final_score = (cnn_proba * 0.4) + (lstm_proba * 0.4) + (rf_proba * 0.2)
        final_label = "PHISHING" if final_score > 0.5 else "LEGITIMATE"

        heuristics = heuristic_warning(url)
        if heuristics:
            print(f"⚠️ Heuristic Warning: {', '.join(heuristics)}")

        print(f"🎯 Final Ensemble Score: {final_score:.2f}")
        print(f"🚨 Final Verdict → {final_label}")

        log_prediction(url, f"{cnn_proba:.2f}", f"{lstm_proba:.2f}", final_label, heuristics)

        return final_label, final_score

    except Exception as e:
        print(f"Error in prediction pipeline: {e}")
        log_prediction(url, "ERROR", "ERROR", "UNKNOWN", [str(e)])
        return "UNKNOWN", 0.0


# ---- Entry Point ----
if __name__ == "__main__":
    print("🔐 Deep Learning Phishing URL Detector\n")
    while True:
        user_input = input("🔗 Enter a URL to test (or type 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            print("👋 Exiting. Stay safe!")
            break
        label, score = predict_url(user_input)
        print(f"✅ Prediction: {label} (Confidence: {score*100:.2f}%)\n")
