import joblib
import numpy as np
import pandas as pd
from feature_extractor import extract_features_from_url
import requests
from urllib.parse import urlparse  # ✅ NEW

# Load trained models and preprocessors
rf_model = joblib.load("rf_model.pkl")
svm_model = joblib.load("svm_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
feature_cols = joblib.load("feature_columns.pkl")

# ✅ NEW: Trusted domain list
trusted_domains = ['google.com', 'microsoft.com', 'apple.com', 'amazon.com', 'youtube.com', 'wikipedia.org']

def is_url_reachable(url):
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except:
        return False

def get_domain(url):
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "").lower()
    return domain

def predict_url(url):
    print(f"\n🔗 Checking URL: {url}\n")

    reachable = is_url_reachable(url)
    print("🌐 URL Reachability:", "Reachable ✅" if reachable else "Unreachable ❌")

    if not reachable:
        print("🧠 Final Decision: PHISHING ⚠️ (Unreachable URL)")
        print("✅ Prediction complete.")
        return

    # ✅ NEW: Check if domain is in trusted list
    domain = get_domain(url)
    if domain in trusted_domains:
        print(f"🛡️ Trusted Domain Detected: {domain}")
        print("🧠 Final Decision: LEGITIMATE ✅ (Whitelisted)")
        print("✅ Prediction complete.")
        return

    # Extract features and convert to DataFrame
    features = extract_features_from_url(url)
    features_df = pd.DataFrame([features])
    features_df = features_df.reindex(columns=feature_cols, fill_value=0)

    # Handle missing and scale
    X = imputer.transform(features_df)
    X = scaler.transform(X)

    # Get predictions and probabilities
    models = {
        "Random Forest": rf_model,
        "SVM": svm_model,
        "XGBoost": xgb_model
    }

    votes = []
    confidences = []

    for name, model in models.items():
        prob = model.predict_proba(X)[0]
        pred = model.predict(X)[0]

        label = "LEGITIMATE" if pred == 0 else "PHISHING"
        legit_prob, phish_prob = prob[0], prob[1]
        confidence = phish_prob if pred == 1 else legit_prob

        print(f"{name.ljust(18)} ➤ Prediction: {label}")
        print(f"                     Probabilities ➜ Legitimate: {legit_prob:.2f}, Phishing: {phish_prob:.2f}")

        votes.append(pred)
        confidences.append(confidence)

    # Confidence-weighted vote
    final_score = np.dot(votes, confidences) / sum(confidences)
    final_pred = 1 if final_score >= 0.5 else 0

    print("\n🧠 Final Decision:", "PHISHING ⚠️" if final_pred else "LEGITIMATE ✅")
    print("✅ Prediction complete.")

# Entry point
if __name__ == "__main__":
    input_url = input("Enter a URL to classify: ").strip()
    predict_url(input_url)
