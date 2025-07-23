import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import joblib
import warnings

warnings.filterwarnings('ignore')
# === Load the dataset ===
df = pd.read_csv("Phishing_Legitimate_full.csv")

# === Split features and label ===
X = df.drop(columns=["CLASS_LABEL"])
y = df["CLASS_LABEL"]

# === Save feature columns for later prediction ===
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, "feature_columns.pkl")

# === Impute missing values ===
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)
joblib.dump(imputer, "imputer.pkl")

# === Scale the features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
joblib.dump(scaler, "scaler.pkl")

# === Reshape for deep learning input ===
X_dl = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X_dl, y, test_size=0.2, random_state=42)

# === Build CNN Model ===
def build_cnn(input_shape):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Dropout(0.3),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# === Build LSTM Model ===
def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# === Train CNN ===
cnn_model = build_cnn((X_train.shape[1], 1))
cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
cnn_model.save("cnn_model.h5")

# === Train LSTM ===
lstm_model = build_lstm((X_train.shape[1], 1))
lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
lstm_model.save("lstm_model.h5")

print("✅ Training completed and models saved.")
