import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Dense, Flatten, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
# -------------------------
# 🔹 1. Load Dataset
# -------------------------
# Replace with your own dataset path
df = pd.read_csv("DLmodels\Phishing_Legitimate_full.csv")  # 🔁 <- change this path as needed

# -------------------------
# 🔹 2. Feature & Label
# -------------------------
X = df.drop(columns=["CLASS_LABEL"])
y = df["CLASS_LABEL"]

# -------------------------
# 🔹 3. Preprocessing
# -------------------------
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape to 3D [samples, timesteps, features] → required for CNN and LSTM
X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# -------------------------
# 🔹 4. Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# -------------------------
# 🔹 5. CNN Model
# -------------------------
def create_cnn(input_shape):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -------------------------
# 🔹 6. BiLSTM Model
# -------------------------
def create_bilstm(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(64), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -------------------------
# 🔹 7. Train Models
# -------------------------
epochs = 15
batch_size = 32

print("\n🚀 Training CNN model...")
cnn_model = create_cnn(X_train.shape[1:])
cnn_history = cnn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

print("\n🚀 Training BiLSTM model...")
lstm_model = create_bilstm(X_train.shape[1:])
lstm_history = lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

# -------------------------
# 🔹 8. Evaluate Models
# -------------------------
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred_classes)
    prec = precision_score(y_test, y_pred_classes)
    rec = recall_score(y_test, y_pred_classes)
    f1 = f1_score(y_test, y_pred_classes)
    print(f"\n📊 {name} Evaluation:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    return acc, prec, rec, f1

evaluate_model(cnn_model, X_test, y_test, "CNN")
evaluate_model(lstm_model, X_test, y_test, "BiLSTM")

# -------------------------
# 🔹 9. Save Models
# -------------------------
cnn_model.save("cnn_model.h5")
lstm_model.save("lstm_model.h5")
print("\n✅ Models saved as cnn_model.h5 and lstm_model.h5")

# -------------------------
# 🔹 10. (Optional) Plot Loss
# -------------------------
def plot_training_history(history, title):
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{title} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_training_history(cnn_history, "CNN")
plot_training_history(lstm_history, "BiLSTM")
