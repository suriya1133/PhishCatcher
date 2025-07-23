import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.impute import SimpleImputer
import joblib

# Load the dataset
df = pd.read_csv("Phishing_Legitimate_full.csv")

# Label column
target_col = "CLASS_LABEL"
X = df.drop(columns=[target_col])
y = df[target_col]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
joblib.dump(imputer, "imputer.pkl")

# Save feature columns for prediction-time alignment
joblib.dump(df.drop(columns=[target_col]).columns.tolist(), "feature_columns.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")

# Handle class imbalance
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, "rf_model.pkl")

# Train SVM
svm_model = SVC(probability=True, kernel='rbf', class_weight=class_weights_dict)
svm_model.fit(X_train, y_train)
joblib.dump(svm_model, "svm_model.pkl")

# Train XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, "xgb_model.pkl")

print("✅ Models trained and saved successfully!")
