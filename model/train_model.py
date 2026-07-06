import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DATASET_PATH = os.path.join(PROJECT_DIR, "dataset", "Crop_recommendation.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "crop_model.pkl")
SCALER_PATH = os.path.join(SCRIPT_DIR, "crop_scaler.pkl")

if not os.path.exists(DATASET_PATH):
    print(f"[ERROR] Dataset not found at {os.path.abspath(DATASET_PATH)}")
    print()
    print("Please download the 'Crop recommendation' dataset from Kaggle:")
    print("  https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset")
    print()
    print("Place the file 'Crop_recommendation.csv' inside the 'dataset/' folder.")
    exit(1)

print("[1] Loading dataset...")
df = pd.read_csv(DATASET_PATH)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
print(f"    Shape: {df.shape}")
print(f"    Columns: {list(df.columns)}")
print(f"    Crops: {df['label'].nunique()}")
print(df['label'].value_counts())

print("\n[2] Separating features and target...")
feature_cols = ["Nitrogen", "phosphorus", "potassium", "temperature", "humidity", "ph", "rainfall"]
X = df[feature_cols].to_numpy(dtype=np.float64)
y = df["label"].to_numpy()

print("\n[3] Splitting into train/test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n[4] Scaling features with StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n[5] Training KNeighborsClassifier (k=5)...")
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

print("\n[6] Evaluating model...")
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"    Test Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"\n[7] Saving model to '{MODEL_PATH}'...")
joblib.dump(model, MODEL_PATH)

print(f"[8] Saving scaler to '{SCALER_PATH}'...")
joblib.dump(scaler, SCALER_PATH)

print("\nDone! Model and scaler are ready for the backend.")
