# training/train_model.py

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data/processed/final_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models/crop_model.pkl")

# -----------------------------
# LOAD DATA
# -----------------------------
print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print(df.head())

# -----------------------------
# PREPROCESSING
# -----------------------------

# Encode categorical features
print("\n🔧 Encoding categorical features...")

le_soil = LabelEncoder()
le_season = LabelEncoder()
le_label = LabelEncoder()

df["soil_type"] = le_soil.fit_transform(df["soil_type"])
df["season"] = le_season.fit_transform(df["season"])
df["label"] = le_label.fit_transform(df["label"])

# Save encoders (VERY IMPORTANT for API)
ENCODER_PATH = os.path.join(BASE_DIR, "models/encoders.pkl")
joblib.dump({
    "soil": le_soil,
    "season": le_season,
    "label": le_label
}, ENCODER_PATH)

# -----------------------------
# FEATURES & TARGET
# -----------------------------
X = df.drop(columns=["label"])
y = df["label"]

# -----------------------------
# SCALING (IMPORTANT)
# -----------------------------
print("\n⚖️ Scaling features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
SCALER_PATH = os.path.join(BASE_DIR, "models/scaler.pkl")
joblib.dump(scaler, SCALER_PATH)

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# MODEL TRAINING
# -----------------------------
print("\n🤖 Training model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
print("\n📊 Evaluating model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# SAVE MODEL
# -----------------------------
print("\n💾 Saving model...")

joblib.dump(model, MODEL_PATH)

print(f"✅ Model saved at: {MODEL_PATH}")
print(f"✅ Scaler saved at: {SCALER_PATH}")
print(f"✅ Encoders saved at: {ENCODER_PATH}")