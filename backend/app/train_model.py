import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
import joblib

# ✅ Fix path issue
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "crop_data.csv")

# Load dataset
df = pd.read_csv(data_path)

# Features and target
X = df.drop("label", axis=1)
y = df["label"]

# Model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# Train
model.fit(X, y)

# Save model
model_path = os.path.join(BASE_DIR, "app", "crop_model.pkl")
joblib.dump(model, model_path)

print("✅ Random Forest model trained and saved successfully!")