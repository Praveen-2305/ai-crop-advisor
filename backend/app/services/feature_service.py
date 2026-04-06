import os
import joblib
import numpy as np

# -----------------------------
# LOAD ARTIFACTS (ONCE)
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

SCALER_PATH = os.path.join(BASE_DIR, "models/scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models/encoders.pkl")

scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODER_PATH)

le_soil = encoders["soil"]
le_season = encoders["season"]

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def get_season(month: int):
    if month in [6, 7, 8, 9]:
        return "kharif"
    elif month in [10, 11, 12, 1, 2]:
        return "rabi"
    else:
        return "zaid"

# -----------------------------
# MAIN FEATURE BUILDER
# -----------------------------

def build_features(data: dict):
    """
    data should contain:
    {
        "N": int,
        "P": int,
        "K": int,
        "temperature": float,
        "humidity": float,
        "rainfall": float,
        "ph": float,
        "soil_type": str,
        "month": int
    }
    """

    # Extract
    N = data["N"]
    P = data["P"]
    K = data["K"]
    temperature = data["temperature"]
    humidity = data["humidity"]
    rainfall = data["rainfall"]
    ph = data["ph"]
    soil_type = data["soil_type"]
    month = data["month"]

    # Season
    season = get_season(month)

    # Encode categorical
    soil_encoded = le_soil.transform([soil_type])[0]
    season_encoded = le_season.transform([season])[0]

    import pandas as pd

    # Feature vector (IMPORTANT: same order as training and exact feature names)
    features = pd.DataFrame([{
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "ph": ph,
        "soil_type": soil_encoded,
        "month": month,
        "season": season_encoded
    }])

    # Scale
    features_scaled = scaler.transform(features)

    return features_scaled