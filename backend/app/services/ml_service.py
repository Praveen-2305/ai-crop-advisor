import os
import joblib
import numpy as np

from app.models.model_loader import model

# -----------------------------
# LOAD LABEL ENCODER (ONCE)
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
ENCODER_PATH = os.path.join(BASE_DIR, "models/encoders.pkl")

encoders = joblib.load(ENCODER_PATH)
label_encoder = encoders["label"]


# -----------------------------
# EXPLAINABLE AI LOGIC
# -----------------------------
# These rules exactly match CROP_RULES in generate_dataset.py
CROP_RULES = {
    "rice": {
        "temp": (20, 35), "humidity": (70, 95), "rainfall": (150, 300), "ph": (5.0, 7.0),
        "N": (80, 120), "P": (35, 60), "K": (35, 60),
        "soil": ["clay", "loamy"], "season": ["kharif"]
    },
    "maize": {
        "temp": (18, 32), "humidity": (50, 80), "rainfall": (50, 150), "ph": (5.5, 7.5),
        "N": (60, 100), "P": (30, 55), "K": (30, 55),
        "soil": ["loamy", "clay", "sandy"], "season": ["kharif", "zaid"]
    },
    "cotton": {
        "temp": (20, 35), "humidity": (40, 70), "rainfall": (50, 100), "ph": (6.0, 8.0),
        "N": (90, 130), "P": (40, 60), "K": (40, 60),
        "soil": ["clay", "loamy"], "season": ["kharif"]
    },
    "wheat": {
        "temp": (10, 25), "humidity": (40, 60), "rainfall": (30, 100), "ph": (6.0, 7.5),
        "N": (70, 110), "P": (35, 50), "K": (35, 50),
        "soil": ["loamy", "clay"], "season": ["rabi"]
    }
}

def get_season_from_month(month):
    if month in [6, 7, 8, 9]: return "kharif"
    if month in [10, 11, 12, 1, 2]: return "rabi"
    return "zaid"

def generate_explanation(data, crop):
    positive = []
    negative = []
    
    rule = CROP_RULES.get(crop)
    if not rule:
        return {"positive": [], "negative": []}

    temp = data.get("temperature", 0)
    hum = data.get("humidity", 0)
    rain = data.get("rainfall", 0)
    ph = data.get("ph", 0)
    N = data.get("N", 0)
    P = data.get("P", 0)
    K = data.get("K", 0)
    soil = data.get("soil_type", "")
    month = data.get("month", 0)
    season = get_season_from_month(month)

    # Weather
    if rule["temp"][0] <= temp <= rule["temp"][1]:
        positive.append(f"Temperature ({temp}°C) is ideal")
    else:
        negative.append(f"Temperature ({temp}°C) is outside ideal range")

    if rule["humidity"][0] <= hum <= rule["humidity"][1]:
        positive.append(f"Humidity ({hum}%) is ideal")
    else:
        negative.append(f"Humidity ({hum}%) is outside ideal range")

    if rule["rainfall"][0] <= rain <= rule["rainfall"][1]:
        positive.append(f"Rainfall ({rain}mm) is ideal")
    else:
        negative.append(f"Rainfall ({rain}mm) is not ideal")

    # Soil Chemistry
    if rule["ph"][0] <= ph <= rule["ph"][1]:
        positive.append(f"Soil pH ({ph}) is ideal")
    else:
        negative.append(f"Soil pH ({ph}) is not ideal")

    if rule["N"][0] <= N <= rule["N"][1]:
        positive.append(f"Nitrogen level ({N}) is excellent")
    if rule["P"][0] <= P <= rule["P"][1]:
        positive.append(f"Phosphorus level ({P}) is excellent")
    if rule["K"][0] <= K <= rule["K"][1]:
        positive.append(f"Potassium level ({K}) is excellent")

    # Ecology
    if soil in rule["soil"]:
        positive.append(f"{soil.title()} soil is perfect for {crop}")
    else:
        negative.append(f"{soil.title()} soil is not usually recommended")

    if season in rule["season"]:
        positive.append(f"Month {month} ({season.title()}) is the correct growing season")
    else:
        negative.append(f"Month {month} ({season.title()}) is out of season for {crop}")

    return {
        "positive": positive,
        "negative": negative
    }


# -----------------------------
# DECISION SCORE
# -----------------------------

def get_decision_score(confidence):
    if confidence >= 0.70:
        return "Highly Recommended"
    elif confidence >= 0.40:
        return "Recommended"
    else:
        return "Try with caution"


# -----------------------------
# MAIN PREDICTION SERVICE
# -----------------------------

def predict_crop_service(features, raw_data):
    """
    features: scaled numpy/dataframe from feature_service.build_features()
    raw_data: original input dictionary containing unscaled feature values
    Returns top 3 crop recommendations with confidence and score.
    """

    # Get raw class probabilities from Random Forest
    probabilities = model.predict_proba(features)[0]

    # Normalize raw probabilities just in case
    total = sum(probabilities)
    if total > 0:
        raw_probs = np.array([p / total for p in probabilities])
    else:
        raw_probs = np.array(probabilities)

    # Smooth the probabilities using Laplace smoothing style logic
    # Why? The synthetic dataset generates perfectly separable classes, making
    # Random Forest overconfident (exactly 1.0 and 0.0 scores).
    # This smoothing blends 80% model confidence with 20% baseline uniform 
    # probability to provide realistic numbers mimicking real-world uncertainty.
    num_classes = len(raw_probs)
    smoothed_probs = (0.80 * raw_probs) + (0.20 * (1.0 / num_classes))

    # Decode class indices → crop name strings using label encoder
    classes = label_encoder.inverse_transform(model.classes_)

    # Pick top 3
    top_indices = np.argsort(smoothed_probs)[-3:][::-1]

    top_crops = []

    for idx in top_indices:
        crop_name = classes[idx]
        base_confidence = float(smoothed_probs[idx])

        # Generate rule-based explanations for this crop using the exact rules from generation
        explanation = generate_explanation(raw_data, crop_name)

        # Count positive and negative factors
        positive_count = len(explanation["positive"])
        negative_count = len(explanation["negative"])

        # Fine-tune confidence score using explanation rules (+0.04 per positive, -0.04 per negative)
        confidence = base_confidence + (positive_count * 0.04) - (negative_count * 0.04)

        # Clamp safely between 0 and 1
        confidence = max(0.01, min(confidence, 0.99))
        confidence = round(confidence, 2)

        top_crops.append({
            "crop": crop_name,
            "confidence": confidence,
            "score": get_decision_score(confidence),
            "factors": explanation
        })

    return {
        "recommended_crops": top_crops
    }