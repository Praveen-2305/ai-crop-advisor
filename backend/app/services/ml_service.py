import numpy as np
from app.models.model_loader import model


# 🔹 Explainable AI function (NEW)
def generate_explanation(data):
    positive = []
    negative = []

    # Nitrogen
    if data["nitrogen"] > 80:
        positive.append("high nitrogen")
    else:
        negative.append("low nitrogen")

    # Phosphorus
    if data["phosphorus"] >= 50:
        positive.append("sufficient phosphorus")
    else:
        negative.append("low phosphorus")

    # Potassium
    if data["potassium"] > 40:
        positive.append("adequate potassium")
    else:
        negative.append("low potassium")

    # Temperature
    if data["temperature"] > 25:
        positive.append("warm temperature")
    else:
        negative.append("low temperature")

    # Humidity
    if data["humidity"] > 60:
        positive.append("high humidity")
    else:
        negative.append("low humidity")

    # Rainfall
    if data["rainfall"] > 100:
        positive.append("good rainfall")
    else:
        negative.append("low rainfall")

    return {
        "positive": positive,
        "negative": negative
    }


# 🔹 Decision Score function (NEW)
def get_decision_score(confidence):
    if confidence > 0.75:
        return "Highly Recommended"
    elif confidence > 0.5:
        return "Recommended"
    else:
        return "Try with caution"


# 🔹 Main Prediction Service (UPGRADED)
def predict_crop_service(data):
    features = np.array([[ 
        data["nitrogen"],
        data["phosphorus"],
        data["potassium"],
        data["temperature"],
        data["humidity"],
        data["ph"],
        data["rainfall"]
    ]])

    # 🔹 Get probabilities
    probabilities = model.predict_proba(features)[0]
    classes = model.classes_

    # 🔹 Top 3 crops
    top_indices = np.argsort(probabilities)[-3:][::-1]

    # 🔹 Explanation
    explanation = generate_explanation(data)

    top_crops = []
    for idx in top_indices:
        crop_name = classes[idx]
        confidence = float(probabilities[idx])

        top_crops.append({
            "crop": crop_name,
            "confidence": round(confidence, 2),
            "score": get_decision_score(confidence),
            "factors": explanation
        })

    return {
        "recommended_crops": top_crops
    }