import numpy as np
from app.models.model_loader import model

# 🔹 Explanation function
def generate_explanation_dict(data):
    reasons = []

    if data["nitrogen"] > 80:
        reasons.append("high nitrogen")

    if data["phosphorus"] < 50:
        reasons.append("low phosphorus")

    if data["potassium"] > 40:
        reasons.append("adequate potassium")

    if data["temperature"] > 25:
        reasons.append("warm temperature")

    if data["humidity"] > 60:
        reasons.append("high humidity")

    if data["rainfall"] > 100:
        reasons.append("good rainfall")

    if not reasons:
        return "Balanced environmental conditions"

    return ", ".join(reasons)


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

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]
    confidence = float(np.max(probabilities))

    explanation = generate_explanation_dict(data)

    return {
        "recommended_crop": prediction,
        "confidence": round(confidence, 2),
        "explanation": f"{explanation} conditions favor {prediction} cultivation"
    }