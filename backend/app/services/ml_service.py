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

    # 🔹 Get probabilities for all crops
    probabilities = model.predict_proba(features)[0]
    classes = model.classes_

    # 🔹 Get top 3 crops
    top_indices = np.argsort(probabilities)[-3:][::-1]

    explanation = generate_explanation_dict(data)

    top_crops = []
    for idx in top_indices:
        crop_name = classes[idx]
        confidence = float(probabilities[idx])

        top_crops.append({
            "crop": crop_name,
            "confidence": round(confidence, 2),
            "explanation": f"{explanation} conditions favor {crop_name} cultivation"
        })

    return {
        "recommended_crops": top_crops
    }