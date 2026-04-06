import numpy as np
from app.models.model_loader import model


# 🔹 Explainable AI function (NEW)
def generate_explanation(data, crop):
    positive = []
    negative = []

    # 🌾 Rice logic
    if crop == "rice":
        if data["humidity"] > 65:
            positive.append("high humidity suitable for rice")
        else:
            negative.append("low humidity for rice")

        if data["rainfall"] > 150:
            positive.append("sufficient rainfall for rice")
        else:
            negative.append("insufficient rainfall for rice")

    # 🌽 Maize logic
    elif crop == "maize":
        if data["temperature"] > 28:
            positive.append("warm temperature for maize")
        else:
            negative.append("temperature too low for maize")

        if data["rainfall"] < 100:
            positive.append("low rainfall suits maize")
        else:
            negative.append("excess rainfall for maize")

    # 🧵 Cotton logic
    elif crop == "cotton":
        if data["potassium"] > 150:
            positive.append("high potassium supports cotton")
        else:
            negative.append("low potassium for cotton")

        if data["ph"] < 6:
            positive.append("slightly acidic soil suits cotton")
        else:
            negative.append("pH not ideal for cotton")

    # 🌾 Wheat logic
    elif crop == "wheat":
        if data["ph"] > 7:
            positive.append("alkaline soil suits wheat")
        else:
            negative.append("pH too low for wheat")

    # 🌿 Default fallback
    else:
        if data["nitrogen"] > 50:
            positive.append("moderate nitrogen level")
        else:
            negative.append("low nitrogen")

    return {
        "positive": positive,
        "negative": negative
    }

# 🔹 Decision Score function (NEW)
def get_decision_score(confidence):
    if confidence > 0.6:
        return "Highly Recommended"
    elif confidence > 0.4:
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
    
    #Normalize probabilities
    total =sum(probabilities)
    probabilities = [p/total for p in probabilities]

    # 🔥 Relative scaling (make best crop close to 1.0)
    max_prob = max(probabilities)

    adjusted_probs = [
        (p / max_prob) if max_prob > 0 else 0
        for p in probabilities
    ]

    classes = model.classes_

    # 🔹 Top 3 crops
    top_indices = np.argsort(adjusted_probs)[-3:][::-1]

    top_crops = []

    for idx in top_indices:
        crop_name = classes[idx]
        base_confidence = float(adjusted_probs[idx])

        explanation = generate_explanation(data, crop_name)

        # Count factors
        positive = len(explanation["positive"])
        negative = len(explanation["negative"])

        # Adjust confidence using explanation
        confidence = base_confidence + (positive * 0.05) - (negative * 0.05)

        # Clamp between 0 and 1
        confidence = max(0, min(confidence, 1))
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