import numpy as np
from app.models.model_loader import model

def predict_crop_service(data):
    features = np.array([[
        data.nitrogen,
        data.phosphorus,
        data.potassium,
        data.temperature,
        data.humidity,
        data.ph,
        data.rainfall
    ]])

    prediction = model.predict(features)[0]

    return {
        "recommended_crop": prediction
    }