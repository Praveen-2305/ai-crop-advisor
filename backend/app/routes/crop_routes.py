from fastapi import APIRouter, HTTPException
from app.models.schemas import CropRequest
from app.services.feature_service import build_features
from app.services.ml_service import predict_crop_service

router = APIRouter()

@router.post("/predict")
def predict(data: CropRequest):
    try:
        # 🔹 Build features (weather + soil)
        processed_data = build_features(data)

        # 🔹 Get prediction
        result = predict_crop_service(processed_data)

        return result

    except ValueError as ve:
        # 🔹 Known errors (like weather failure)
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )

    except Exception as e:
        # 🔹 Unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )