from fastapi import APIRouter, HTTPException
from app.models.schemas import CropRequest
from app.services.feature_service import build_features
from app.services.ml_service import predict_crop_service

router = APIRouter()

@router.post("/predict")
def predict(data: CropRequest):
    try:
        # ✅ FIX: convert to dict
        input_data = data.dict()

        # ✅ Build features
        processed_data = build_features(input_data)

        # ✅ Predict
        result = predict_crop_service(processed_data, input_data)

        return result

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")