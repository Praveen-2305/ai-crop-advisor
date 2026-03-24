from fastapi import APIRouter
from app.models.schemas import CropRequest
from app.services.feature_service import build_features
from app.services.ml_service import predict_crop_service

router = APIRouter()

@router.post("/predict")
def predict(data: CropRequest):

    processed_data = build_features(data)

    result = predict_crop_service(processed_data)

    return result