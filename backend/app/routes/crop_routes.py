from fastapi import APIRouter
from app.models.schemas import CropInput
from app.services.ml_service import predict_crop_service

router = APIRouter()

@router.post("/predict")
def predict(data: CropInput):
    return predict_crop_service(data)