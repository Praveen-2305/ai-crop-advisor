from pydantic import BaseModel, Field

class CropRequest(BaseModel):
    N: float = Field(..., ge=0, le=140)
    P: float = Field(..., ge=0, le=140)
    K: float = Field(..., ge=0, le=200)

    temperature: float = Field(..., ge=0, le=50)
    humidity: float = Field(..., ge=0, le=100)
    rainfall: float = Field(..., ge=0, le=500)

    ph: float = Field(..., ge=0, le=14)

    soil_type: str
    month: int = Field(..., ge=1, le=12)