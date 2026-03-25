from pydantic import BaseModel, Field

class CropRequest(BaseModel):
    city: str
    N: float = Field(..., ge=0, le=140)
    P: float = Field(..., ge=0, le=140)
    K: float = Field(..., ge=0, le=200)
    ph: float = Field(..., ge=0, le=14)