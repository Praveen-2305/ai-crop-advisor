from pydantic import BaseModel

class CropRequest(BaseModel):
    city: str
    N: float
    P: float
    K: float
    ph: float