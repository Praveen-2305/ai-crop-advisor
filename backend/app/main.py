from fastapi import FastAPI
from app.routes.crop_routes import router as crop_router

app = FastAPI()

app.include_router(crop_router)

@app.get("/")
def home():
    return {"message": "AI Crop Advisor Running 🚀"}