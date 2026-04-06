import joblib
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

MODEL_PATH = os.path.join(BASE_DIR, "models/crop_model.pkl")

model = joblib.load(MODEL_PATH)