import pickle

with open("crop_model.pkl", "rb") as f:
    model = pickle.load(f)