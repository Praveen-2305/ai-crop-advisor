import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

df = pd.read_csv("../data/crop_data.csv")

X = df.drop("label", axis=1)
y = df["label"]

model = DecisionTreeClassifier()
model.fit(X, y)

with open("crop_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained ✅")