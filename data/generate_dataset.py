import pandas as pd
import numpy as np
import random
import os
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
NUM_SAMPLES = 2000

CROPS = ["rice", "maize", "cotton", "wheat"]

SOIL_TYPES = ["clay", "loamy", "sandy"]

SEASONS = {
    "kharif": [6, 7, 8, 9],
    "rabi": [10, 11, 12, 1, 2],
    "zaid": [3, 4, 5]
}

# 🚀 STEP 1: Upgraded Dictionary (Adding NPK, Soil, and Season)
CROP_RULES = {
    "rice": {
        "temp": (20, 35), "humidity": (70, 95), "rainfall": (150, 300), "ph": (5.0, 7.0),
        "N": (80, 120), "P": (35, 60), "K": (35, 60),
        "soil": ["clay", "loamy"], "season": ["kharif"]
    },
    "maize": {
        "temp": (18, 32), "humidity": (50, 80), "rainfall": (50, 150), "ph": (5.5, 7.5),
        "N": (60, 100), "P": (30, 55), "K": (30, 55),
        "soil": ["loamy", "clay", "sandy"], "season": ["kharif", "zaid"]
    },
    "cotton": {
        "temp": (20, 35), "humidity": (40, 70), "rainfall": (50, 100), "ph": (6.0, 8.0),
        "N": (90, 130), "P": (40, 60), "K": (40, 60),
        "soil": ["clay", "loamy"], "season": ["kharif"]
    },
    "wheat": {
        "temp": (10, 25), "humidity": (40, 60), "rainfall": (30, 100), "ph": (6.0, 7.5),
        "N": (70, 110), "P": (35, 50), "K": (35, 50),
        "soil": ["loamy", "clay"], "season": ["rabi"]
    }
}

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------

def random_soil():
    return random.choice(SOIL_TYPES)

def generate_npk():
    return {
        "N": np.random.randint(10, 140),
        "P": np.random.randint(5, 145),
        "K": np.random.randint(5, 205),
    }

def generate_weather():
    return {
        "temperature": round(np.random.uniform(10, 40), 2),
        "humidity": round(np.random.uniform(30, 95), 2),
        "rainfall": round(np.random.uniform(0, 300), 2),
    }

def generate_ph():
    return round(np.random.uniform(4.5, 8.5), 2)

def generate_month():
    return random.randint(1, 12)

def get_season(month):
    for season, months in SEASONS.items():
        if month in months:
            return season
    return "unknown"

# -----------------------------
# SCORING FUNCTION (CORE LOGIC)
# -----------------------------

# 🚀 STEP 2: Empowering the Logic - now checking all 9 properties
def score_crop(crop, features):
    rule = CROP_RULES[crop]
    score = 0
    
    # 🌡️ Weather (1 pt each)
    if rule["temp"][0] <= features["temperature"] <= rule["temp"][1]:
        score += 1
    if rule["humidity"][0] <= features["humidity"] <= rule["humidity"][1]:
        score += 1
    if rule["rainfall"][0] <= features["rainfall"] <= rule["rainfall"][1]:
        score += 1
    
    # 🧪 Soil Science (1 pt each)
    if rule["ph"][0] <= features["ph"] <= rule["ph"][1]:
        score += 1
    if rule["N"][0] <= features["N"] <= rule["N"][1]:
        score += 1
    if rule["P"][0] <= features["P"] <= rule["P"][1]:
        score += 1
    if rule["K"][0] <= features["K"] <= rule["K"][1]:
        score += 1

    # 🌍 Ecology (Soil Type: 1 pt)
    if features.get("soil_type") in rule["soil"]:
        score += 1

    # ⏳ Seasonality (Massive 2 point weight, because breaking seasonal trends usually causes massive drops in yield!)
    if features.get("season") in rule["season"]:
        score += 2

    return score


def assign_crop(features):
    scores = {crop: score_crop(crop, features) for crop in CROPS}
    
    # Add light randomness for realism (tie breaker)
    for crop in scores:
        scores[crop] += np.random.uniform(0, 1)
    
    # Choose best statistically scoring crop
    return max(scores, key=scores.get)

# -----------------------------
# MAIN DATA GENERATOR
# -----------------------------

def generate_sample():
    weather = generate_weather()
    npk = generate_npk()
    ph = generate_ph()
    month = generate_month()
    season = get_season(month) # Grab season mapping
    soil = random_soil()
    
    # Include EVERYTHING for the scorer now
    features = {
        **weather,
        **npk,
        "ph": ph,
        "month": month,
        "season": season,
        "soil_type": soil
    }
    
    crop = assign_crop(features)
    
    return {
        "N": npk["N"],
        "P": npk["P"],
        "K": npk["K"],
        "temperature": weather["temperature"],
        "humidity": weather["humidity"],
        "rainfall": weather["rainfall"],
        "ph": ph,
        "soil_type": soil,
        "month": month,
        "season": season,
        "label": crop
    }

# -----------------------------
# GENERATE DATASET
# -----------------------------

def generate_dataset(n=NUM_SAMPLES):
    data = []
    
    print("🌾 Generating smart agronomy dataset...")
    for _ in tqdm(range(n)):
        data.append(generate_sample())
    
    df = pd.DataFrame(data)
    
    return df

if __name__ == "__main__":
    df = generate_dataset()

    # Use absolute path so the script works from any working directory
    OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed", "final_dataset.csv")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Dataset generated: {OUTPUT_PATH}")
    print(df.head())