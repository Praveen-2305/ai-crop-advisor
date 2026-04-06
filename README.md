# 🌾 AI Crop Advisor

An intelligent crop recommendation system powered by a **Random Forest ML model**, **real-time weather data**, and a **FastAPI** backend. Given soil nutrient values and a city name, the system fetches live weather conditions and recommends the top 3 best-suited crops along with explainable AI insights.

---

## 🚀 Features

- 🌦️ **Real-time Weather Integration** — Fetches live temperature, humidity, and rainfall using the Tomorrow.io API (with OpenStreetMap geocoding via Nominatim)
- 🤖 **ML-Powered Predictions** — Random Forest classifier (100 estimators, max depth 10) trained on agricultural data to recommend optimal crops
- 🔍 **Explainable AI (XAI)** — Explains why a crop is recommended based on positive/negative soil and climate factors; confidence is adjusted by ±0.05 per factor
- 🏆 **Top-3 Crop Recommendations** — Returns the top 3 crops with adjusted confidence scores and decision labels
- 🛡️ **Graceful Fallbacks** — Falls back to default weather values (temp: 25°C, humidity: 50%, rainfall: 0 mm) if the live API is unavailable
- ⚙️ **FastAPI Backend** — Fast, async-ready REST API with automatic Swagger docs at `/docs`

---

## 🗂️ Project Structure

```
ai-crop-advisor/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app entry point
│   │   ├── crop_model.pkl           # Trained Random Forest model (auto-generated)
│   │   ├── train_model.py           # Script to train and save the ML model
│   │   ├── models/
│   │   │   ├── schemas.py           # Pydantic request/response schemas
│   │   │   └── model_loader.py      # Loads the trained .pkl model via joblib
│   │   ├── routes/
│   │   │   └── crop_routes.py       # POST /predict API endpoint
│   │   ├── services/
│   │   │   ├── feature_service.py   # Combines soil + weather into feature vector (with clamping)
│   │   │   ├── weather_service.py   # Geocoding + real-time weather fetching
│   │   │   └── ml_service.py        # ML prediction, XAI explanation, confidence adjustment
│   │   └── utils/
│   │       └── helpers.py           # Utility/helper functions
│   └── data/
│       └── crop_data.csv            # Training dataset (NPK, weather, crop labels)
└── frontend/                        # (Planned — not yet implemented)
```

---

## 🧠 How It Works

```
User Request (city + soil values)
        │
        ▼
  feature_service.py
  ┌─────────────────────────────────────────┐
  │  1. Geocode city → lat/lon              │  ← Nominatim (OpenStreetMap)
  │  2. Fetch weather → temp/humidity/rain  │  ← Tomorrow.io API (rainIntensity)
  │  3. Clamp values (temp: 15–40°C,        │
  │     humidity: 30–100%, rain: 0–300 mm)  │
  │  4. Scale rainfall ×10, cap at 300 mm   │
  │  5. Combine soil + weather features     │
  └────────────────┬────────────────────────┘
                   │
                   ▼
           ml_service.py
  ┌─────────────────────────────────────────┐
  │  1. Predict crop probabilities          │  ← Random Forest (crop_model.pkl)
  │  2. Normalize + scale relative to best │
  │  3. Pick Top 3 crops                   │
  │  4. Generate XAI explanation per crop  │  ← Factor analysis (N, P, K, pH, weather)
  │  5. Adjust confidence (+0.05 positive, │
  │     −0.05 negative), clamp to [0, 1]   │
  │  6. Assign decision score label        │
  └────────────────┬────────────────────────┘
                   │
                   ▼
         JSON Response to User
```

---

## 📡 API Reference

### `GET /`

Health check endpoint.

**Response**
```json
{"message": "AI Crop Advisor Running 🚀"}
```

---

### `POST /predict`

Recommend the best crops based on soil and location data.

**Request Body**

| Field  | Type     | Description                          | Constraints |
|--------|----------|--------------------------------------|-------------|
| `city` | `string` | City name for live weather lookup    | —           |
| `N`    | `float`  | Nitrogen content in soil (mg/kg)     | 0 – 140     |
| `P`    | `float`  | Phosphorus content in soil (mg/kg)   | 0 – 140     |
| `K`    | `float`  | Potassium content in soil (mg/kg)    | 0 – 200     |
| `ph`   | `float`  | pH level of the soil                 | 0 – 14      |

**Example Request**

```json
POST /predict
{
  "city": "Hyderabad",
  "N": 90,
  "P": 42,
  "K": 43,
  "ph": 6.5
}
```

**Example Response**

```json
{
  "recommended_crops": [
    {
      "crop": "rice",
      "confidence": 1.0,
      "score": "Highly Recommended",
      "factors": {
        "positive": ["high humidity suitable for rice", "sufficient rainfall for rice"],
        "negative": []
      }
    },
    {
      "crop": "maize",
      "confidence": 0.62,
      "score": "Recommended",
      "factors": {
        "positive": ["warm temperature for maize"],
        "negative": ["excess rainfall for maize"]
      }
    },
    {
      "crop": "cotton",
      "confidence": 0.38,
      "score": "Try with caution",
      "factors": {
        "positive": [],
        "negative": ["low potassium for cotton", "pH not ideal for cotton"]
      }
    }
  ]
}
```

**Decision Score Labels**

> Confidence is a **relative score** — the best-matching crop is always scaled to `1.0`, then adjusted ±0.05 per XAI factor.

| Confidence | Label              |
|------------|--------------------|
| > 0.60     | Highly Recommended |
| 0.40 – 0.60 | Recommended       |
| < 0.40     | Try with caution   |

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.9+
- pip or conda
- A [Tomorrow.io](https://www.tomorrow.io/) API key (free tier available)

---

### 1. Clone the Repository

```bash
git clone https://github.com/Praveen-2305/ai-crop-advisor.git
cd ai-crop-advisor
```

### 2. Create a Virtual Environment

```bash
# Using venv
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # macOS / Linux

# OR using conda
conda create -n crop-advisor python=3.10
conda activate crop-advisor
```

### 3. Install Dependencies

```bash
pip install fastapi uvicorn scikit-learn numpy pandas joblib requests
```

### 4. Configure the Weather API Key

Open `backend/app/services/weather_service.py` and replace the placeholder with your actual key:

```python
API_KEY = "your_tomorrow_io_api_key_here"
```

> 💡 **Tip:** Move this to a `.env` file and load it with `python-dotenv` for better security in production.

### 5. Train the ML Model

> Only needed once. The trained model is saved as `backend/app/crop_model.pkl`.

```bash
cd backend
python app/train_model.py
```

Expected output:
```
✅ Random Forest model trained and saved successfully!
```

### 6. Run the Backend Server

```bash
cd backend
uvicorn app.main:app --reload
```

The server will start at: **`http://localhost:8000`**

---

## 🧪 Testing the API

### Using Swagger UI (Built-in)

Visit: [http://localhost:8000/docs](http://localhost:8000/docs)

### Using curl

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"city": "Hyderabad", "N": 90, "P": 42, "K": 43, "ph": 6.5}'
```

### Using Python `requests`

```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "city": "Hyderabad",
    "N": 90,
    "P": 42,
    "K": 43,
    "ph": 6.5
})

print(response.json())
```

---

## 🌦️ Weather Fallback Behavior

If the live weather API is unavailable (network issues, rate limits, invalid city name), the system automatically falls back to safe defaults:

| Parameter   | Fallback Value | Clamp Range  |
|-------------|----------------|--------------|
| Temperature | 25°C           | 15°C – 40°C  |
| Humidity    | 50%            | 30% – 100%   |
| Rainfall    | 0 mm           | 0 – 300 mm   |

> Rainfall is sourced from `rainIntensity` (Tomorrow.io) and scaled ×10, then capped at 300 mm to align with the training data distribution.

---

## 🔍 XAI — Explainability Logic

The system generates crop-specific explanations based on the following rules:

| Crop      | Positive Factor                              | Negative Factor                         |
|-----------|----------------------------------------------|-----------------------------------------|
| **Rice**  | humidity > 65%, rainfall > 150 mm            | humidity ≤ 65%, rainfall ≤ 150 mm       |
| **Maize** | temperature > 28°C, rainfall < 100 mm        | temperature ≤ 28°C, rainfall ≥ 100 mm  |
| **Cotton**| potassium > 150 mg/kg, pH < 6               | potassium ≤ 150 mg/kg, pH ≥ 6          |
| **Wheat** | pH > 7                                       | pH ≤ 7                                  |
| **Others**| nitrogen > 50 mg/kg                          | nitrogen ≤ 50 mg/kg                     |

Each positive factor adds **+0.05** and each negative factor subtracts **−0.05** from the crop's base confidence score.

---

## 📊 Dataset

The model is trained on `backend/data/crop_data.csv`, which contains:

| Column        | Description                             |
|---------------|-----------------------------------------|
| `N`           | Nitrogen content (mg/kg)                |
| `P`           | Phosphorus content (mg/kg)              |
| `K`           | Potassium content (mg/kg)               |
| `temperature` | Average temperature (°C)                |
| `humidity`    | Relative humidity (%)                   |
| `ph`          | Soil pH level                           |
| `rainfall`    | Annual rainfall (mm)                    |
| `label`       | Target crop (e.g., rice, maize, cotton) |

---

## 🛠️ Tech Stack

| Layer          | Technology                                      |
|----------------|-------------------------------------------------|
| **API**        | FastAPI, Uvicorn                                |
| **ML Model**   | Scikit-learn (RandomForestClassifier), Joblib   |
| **Data**       | Pandas, NumPy                                   |
| **Weather**    | Tomorrow.io API (`rainIntensity` field), Nominatim (OSM) |
| **Validation** | Pydantic (with `Field` constraints)             |

---

## 🔮 Roadmap

- [ ] Frontend UI (React / HTML dashboard)
- [ ] Move API keys to `.env` with `python-dotenv`
- [ ] Add `requirements.txt` for easier dependency management
- [ ] Add fertilizer recommendation module
- [ ] Docker containerization
- [ ] Deploy to cloud (Railway / Render / AWS)
- [ ] Expand crop coverage and training dataset

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

Built with ❤️ by **Praveen** — [NavigateLabs](https://github.com/Praveen-2305)
