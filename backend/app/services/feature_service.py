from app.services.weather_service import get_weather

def build_features(data):
    try:
        weather = get_weather(data.city)
    except Exception as e:
        print("Weather API failed:", e)

        # 🔹 fallback values
        weather = {
            "temperature": 25,
            "humidity": 50,
            "rainfall": 0
        }

    return {
        "nitrogen": data.N,
        "phosphorus": data.P,
        "potassium": data.K,
        "ph": data.ph,
        "temperature": weather["temperature"],
        "humidity": weather["humidity"],
        "rainfall": weather["rainfall"]
    }