from app.services.weather_service import get_weather

def build_features(data):
    try:
        weather = get_weather(data.city)
    except Exception as e:
        print("Weather API failed:", e)

        weather = {
            "temperature": 25,
            "humidity": 50,
            "rainfall": 0
        }

    # Clamp weather values
    weather["temperature"] = max(15, min(weather["temperature"], 40))
    weather["humidity"] = max(30, min(weather["humidity"], 100))
    weather["rainfall"] = max(0, min(weather["rainfall"], 300))

    # Derived features (for future use)
    moisture = weather["humidity"] + weather["rainfall"] / 2
    nutrient_balance = (data.N + data.P + data.K) / 3

    return {
        "nitrogen": data.N,
        "phosphorus": data.P,
        "potassium": data.K,
        "ph": data.ph,
        "temperature": weather["temperature"],
        "humidity": weather["humidity"],
        "rainfall": weather["rainfall"],

        # Optional debug (you can remove later)
        "moisture": moisture,
        "nutrient_balance": nutrient_balance
    }