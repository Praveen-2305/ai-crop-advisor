from app.services.weather_service import get_weather

def build_features(data):
    weather = get_weather(data.city)

    return {
        "nitrogen": data.N,
        "phosphorus": data.P,
        "potassium": data.K,
        "ph": data.ph,
        "temperature": weather["temperature"],
        "humidity": weather["humidity"],
        "rainfall": weather["rainfall"]
    }