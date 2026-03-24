
import requests

API_KEY = "cJgHRFyIjtEUcOZnoVhNig6S4xIdWFvP"

def get_coordinates(city: str):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": city,
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "MyWeatherApp/1.0"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        results = response.json()
    except Exception as e:
        raise Exception(f"Geocoding failed: {e}")

    if not results:
        raise ValueError(f"City not found: {city}")

    return float(results[0]["lat"]), float(results[0]["lon"])


def get_weather(city: str):
    lat, lon = get_coordinates(city)

    url = f"https://api.tomorrow.io/v4/weather/realtime?location={lat},{lon}&apikey={API_KEY}"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        raise Exception(f"Weather API failed: {e}")

    values = data["data"]["values"]

    return {
        "temperature": values.get("temperature"),
        "humidity": values.get("humidity"),
        "rainfall": values.get("rainIntensity", 0)  # ✅ FIXED
    }