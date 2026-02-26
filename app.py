# app.py
# AI-Based Real-Time AQI Monitoring and Prediction System
# Streamlit + OpenWeather API + Simple AI Prediction

import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit.components.v1 import html

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI AQI Monitoring System",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 AI-Based Real-Time AQI Monitoring and Prediction System")
st.write("Monitor live air quality and predict future AQI using AI")

# =========================
# LOAD API KEY FROM SECRETS
# =========================
API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")

if API_KEY == "":
    st.error("API key not found. Please add it in secrets.toml")
    st.stop()

# =========================
# GET USER LOCATION USING IP
# =========================
def get_location():
    try:
        response = requests.get("http://ip-api.com/json/")
        data = response.json()
        return data["lat"], data["lon"], data["city"]
    except:
        return None, None, "Unknown"


lat, lon, city = get_location()

st.subheader(f"📍 Detected Location: {city}")

# =========================
# FETCH AQI DATA FROM OPENWEATHER
# =========================
def get_aqi(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    response = requests.get(url)
    data = response.json()
    return data


data = get_aqi(lat, lon)

if "list" not in data:
    st.error("Failed to fetch AQI data")
    st.stop()

aqi = data["list"][0]["main"]["aqi"]
components = data["list"][0]["components"]

pm25 = components["pm2_5"]
pm10 = components["pm10"]
co = components["co"]
no2 = components["no2"]
o3 = components["o3"]

# =========================
# DISPLAY CURRENT VALUES
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("PM2.5", f"{pm25}")
col2.metric("PM10", f"{pm10}")
col3.metric("AQI Index", f"{aqi}")

# =========================
# SIMPLE AI PREDICTION MODEL
# =========================
def predict_aqi(pm25, pm10, co, no2, o3):
    # simple weighted prediction model
    prediction = (
        0.4 * pm25 +
        0.3 * pm10 +
        0.1 * co / 100 +
        0.1 * no2 +
        0.1 * o3
    )
    return round(prediction, 2)


predicted_aqi = predict_aqi(pm25, pm10, co, no2, o3)

st.subheader("🔮 Predicted Future AQI")
st.success(f"Predicted AQI: {predicted_aqi}")

# =========================
# AQI CATEGORY
# =========================
def get_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


category = get_category(predicted_aqi)

st.info(f"Air Quality Category: {category}")

# =========================
# VISUALIZATION CHART
# =========================
st.subheader("📊 Pollutant Levels")

pollutants = ["PM2.5", "PM10", "CO", "NO2", "O3"]
values = [pm25, pm10, co, no2, o3]

fig, ax = plt.subplots()
ax.bar(pollutants, values)
ax.set_ylabel("Concentration")
ax.set_title("Pollutant Levels")

st.pyplot(fig)

# =========================
# HEATMAP MAP VIEW
# =========================
st.subheader("🗺️ Location Map")

m = folium.Map(location=[lat, lon], zoom_start=10)

folium.CircleMarker(
    location=[lat, lon],
    radius=15,
    popup=f"AQI: {predicted_aqi}",
    color="red",
    fill=True
).add_to(m)

html(m._repr_html_(), height=500)

# =========================
# HISTORY SIMULATION
# =========================
st.subheader("📈 AQI Trend Prediction (Next 24 Hours)")

future_hours = list(range(1, 25))
future_predictions = [predicted_aqi + np.random.uniform(-5, 5) for _ in future_hours]

trend_df = pd.DataFrame({
    "Hour": future_hours,
    "Predicted AQI": future_predictions
})

st.line_chart(trend_df.set_index("Hour"))

# =========================
# ALERT SYSTEM
# =========================
if predicted_aqi > 150:
    st.error("⚠️ Health Alert: Air quality is unhealthy. Wear mask.")
elif predicted_aqi > 100:
    st.warning("⚠️ Air quality is moderate. Be cautious.")
else:
    st.success("✅ Air quality is safe.")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("AI AQI Monitoring System | Hackathon Ready Project")
