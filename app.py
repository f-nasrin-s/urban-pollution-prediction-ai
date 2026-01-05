# ==============================
# Urban Pollution Prediction App
# Hackathon Stable Version 🚀
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import pytz
import folium
import shap
import matplotlib.pyplot as plt

from datetime import datetime
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium
from folium.plugins import HeatMap

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Urban AQI Predictor", layout="wide")
st.title("🌍 Smart Urban Pollution Prediction")
st.caption("ML-powered AQI prediction with live data")

ist = pytz.timezone("Asia/Kolkata")
st.caption(f"⏱️ Updated: {datetime.now(ist).strftime('%H:%M:%S IST')}")

# -------------------------------
# LOAD PRETRAINED MODEL
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("aqi_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    features = joblib.load("model_features.pkl")
    return model, encoders, features

model, label_encoders, feature_cols = load_model()

# -------------------------------
# MAP CLICK
# -------------------------------
st.subheader("🗺️ Select Location on Map")

base_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
map_data = st_folium(base_map, width=700, height=450)

if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    st.success(f"📍 Selected: {lat:.4f}, {lon:.4f}")

    # Reverse Geocode
    geolocator = Nominatim(user_agent="aqi_app")
    place = geolocator.reverse((lat, lon), language="en")
    city = place.raw["address"].get("city", "Unknown")

    st.info(f"🏙️ Location: {city}")

    # -------------------------------
    # LIVE AQI API
    # -------------------------------
    API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")

    if API_KEY:
        url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        res = requests.get(url).json()

        pm25 = res["list"][0]["components"]["pm2_5"]
        pm10 = res["list"][0]["components"]["pm10"]

        c1, c2 = st.columns(2)
        c1.metric("PM2.5", pm25)
        c2.metric("PM10", pm10)

        # -------------------------------
        # ML PREDICTION
        # -------------------------------
        live_input = {}

        for col in feature_cols:
            if col.lower() == "pm2.5":
                live_input[col] = pm25
            elif col.lower() == "pm10":
                live_input[col] = pm10
            elif col in label_encoders:
                live_input[col] = label_encoders[col].transform(
                    [label_encoders[col].classes_[0]]
                )[0]
            else:
                live_input[col] = 0

        live_df = pd.DataFrame([live_input])[feature_cols]
        prediction = model.predict(live_df)[0]

        st.subheader("🔮 Predicted AQI")
        st.metric("AQI", f"{prediction:.2f}")

        # -------------------------------
        # AQI VISUALIZATION (FIXED)
        # -------------------------------
        st.subheader("🗺️ AQI Visualization")

        lat = float(lat)
        lon = float(lon)
        prediction = float(prediction)

        m2 = folium.Map(location=[lat, lon], zoom_start=10)

        folium.CircleMarker(
            location=[lat, lon],
            radius=14,
            color="black",
            fill=True,
            fill_color="red",
            fill_opacity=0.8,
            popup=f"AQI: {round(prediction, 2)}"
        ).add_to(m2)

        HeatMap([[lat, lon, prediction]], radius=25).add_to(m2)

        st_folium(m2, width=700, height=450)


