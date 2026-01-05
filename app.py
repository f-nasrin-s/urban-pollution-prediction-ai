# ==============================
# Smart Urban Pollution Prediction
# Hackathon-Stable Version 🚀
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
import pytz

from datetime import datetime
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium
from folium.plugins import HeatMap

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Smart Urban Pollution Prediction", layout="wide")

st.title("🌍 Smart Urban Pollution Prediction")
st.caption("ML-powered AQI prediction with live data")

ist = pytz.timezone("Asia/Kolkata")
st.caption(f"⏱️ Updated: {datetime.now(ist).strftime('%H:%M:%S IST')}")

# -------------------------------
# LOAD + TRAIN MODEL (CACHED)
# -------------------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("TRAQID.csv")

    # Detect AQI column
    aqi_col = [c for c in df.columns if "aqi" in c.lower()][0]

    drop_cols = ["Image", "created_at", "Sequence", "aqi_cat", aqi_col]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[aqi_col]

    label_encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ⚡ FAST model (hackathon-optimized)
    model = XGBRegressor(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, label_encoders, X.columns.tolist()

model, label_encoders, feature_cols = load_model()

# -------------------------------
# MAP CLICK
# -------------------------------
st.subheader("🗺️ Click Anywhere to Predict AQI")

base_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
map_data = st_folium(base_map, width=700, height=450)

# -------------------------------
# PROCESS LOCATION
# -------------------------------
if map_data and map_data.get("last_clicked"):
    lat = float(map_data["last_clicked"]["lat"])
    lon = float(map_data["last_clicked"]["lng"])

    st.success(f"📍 Location Selected: {lat:.4f}, {lon:.4f}")

    # Reverse geocoding
    geolocator = Nominatim(user_agent="aqi_app")
    place = geolocator.reverse((lat, lon), language="en")

    city = (
        place.raw["address"].get("city")
        or place.raw["address"].get("town")
        or place.raw["address"].get("state")
        or "Unknown"
    )

    st.info(f"🏙️ Area: {city}")

    # -------------------------------
    # LIVE AQI API
    # -------------------------------
    API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")

    if not API_KEY:
        st.error("❌ OpenWeather API key missing")
        st.stop()

    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    res = requests.get(url).json()

    components = res["list"][0]["components"]
    pm25 = float(components["pm2_5"])
    pm10 = float(components["pm10"])

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
    prediction = float(model.predict(live_df)[0])

    st.subheader("🔮 Predicted AQI")
    st.metric("AQI Value", f"{prediction:.2f}")

    # -------------------------------
    # AQI VISUALIZATION (SAFE)
    # -------------------------------
    st.subheader("🗺️ AQI Visualization")

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
