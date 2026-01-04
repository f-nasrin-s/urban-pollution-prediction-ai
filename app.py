# ==============================
# Urban Pollution Prediction App
# Interactive Map Click Version
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import requests
from datetime import datetime
import pytz

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

from streamlit_autorefresh import st_autorefresh
from geopy.geocoders import Nominatim

# NEW: folium map
from streamlit_folium import st_folium
import folium

# -------------------------------
# AUTO REFRESH
# -------------------------------
st_autorefresh(interval=60 * 1000, key="refresh")

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Urban AQI Map Prediction", layout="wide")
st.title("Urban Pollution Prediction 🚦")
st.write("ML-powered AQI prediction with live data and interactive map click location")

ist = pytz.timezone("Asia/Kolkata")
st.caption(f"⏱️ Last updated at {datetime.now(ist).strftime('%H:%M:%S IST')}")
st.caption("📡 Live AQI source: OpenWeatherMap API")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("TRAQID.csv")
aqi_col = [c for c in df.columns if "aqi" in c.lower()][0]

# -------------------------------
# PREPARE FEATURES
# -------------------------------
drop_cols = ["Image", "created_at", "Sequence", "aqi_cat", aqi_col]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df[aqi_col]

label_encoders = {}
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# -------------------------------
# INTERACTIVE MAP CLICK
# -------------------------------
st.subheader("🗺️ Click on the Map to Select Location")

# Default map
m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

# Display map and capture clicks
map_data = st_folium(m, width=700, height=500)

if map_data and map_data.get("last_clicked"):
    clicked_lat = map_data["last_clicked"]["lat"]
    clicked_lon = map_data["last_clicked"]["lng"]

    st.success(f"Selected Location: {clicked_lat:.4f}, {clicked_lon:.4f}")

    # -------------------------------
    # AUTO CITY DETECTION
    # -------------------------------
    geolocator = Nominatim(user_agent="aqi_map_app")
    place = geolocator.reverse((clicked_lat, clicked_lon), language="en")
    city = (
        place.raw["address"].get("city")
        or place.raw["address"].get("town")
        or place.raw["address"].get("state")
        or "Unknown"
    )
    st.info(f"📌 Detected City: {city}")

    # -------------------------------
    # LIVE AQI FROM API
    # -------------------------------
    API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")

    if API_KEY:
        url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={clicked_lat}&lon={clicked_lon}&appid={API_KEY}"
        res = requests.get(url).json()
        components = res["list"][0]["components"]

        pm25 = components["pm2_5"]
        pm10 = components["pm10"]

        st.metric("PM2.5", pm25)
        st.metric("PM10", pm10)

        # -------------------------------
        # ML PREDICTION
        # -------------------------------
        live_input = {}

        for col in X.columns:
            if col == "PM2.5":
                live_input[col] = pm25
            elif col == "PM10":
                live_input[col] = pm10
            elif col in label_encoders:
                live_input[col] = label_encoders[col].transform(
                    [label_encoders[col].classes_[0]]
                )[0]
            else:
                live_input[col] = X[col].mean()

        live_df = pd.DataFrame([live_input])
        prediction = model.predict(live_df)[0]

        # -------------------------------
        # AQI LABEL + ALERT
        # -------------------------------
        def aqi_label(val):
            if val <= 50:
                return "Good", "🟢 Safe"
            elif val <= 100:
                return "Moderate", "🟡 Sensitive people be cautious"
            elif val <= 150:
                return "Unhealthy (Sensitive)", "🟠 Avoid long exposure"
            elif val <= 200:
                return "Unhealthy", "🔴 Stay indoors"
            elif val <= 300:
                return "Very Unhealthy", "🟣 Health warning"
            else:
                return "Hazardous", "⚫ Emergency"

        label, alert = aqi_label(prediction)

        st.subheader("🔮 AQI Prediction for Selected Location")
        st.metric("Predicted AQI", f"{prediction:.2f}")
        st.warning(f"{label} — {alert}")

        health_tips = {
            "Good": "Enjoy outdoor activities 🌿",
            "Moderate": "Limit outdoor exertion",
            "Unhealthy (Sensitive)": "Children & elderly should stay indoors",
            "Unhealthy": "Avoid outdoor activity",
            "Very Unhealthy": "Wear masks and use air purifiers",
            "Hazardous": "Emergency conditions"
        }
        st.error(f"🚨 Health Advisory: {health_tips[label]}")

        # -------------------------------
        # COLOR FUNCTION
        # -------------------------------
        def aqi_color(aqi):
            if aqi <= 50:
                return [0, 255, 0]
            elif aqi <= 100:
                return [255, 255, 0]
            elif aqi <= 150:
                return [255, 165, 0]
            elif aqi <= 200:
                return [255, 0, 0]
            else:
                return [128, 0, 128]

        # -------------------------------
        # SHOW MAP WITH MARKER
        # -------------------------------
        m2 = folium.Map(location=[clicked_lat, clicked_lon], zoom_start=10)
        folium.CircleMarker(
            location=[clicked_lat, clicked_lon],
            radius=15,
            color='black',
            fill=True,
            fill_color=f"#{''.join([format(c,'02x') for c in aqi_color(prediction)])}",
            fill_opacity=0.8,
            popup=f"AQI: {prediction:.2f} — {label}"
        ).add_to(m2)

        st.subheader("🗺️ Pollution Marker for Selected Location")
        st_folium(m2, width=700, height=500)
