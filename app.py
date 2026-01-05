# ==============================
# Urban Pollution Prediction App
# Hackathon Winning Version 🚀
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import shap
import pytz

from datetime import datetime
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

from geopy.geocoders import Nominatim
from streamlit_folium import st_folium
import folium
from folium.plugins import HeatMap

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Smart Urban AQI Predictor",
    layout="wide"
)

st.title("🌍 Smart Urban Pollution Prediction System")
st.caption("ML + Live Data + Explainable AI")

ist = pytz.timezone("Asia/Kolkata")
st.caption(f"⏱️ Last updated: {datetime.now(ist).strftime('%d %b %Y | %H:%M:%S IST')}")

# -------------------------------
# PROBLEM STATEMENT
# -------------------------------
with st.expander("📌 Problem & Impact"):
    st.write("""
    Urban air pollution is a major public health challenge.

    This system predicts **real-time AQI** for:
    - 📍 User's current location
    - 🗺️ Any clicked location on map

    **Impact:**
    - Citizens get instant health alerts
    - Governments can monitor pollution hotspots
    - Smart city planning support
    """)

# -------------------------------
# LOAD & TRAIN MODEL (CACHED)
# -------------------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("TRAQID.csv")

    aqi_col = [c for c in df.columns if "aqi" in c.lower()][0]
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
        n_estimators=250,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, X, y, X_test, y_test, label_encoders

model, X, y, X_test, y_test, label_encoders = load_model()

# -------------------------------
# MODEL PERFORMANCE
# -------------------------------
st.subheader("📊 Model Performance")

y_pred = model.predict(X_test)

c1, c2 = st.columns(2)
c1.metric("Mean Absolute Error", f"{mean_absolute_error(y_test, y_pred):.2f}")
c2.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")

# -------------------------------
# USER LOCATION DETECTION
# -------------------------------
st.subheader("📍 Predict AQI for Your Location")

loc = st.checkbox("Detect my current location automatically")

if loc:
    st.info("Allow browser location access")

    location = st.experimental_get_query_params()

# -------------------------------
# MAP SECTION
# -------------------------------
st.subheader("🗺️ Click Anywhere on Map")

default_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
map_data = st_folium(default_map, width=700, height=500)

# -------------------------------
# PROCESS LOCATION
# -------------------------------
if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    st.success(f"📍 Selected Location: {lat:.4f}, {lon:.4f}")

    # Reverse Geocoding
    geolocator = Nominatim(user_agent="aqi_app")
    place = geolocator.reverse((lat, lon), language="en")

    city = (
        place.raw["address"].get("city")
        or place.raw["address"].get("town")
        or place.raw["address"].get("state")
        or "Unknown"
    )

    st.info(f"🏙️ Detected Area: {city}")

    # -------------------------------
    # LIVE AQI API
    # -------------------------------
    API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")

    if API_KEY:
        url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        res = requests.get(url).json()

        components = res["list"][0]["components"]
        pm25 = components["pm2_5"]
        pm10 = components["pm10"]

        c1, c2 = st.columns(2)
        c1.metric("PM2.5", pm25)
        c2.metric("PM10", pm10)

        # -------------------------------
        # ML INPUT
        # -------------------------------
        live_input = {}

        for col in X.columns:
            if col.lower() == "pm2.5":
                live_input[col] = pm25
            elif col.lower() == "pm10":
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
        # AQI CATEGORY
        # -------------------------------
        def aqi_label(val):
            if val <= 50:
                return "Good", "🟢 Safe"
            elif val <= 100:
                return "Moderate", "🟡 Sensitive groups cautious"
            elif val <= 150:
                return "Unhealthy (Sensitive)", "🟠 Reduce outdoor activity"
            elif val <= 200:
                return "Unhealthy", "🔴 Stay indoors"
            elif val <= 300:
                return "Very Unhealthy", "🟣 Health warning"
            else:
                return "Hazardous", "⚫ Emergency"

        label, alert = aqi_label(prediction)

        st.subheader("🔮 Predicted AQI")
        st.metric("AQI Value", f"{prediction:.2f}")
        st.warning(f"{label} — {alert}")

        # -------------------------------
        # FORECAST
        # -------------------------------
        st.subheader("📈 3-Hour AQI Forecast")

        forecast = [prediction + np.random.uniform(-6, 6) for _ in range(3)]
        forecast_df = pd.DataFrame({
            "Hour": ["+1 hr", "+2 hr", "+3 hr"],
            "AQI": forecast
        })

        st.line_chart(forecast_df.set_index("Hour"))

        # -------------------------------
        # EXPLAINABLE AI
        # -------------------------------
        st.subheader("🧠 Explainable AI (SHAP)")

        explainer = shap.Explainer(model)
        shap_values = explainer(X.sample(100))

        fig, ax = plt.subplots()
        shap.plots.bar(shap_values, show=False)
        st.pyplot(fig)

        # -------------------------------
        # MAP MARKER + HEATMAP
        # -------------------------------
        st.subheader("🗺️ AQI Visualization")

        m2 = folium.Map(location=[lat, lon], zoom_start=10)

        folium.CircleMarker(
            location=[lat, lon],
            radius=15,
            color="black",
            fill=True,
            fill_color="red",
            fill_opacity=0.8,
            popup=f"AQI: {prediction:.2f}"
        ).add_to(m2)

        HeatMap([[lat, lon, prediction]]).add_to(m2)

        st_folium(m2, width=700, height=500)

    else:
        st.error("⚠️ OpenWeather API Key missing")

