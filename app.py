# ==========================================================
# AI-Driven Smart Urban Pollution Intelligence System
# Domain: Smart Cities & Urban Intelligence
# Hackathon Advanced Edition 🏆
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
import pytz

from datetime import datetime
from streamlit_folium import st_folium
from folium.plugins import HeatMap

from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Urban Pollution Intelligence", layout="wide")

st.title("🌆 Smart Urban Pollution Intelligence System")
st.caption("AI-powered AQI Prediction • Risk Analysis • Factor Attribution")

ist = pytz.timezone("Asia/Kolkata")
st.caption(f"⏱️ Live System Time: {datetime.now(ist).strftime('%d %b %Y | %H:%M:%S IST')}")

# ------------------------------
# FAST MODEL LOAD (CACHED)
# ------------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("TRAQID.csv")

    aqi_col = [c for c in df.columns if "aqi" in c.lower()][0]
    drop_cols = ["Image", "created_at", "Sequence", "aqi_cat", aqi_col]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[aqi_col]

    encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=80,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    return model, encoders, X.columns.tolist()

model, encoders, features = train_model()

# ------------------------------
# LOCATION MODE
# ------------------------------
st.subheader("📍 Location Intelligence")

mode = st.radio(
    "Select prediction mode:",
    ["📌 Auto Detect My Location", "🗺️ Predict Anywhere on Map"]
)

def auto_location():
    try:
        res = requests.get("https://ipapi.co/json/").json()
        return float(res["latitude"]), float(res["longitude"])
    except:
        return None, None

if mode == "📌 Auto Detect My Location":
    lat, lon = auto_location()
    if lat is None:
        st.error("Location detection failed")
        st.stop()
    st.success(f"Detected → {lat:.4f}, {lon:.4f}")
else:
    base_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    map_data = st_folium(base_map, height=450)
    if not map_data or not map_data.get("last_clicked"):
        st.info("Click on the map to select a location")
        st.stop()
    lat = float(map_data["last_clicked"]["lat"])
    lon = float(map_data["last_clicked"]["lng"])
    st.success(f"Selected → {lat:.4f}, {lon:.4f}")

# ------------------------------
# LIVE POLLUTION DATA
# ------------------------------
API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")
if not API_KEY:
    st.error("OpenWeather API key missing")
    st.stop()

url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
pollution = requests.get(url).json()
comp = pollution["list"][0]["components"]

pm25 = float(comp["pm2_5"])
pm10 = float(comp["pm10"])

c1, c2 = st.columns(2)
c1.metric("PM2.5", pm25)
c2.metric("PM10", pm10)

# ------------------------------
# ML PREDICTION
# ------------------------------
input_row = {}
for col in features:
    if col.lower() == "pm2.5":
        input_row[col] = pm25
    elif col.lower() == "pm10":
        input_row[col] = pm10
    else:
        input_row[col] = 0

predicted_aqi = float(model.predict(pd.DataFrame([input_row]))[0])

# ------------------------------
# ADVANCED AQI ANALYSIS
# ------------------------------
risk_score = min(100, round(predicted_aqi / 3))

if pm25 > pm10 and pm25 > 60:
    source = "🚗 Traffic & Combustion Sources"
elif pm10 > pm25 and pm10 > 80:
    source = "🏗️ Dust & Construction Activity"
else:
    source = "🏭 Mixed Urban Emissions"

# ------------------------------
# HEALTH IMPACT INDEX
# ------------------------------
health_impact = {
    "Children": "High" if predicted_aqi > 120 else "Moderate",
    "Elderly": "High" if predicted_aqi > 100 else "Moderate",
    "Asthma Patients": "Severe" if predicted_aqi > 90 else "Moderate"
}

# ------------------------------
# DISPLAY RESULTS
# ------------------------------
st.subheader("🔮 AI AQI Prediction")
st.metric("Predicted AQI", f"{predicted_aqi:.2f}")
st.metric("Urban Risk Score", f"{risk_score} / 100")

st.subheader("🧪 Pollution Intelligence")
st.write(f"**Likely Pollution Source:** {source}")
st.write(f"**Dominant Factor:** {'PM2.5' if pm25 > pm10 else 'PM10'}")

st.subheader("❤️ Health Impact Index")
for k, v in health_impact.items():
    st.write(f"- **{k}** → {v} risk")

# ------------------------------
# VISUAL HOTSPOT MAP
# ------------------------------
st.subheader("🗺️ Urban Pollution Hotspot")

m2 = folium.Map(location=[lat, lon], zoom_start=12)

folium.CircleMarker(
    [lat, lon],
    radius=20,
    fill=True,
    fill_color="red",
    fill_opacity=0.85,
    popup=f"AQI: {round(predicted_aqi,2)}"
).add_to(m2)

HeatMap([[lat, lon, predicted_aqi]], radius=35).add_to(m2)

st_folium(m2, height=450)

# ------------------------------
# SMART CITY DECISION ENGINE
# ------------------------------
st.subheader("🏙️ Smart City Decision Support")

if predicted_aqi > 150:
    st.error("""
    🚨 **Critical Pollution Zone Detected**
    - Trigger public health alert
    - Restrict heavy vehicle movement
    - Activate air purification zones
    """)
elif predicted_aqi > 100:
    st.warning("""
    ⚠️ **Moderate Risk Zone**
    - Monitor traffic congestion
    - Advise remote work policies
    - Increase green cover monitoring
    """)
else:
    st.success("""
    ✅ **Low Risk Zone**
    - Normal activities permitted
    - Encourage outdoor mobility
    """)

