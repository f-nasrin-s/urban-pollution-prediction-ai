# ============================================================
# SMART CITY AI – URBAN POLLUTION COMMAND CENTER
# Domain: Smart Cities & Urban Intelligence
# ============================================================

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

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Smart City Pollution AI", layout="wide")
st.title("🏙️ Smart City Pollution AI – Command Center")
st.caption("AI Prediction • Health Risk • Policy Simulation • Interactive Insights")

ist = pytz.timezone("Asia/Kolkata")
st.caption(f"⏱️ {datetime.now(ist).strftime('%d %b %Y | %H:%M:%S IST')}")

# ------------------ MODEL TRAINING ------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("TRAQID.csv")

    aqi_col = [c for c in df.columns if "aqi" in c.lower()][0]
    drop_cols = ["Image", "created_at", "Sequence", "aqi_cat", aqi_col]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[aqi_col]

    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42
    )
    model.fit(X_train, y_train)

    return model, X.columns.tolist()

model, features = train_model()

# ------------------ LOCATION SELECTION ------------------
st.subheader("📍 Select Location (India)")

m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
map_data = st_folium(m, height=420)

if not map_data or not map_data.get("last_clicked"):
    st.info("Click anywhere on the map to select a location")
    st.stop()

lat = map_data["last_clicked"]["lat"]
lon = map_data["last_clicked"]["lng"]

# ------------------ CITY NAME ------------------
API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")

geo_url = f"https://api.openweathermap.org/geo/1.0/reverse?lat={lat}&lon={lon}&limit=1&appid={API_KEY}"
geo = requests.get(geo_url).json()
city = geo[0]["name"] if geo else "Unknown City"

st.success(f"📍 Location: {city}, India")

# ------------------ LIVE POLLUTION DATA ------------------
poll_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
poll = requests.get(poll_url).json()["list"][0]["components"]

pm25 = poll["pm2_5"]
pm10 = poll["pm10"]

c1, c2 = st.columns(2)
c1.metric("PM2.5 (µg/m³)", round(pm25, 2))
c2.metric("PM10 (µg/m³)", round(pm10, 2))

# ------------------ AQI PREDICTION ------------------
row = {}
for col in features:
    if "pm2.5" in col.lower():
        row[col] = pm25
    elif "pm10" in col.lower():
        row[col] = pm10
    else:
        row[col] = 0

predicted_aqi = float(model.predict(pd.DataFrame([row]))[0])
risk_score = min(100, int(predicted_aqi / 3))

# ------------------ RESULTS ------------------
st.subheader("🔮 AQI Prediction")
st.metric("Predicted AQI", round(predicted_aqi, 2))
st.metric("Urban Risk Index", f"{risk_score}/100")

# ------------------ HEALTH IMPACT ANALYSIS ------------------
st.subheader("❤️ Health Impact Assessment")

health_impact = {
    "Children": "Low",
    "Elderly": "Low",
    "Asthma Patients": "Low",
    "General Public": "Low"
}

if predicted_aqi > 180:
    health_impact.update({
        "Children": "Severe",
        "Elderly": "Severe",
        "Asthma Patients": "Very Severe",
        "General Public": "High"
    })
elif predicted_aqi > 120:
    health_impact.update({
        "Children": "High",
        "Elderly": "High",
        "Asthma Patients": "Severe",
        "General Public": "Moderate"
    })
elif predicted_aqi > 80:
    health_impact.update({
        "Children": "Moderate",
        "Elderly": "Moderate",
        "Asthma Patients": "High",
        "General Public": "Low"
    })

for group, risk in health_impact.items():
    if risk in ["Very Severe", "Severe"]:
        st.error(f"**{group}** → {risk} risk")
    elif risk == "High":
        st.warning(f"**{group}** → {risk} risk")
    elif risk == "Moderate":
        st.info(f"**{group}** → {risk} risk")
    else:
        st.success(f"**{group}** → {risk} risk")

# ------------------ WHAT-IF SIMULATION ------------------
st.subheader("🧠 What-If Pollution Control Simulation")

traffic = st.slider("🚗 Traffic Reduction (%)", 0, 50, 0)
construction = st.slider("🏗️ Construction Control (%)", 0, 50, 0)

sim_pm25 = pm25 * (1 - traffic / 100)
sim_pm10 = pm10 * (1 - construction / 100)

sim_row = row.copy()
for k in sim_row:
    if "pm2.5" in k.lower():
        sim_row[k] = sim_pm25
    if "pm10" in k.lower():
        sim_row[k] = sim_pm10

sim_aqi = model.predict(pd.DataFrame([sim_row]))[0]
st.metric("Simulated AQI", round(sim_aqi, 2))

# ------------------ HOTSPOT MAP ------------------
st.subheader("🗺️ Pollution Hotspot Map")

m2 = folium.Map(location=[lat, lon], zoom_start=12)
HeatMap([[lat, lon, predicted_aqi]], radius=35).add_to(m2)
folium.CircleMarker(
    [lat, lon],
    radius=15,
    fill=True,
    fill_color="red",
    popup=f"AQI: {round(predicted_aqi,2)}"
).add_to(m2)

st_folium(m2, height=420)

# ------------------ AI RECOMMENDATIONS ------------------
st.subheader("🤖 AI Recommended Actions")

if predicted_aqi > 180:
    st.error("🚨 Emergency: Stop outdoor activities, restrict traffic, halt construction")
elif predicted_aqi > 120:
    st.warning("⚠️ Advisory: Reduce traffic, promote remote work, monitor hotspots")
else:
    st.success("✅ Safe: Encourage green mobility and public transport")

# ------------------ CHATBOT ------------------
st.subheader("💬 Ask the Pollution AI")

q = st.text_input("Ask about AQI, PM2.5, PM10, or health risk")

if q:
    q = q.lower()
    if "aqi" in q:
        st.info(f"Predicted AQI at this location is {round(predicted_aqi,2)}")
    elif "pm2.5" in q:
        st.info(f"PM2.5 level is {round(pm25,2)} µg/m³")
    elif "pm10" in q:
        st.info(f"PM10 level is {round(pm10,2)} µg/m³")
    elif "health" in q or "risk" in q:
        st.info(", ".join([f"{k}: {v}" for k,v in health_impact.items()]))
    else:
        st.info("Try asking about AQI, PM2.5, PM10 or health risks.")
