# ============================================================
# SMART CITY AI – URBAN POLLUTION COMMAND CENTER
# Hackathon Winning Version 🏆
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Smart City Pollution AI", layout="wide")
st.title("🏙️ Smart City Pollution AI – Command Center")
st.caption("AI Prediction • Factor Attribution • Health Risk • Policy Simulation • Semantic Chatbot")

ist = pytz.timezone("Asia/Kolkata")
st.caption(f"⏱️ System Time: {datetime.now(ist).strftime('%d %b %Y | %H:%M:%S IST')}")

# ------------------ LOAD PRE-TRAINED MODEL ------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("aqi_model.pkl")
        encoders = joblib.load("label_encoders.pkl")
        features = joblib.load("feature_cols.pkl")
        return model, encoders, features
    except:
        st.warning("Pre-trained model not found. Please train model first.")
        st.stop()

model, encoders, features = load_model()

# ------------------ LOCATION SELECTION ------------------
st.subheader("📍 Location Selection")

def auto_location():
    try:
        res = requests.get("https://ipapi.co/json/", timeout=5).json()
        return float(res["latitude"]), float(res["longitude"])
    except:
        return None, None

mode = st.radio(
    "Choose location method:",
    ["📌 Auto Detect My Location", "🗺️ Select Location on Map"]
)

if mode == "📌 Auto Detect My Location":
    lat, lon = auto_location()
    if lat is None:
        st.warning("Auto detection unavailable — please select location on map.")
        mode = "🗺️ Select Location on Map"
    else:
        st.success(f"Detected → {lat:.4f}, {lon:.4f}")

if mode == "🗺️ Select Location on Map":
    base_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    map_data = st_folium(base_map, height=420)
    if not map_data or not map_data.get("last_clicked"):
        st.info("Click anywhere on the map to select location")
        st.stop()
    lat = float(map_data["last_clicked"]["lat"])
    lon = float(map_data["last_clicked"]["lng"])
    st.success(f"Selected → {lat:.4f}, {lon:.4f}")

# ------------------ LIVE AIR POLLUTION DATA ------------------
API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")
if not API_KEY:
    st.error("OpenWeather API key missing")
    st.stop()

url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
pollution = requests.get(url).json()
components = pollution["list"][0]["components"]

pm25 = float(components["pm2_5"])
pm10 = float(components["pm10"])

c1, c2 = st.columns(2)
c1.metric("PM2.5 (µg/m³)", pm25)
c2.metric("PM10 (µg/m³)", pm10)

# ------------------ ML PREDICTION WITH DERIVED FEATURES ------------------
row = {}
for col in features:
    if col.lower() == "pm2.5":
        row[col] = pm25
    elif col.lower() == "pm10":
        row[col] = pm10
    elif col.lower() == "pm_ratio":
        row[col] = pm25 / (pm10 + 1e-5)
    elif col.lower() == "hour_sin":
        hour = datetime.now().hour
        row[col] = np.sin(2*np.pi*hour/24)
    elif col.lower() == "hour_cos":
        hour = datetime.now().hour
        row[col] = np.cos(2*np.pi*hour/24)
    elif col in encoders:
        row[col] = encoders[col].transform([encoders[col].classes_[0]])[0]
    else:
        row[col] = 0

predicted_aqi = float(model.predict(pd.DataFrame([row]))[0])

# ------------------ HEALTH & RISK ------------------
risk_score = min(100, round(predicted_aqi / 3))
source = "🚗 Traffic & Combustion" if pm25 > 1.3*pm10 else ("🏗️ Dust / Construction" if pm10>1.3*pm25 else "🏭 Mixed Emissions")
health_risk = {"Children": "High" if predicted_aqi>120 else "Moderate",
               "Elderly": "High" if predicted_aqi>100 else "Moderate",
               "Asthma Patients": "Severe" if predicted_aqi>90 else "Moderate"}

# ------------------ DISPLAY RESULTS ------------------
st.subheader("🔮 AQI Prediction")
st.metric("Predicted AQI", f"{predicted_aqi:.2f}")
st.metric("Urban Risk Index", f"{risk_score}/100")

st.subheader("🧪 Factor Attribution")
st.write(f"**Dominant Pollutant:** {'PM2.5' if pm25>pm10 else 'PM10'}")
st.write(f"**Likely Source:** {source}")

st.subheader("❤️ Health Impact")
for k,v in health_risk.items():
    st.write(f"- **{k}** → {v} risk")

# ------------------ WHAT-IF SIMULATION ------------------
st.subheader("🧠 What-If Simulation")
traffic = st.slider("🚗 Traffic Reduction (%)", 0, 50, 0)
construction = st.slider("🏗️ Construction Reduction (%)", 0, 50, 0)

sim_pm25 = pm25*(1-traffic/200)
sim_pm10 = pm10*(1-construction/200)
sim_row = row.copy()
sim_row[next(k for k in row if k.lower()=="pm2.5")] = sim_pm25
sim_row[next(k for k in row if k.lower()=="pm10")] = sim_pm10
sim_row["PM_Ratio"] = sim_pm25 / (sim_pm10 + 1e-5)
sim_aqi = model.predict(pd.DataFrame([sim_row]))[0]
st.metric("Simulated AQI", round(sim_aqi,2))

# ------------------ HOTSPOT MAP ------------------
st.subheader("🗺️ Pollution Hotspot Map")
m2 = folium.Map(location=[lat, lon], zoom_start=12)
color = "#00FF00" if predicted_aqi<=50 else "#FFFF00" if predicted_aqi<=100 else "#FFA500" if predicted_aqi<=150 else "#FF0000" if predicted_aqi<=200 else "#800080"
folium.CircleMarker([lat, lon], radius=20, fill=True, fill_color=color, fill_opacity=0.85,
                    popup=f"AQI: {round(predicted_aqi,2)}").add_to(m2)
HeatMap([[lat, lon, predicted_aqi]], radius=35).add_to(m2)
st_folium(m2, height=420)

# ------------------ DYNAMIC AI RECOMMENDATIONS ------------------
st.subheader("🤖 AI Recommended Actions")
if sim_aqi > 180:
    st.error("🚨 SEVERE POLLUTION ALERT\n• Emergency advisory\n• Odd-even traffic\n• Stop construction\n• Deploy mobile purifiers")
elif sim_aqi > 120:
    st.warning("⚠️ MODERATE-HIGH POLLUTION\n• Limit outdoor activity\n• Control traffic\n• Monitor hotspots")
else:
    st.success("✅ LOW POLLUTION ZONE\n• Normal activities allowed\n• Promote green mobility")

# ------------------ SEMANTIC CHATBOT ------------------
st.subheader("💬 Ask the Smart City AI")
user_q = st.text_input("Type your question here:")

# ------------------ FAQ KNOWLEDGE BASE ------------------
faq_data = [
    {"q":"What is AQI?","a":"AQI (Air Quality Index) measures overall air pollution."},
    {"q":"What is PM2.5?","a":"Fine particulate matter smaller than 2.5 microns, harmful to lungs."},
    {"q":"What is PM10?","a":"Coarse particulate matter smaller than 10 microns, affects respiratory system."},
    {"q":"Is outdoor activity safe?","a":f"Predicted AQI at this location is {round(predicted_aqi,2)}. Health risk for children, elderly, asthma patients is {', '.join([f'{k}: {v}' for k,v in health_risk.items()])}."},
    {"q":"What health risks exist?","a":"High AQI affects children, elderly, and asthma patients. Stay indoors in severe conditions."},
    {"q":"How can pollution be reduced?","a":"Reduce traffic, limit construction, plant trees, use air purifiers."},
    {"q":"Which pollutant is dominant?","a":'PM2.5' if pm25>pm10 else 'PM10'},
    {"q":"Emergency measures?","a":"Deploy air purifiers, reduce traffic, stop construction, avoid outdoor activity in high pollution."}
]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
faq_questions = [item["q"] for item in faq_data]
faq_vectors = vectorizer.fit_transform(faq_questions)

def semantic_answer(user_input):
    user_vec = vectorizer.transform([user_input])
    sim_scores = cosine_similarity(user_vec, faq_vectors)[0]
    best_idx = sim_scores.argmax()
    if sim_scores[best_idx] < 0.2:
        return "🤖 Sorry, I am not sure. Ask about AQI, PM2.5, PM10, or health risks."
    return faq_data[best_idx]["a"]

if user_q:
    ans = semantic_answer(user_q)
    st.info(ans)
