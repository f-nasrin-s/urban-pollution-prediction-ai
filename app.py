# ============================================================
# SMART CITY AI – URBAN POLLUTION COMMAND CENTER (HACKATHON VERSION)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
import pytz
import streamlit.components.v1 as components

from datetime import datetime, timedelta
from streamlit_folium import st_folium
from folium.plugins import HeatMap

from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart City Pollution AI Command Center",
    layout="wide"
)

st.title("🌍 Smart City AI – Pollution Command Center")
st.caption("Predictive Intelligence • Decision Optimization • Smart Governance")

ist = pytz.timezone("Asia/Kolkata")
st.caption(datetime.now(ist).strftime("⏱️ %d %b %Y | %H:%M:%S IST"))

# ============================================================
# LOAD MODEL
# ============================================================

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

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, X.columns.tolist()

model, features = train_model()

# ============================================================
# GPS FUNCTION
# ============================================================

def gps_button():

    html = """
    <script>
    navigator.geolocation.getCurrentPosition(
        (pos) => {
            const coords = pos.coords.latitude + "," + pos.coords.longitude;
            const input = window.parent.document.querySelector(
                'input[data-testid="stTextInput"]'
            );
            input.value = coords;
            input.dispatchEvent(new Event('input', { bubbles: true }));
        }
    );
    </script>
    """

    components.html(html)

# ============================================================
# LOCATION SELECTION
# ============================================================

st.header("📍 Location Intelligence")

mode = st.radio(
    "Choose Location Method",
    ["Map Selection", "Auto Detect GPS"]
)

lat, lon = None, None

if mode == "Auto Detect GPS":

    gps_val = st.text_input("GPS Output")

    if st.button("Detect Location"):
        gps_button()

    if gps_val:
        lat, lon = map(float, gps_val.split(","))

if mode == "Map Selection":

    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

    data = st_folium(m)

    if data and data.get("last_clicked"):

        lat = data["last_clicked"]["lat"]
        lon = data["last_clicked"]["lng"]

if lat is None:
    st.stop()

# ============================================================
# OPENWEATHER API
# ============================================================

API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")

geo_url = f"https://api.openweathermap.org/geo/1.0/reverse?lat={lat}&lon={lon}&limit=1&appid={API_KEY}"

geo = requests.get(geo_url).json()

city = geo[0]["name"] if geo else "Unknown"

st.success(f"Location: {city}")

poll_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"

poll = requests.get(poll_url).json()["list"][0]["components"]

pm25 = poll["pm2_5"]
pm10 = poll["pm10"]

# ============================================================
# DISPLAY METRICS
# ============================================================

col1, col2 = st.columns(2)

col1.metric("PM2.5", round(pm25, 2))
col2.metric("PM10", round(pm10, 2))

# ============================================================
# AI AQI PREDICTION
# ============================================================

row = {}

for col in features:

    if "pm2.5" in col.lower():
        row[col] = pm25

    elif "pm10" in col.lower():
        row[col] = pm10

    else:
        row[col] = 0

predicted_aqi = float(model.predict(pd.DataFrame([row]))[0])

st.header("🔮 AI Prediction")

st.metric("Predicted AQI", round(predicted_aqi, 2))

# ============================================================
# AQI CATEGORY
# ============================================================

def classify(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Moderate"

    elif aqi <= 150:
        return "Unhealthy (Sensitive)"

    elif aqi <= 200:
        return "Unhealthy"

    elif aqi <= 300:
        return "Very Unhealthy"

    else:
        return "Hazardous"

category = classify(predicted_aqi)

st.metric("Risk Category", category)

# ============================================================
# FORECAST
# ============================================================

st.header("📈 24 Hour Forecast")

future = []

for i in range(24):

    sim_pm25 = pm25 * np.random.uniform(0.9, 1.1)
    sim_pm10 = pm10 * np.random.uniform(0.9, 1.1)

    temp = row.copy()

    for k in temp:

        if "pm2.5" in k.lower():
            temp[k] = sim_pm25

        elif "pm10" in k.lower():
            temp[k] = sim_pm10

    future.append(model.predict(pd.DataFrame([temp]))[0])

forecast = pd.DataFrame({"AQI": future})

st.line_chart(forecast)

# ============================================================
# POLICY OPTIMIZER
# ============================================================

st.header("🧠 Policy Optimizer")

best = predicted_aqi
policy = ""

for t in range(0, 50, 10):

    for c in range(0, 50, 10):

        sim25 = pm25 * (1 - t/100)
        sim10 = pm10 * (1 - c/100)

        temp = row.copy()

        for k in temp:

            if "pm2.5" in k.lower():
                temp[k] = sim25

            elif "pm10" in k.lower():
                temp[k] = sim10

        val = model.predict(pd.DataFrame([temp]))[0]

        if val < best:

            best = val
            policy = f"Traffic ↓{t}%, Construction ↓{c}%"

st.success(policy)

# ============================================================
# HOTSPOT MAP
# ============================================================

st.header("🗺️ Pollution Intelligence Map")

points = []

for i in range(25):

    points.append([
        lat + np.random.uniform(-0.02, 0.02),
        lon + np.random.uniform(-0.02, 0.02),
        predicted_aqi + np.random.uniform(-20, 20)
    ])

map2 = folium.Map(location=[lat, lon], zoom_start=12)

HeatMap(points).add_to(map2)

st_folium(map2)

# ============================================================
# HEALTH ANALYSIS
# ============================================================

st.header("❤️ Health Risk")

if predicted_aqi > 200:

    st.error("Severe Risk")

elif predicted_aqi > 100:

    st.warning("Moderate Risk")

else:

    st.success("Low Risk")

# ============================================================
# REPORT DOWNLOAD
# ============================================================

report = pd.DataFrame({

    "City":[city],
    "AQI":[predicted_aqi],
    "PM2.5":[pm25],
    "PM10":[pm10],
    "Category":[category]

})

st.download_button(

    "Download Report",
    report.to_csv(index=False),
    "pollution_report.csv"

)

# ============================================================
# CHATBOT
# ============================================================

st.header("💬 AI Assistant")

q = st.text_input("Ask")

if q:

    if "aqi" in q.lower():

        st.write(predicted_aqi)

    elif "risk" in q.lower():

        st.write(category)

    else:

        st.write("Ask about AQI or risk")

# ============================================================
# END
# ============================================================
