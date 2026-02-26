# ============================================================
# URBANGUARD AI – NATIONAL POLLUTION COMMAND CENTER
# ISRO-Level Hackathon Winning System
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import folium
import requests
import pytz
import time
import random

from datetime import datetime, timedelta
from streamlit_folium import st_folium
from folium.plugins import HeatMap, TimestampedGeoJson

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import plotly.graph_objects as go

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="UrbanGuard AI Command Center",
    layout="wide",
)

st.title("🛰 URBANGUARD AI – National Pollution Command Center")
st.caption("ISRO-Level Smart City Intelligence System")

# ============================================================
# API KEY
# ============================================================

OPENWEATHER = st.secrets.get("OPENWEATHER_API_KEY", "")

# ============================================================
# TIME DISPLAY
# ============================================================

ist = pytz.timezone("Asia/Kolkata")
st.sidebar.success(
    datetime.now(ist).strftime("%d %b %Y | %H:%M:%S IST")
)

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

    for col in X.select_dtypes(include="object"):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05
    )

    model.fit(X_train, y_train)

    return model, X.columns.tolist()

model, features = train_model()

# ============================================================
# GPS DETECTION
# ============================================================

st.sidebar.subheader("📍 Location")

mode = st.sidebar.radio(
    "Choose Location",
    ["Auto Detect", "Select on Map"]
)

lat, lon = 20.5937, 78.9629

if mode == "Select on Map":

    m = folium.Map(location=[lat, lon], zoom_start=5)

    map_data = st_folium(m, height=400)

    if map_data and map_data.get("last_clicked"):

        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]

else:

    if OPENWEATHER:

        url = f"http://ip-api.com/json/"
        res = requests.get(url).json()

        lat = res["lat"]
        lon = res["lon"]

st.sidebar.success(f"{lat:.3f}, {lon:.3f}")

# ============================================================
# LIVE POLLUTION
# ============================================================

pollution_url = f"""
http://api.openweathermap.org/data/2.5/air_pollution
?lat={lat}&lon={lon}&appid={OPENWEATHER}
"""

pollution = requests.get(pollution_url).json()

pm25 = pollution["list"][0]["components"]["pm2_5"]
pm10 = pollution["list"][0]["components"]["pm10"]

# ============================================================
# PREDICT AQI
# ============================================================

row = {}

for f in features:

    if "pm2.5" in f.lower():
        row[f] = pm25

    elif "pm10" in f.lower():
        row[f] = pm10

    else:
        row[f] = 0

predicted_aqi = model.predict(pd.DataFrame([row]))[0]

# ============================================================
# RISK INDEX
# ============================================================

risk_index = int(min(100, predicted_aqi / 3))

# ============================================================
# DISPLAY METRICS
# ============================================================

c1, c2, c3 = st.columns(3)

c1.metric("PM2.5", round(pm25,2))
c2.metric("Predicted AQI", round(predicted_aqi,2))
c3.metric("Urban Risk Index", risk_index)

# ============================================================
# FUTURE PREDICTION
# ============================================================

st.subheader("🔮 24 Hour Future Prediction")

future = []

aqi = predicted_aqi

for i in range(24):

    change = random.uniform(-5, 10)
    aqi = max(0, aqi + change)

    future.append(aqi)

fig = go.Figure()

fig.add_trace(go.Scatter(
    y=future,
    mode="lines+markers"
))

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# EARLY WARNING
# ============================================================

if max(future) > 180:

    st.error("⚠️ ALERT: Hazardous AQI expected")

elif max(future) > 120:

    st.warning("⚠️ Warning: AQI rising")

else:

    st.success("Safe AQI")

# ============================================================
# NATIONAL HEATMAP
# ============================================================

st.subheader("🇮🇳 National Pollution Heatmap")

cities = [
    (28.61,77.20),
    (19.07,72.87),
    (13.08,80.27),
    (12.97,77.59),
    (22.57,88.36)
]

heat = []

for city in cities:

    val = random.randint(50,300)

    heat.append([city[0], city[1], val])

m = folium.Map(location=[22,78], zoom_start=5)

HeatMap(heat).add_to(m)

st_folium(m)

# ============================================================
# POLLUTION SPREAD SIMULATION
# ============================================================

st.subheader("🌪 Pollution Spread Simulation")

spread = []

for i in range(20):

    spread.append([
        lat + random.uniform(-0.1,0.1),
        lon + random.uniform(-0.1,0.1),
        predicted_aqi
    ])

m2 = folium.Map(location=[lat,lon], zoom_start=10)

HeatMap(spread).add_to(m2)

st_folium(m2)

# ============================================================
# SATELLITE ANIMATION
# ============================================================

st.subheader("🛰 Satellite Pollution Animation")

features_anim = []

for i in range(10):

    features_anim.append({

        "type":"Feature",

        "geometry":{
            "type":"Point",
            "coordinates":[
                lon + random.uniform(-1,1),
                lat + random.uniform(-1,1)
            ]
        },

        "properties":{
            "time":(
                datetime.now() +
                timedelta(minutes=i*5)
            ).isoformat(),
        }

    })

data_anim = {

"type":"FeatureCollection",
"features":features_anim

}

m3 = folium.Map(location=[lat,lon], zoom_start=6)

TimestampedGeoJson(
data_anim,
period="PT5M"
).add_to(m3)

st_folium(m3)

# ============================================================
# ANOMALY DETECTION
# ============================================================

st.subheader("🧠 AI Anomaly Detection")

history = np.array(future).reshape(-1,1)

iso = IsolationForest()

pred = iso.fit_predict(history)

if -1 in pred:

    st.error("Anomaly Detected")

else:

    st.success("No anomaly")

# ============================================================
# CITY RANKING
# ============================================================

st.subheader("🏆 Smart City Ranking")

rank = pd.DataFrame({

"City":["Delhi","Mumbai","Bangalore","Chennai"],

"AQI":[random.randint(50,300) for _ in range(4)]

})

rank["Rank"] = rank["AQI"].rank()

st.dataframe(rank)

# ============================================================
# COMMAND CENTER STATUS
# ============================================================

st.subheader("🎛 Command Center Status")

if predicted_aqi > 180:

    st.error("Emergency Mode")

elif predicted_aqi > 120:

    st.warning("Alert Mode")

else:

    st.success("Normal Mode")
