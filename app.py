# ============================================================
# 🌍 UrbanGuard AI – National Level Smart City System
# With Gemini AI, Satellite Simulation, Multi-City Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
import pytz
import os
import plotly.express as px

from datetime import datetime
from streamlit_folium import st_folium
from folium.plugins import HeatMap

from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import google.generativeai as genai

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(page_title="UrbanGuard AI", layout="wide")

st.title("🌍 UrbanGuard AI – National Smart City Pollution Intelligence System")

# ============================================================
# API KEYS
# ============================================================

OPENWEATHER = st.secrets["83350e70e4de15a991533bdd03e028ab"]
GEMINI = st.secrets["AIzaSyC07SeyE7T6oTxP4PKk_k8WVBj4ATE_bGg"]

genai.configure(api_key=GEMINI)

gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ============================================================
# TIME
# ============================================================

ist = pytz.timezone("Asia/Kolkata")
time_now = datetime.now(ist)

st.caption(time_now.strftime("%d %b %Y | %H:%M:%S IST"))

# ============================================================
# MODEL TRAIN
# ============================================================

@st.cache_resource
def train():

    df = pd.read_csv("TRAQID.csv")

    target = [c for c in df.columns if "aqi" in c.lower()][0]

    drop = ["Image", "created_at", "Sequence", "aqi_cat", target]

    X = df.drop(columns=[c for c in drop if c in df.columns])

    y = df[target]

    for col in X.select_dtypes(include="object"):

        le = LabelEncoder()

        X[col] = le.fit_transform(X[col])

    X_train, _, y_train, _ = train_test_split(X, y)

    model = XGBRegressor()

    model.fit(X_train, y_train)

    return model, X.columns


model, features = train()

# ============================================================
# MULTI CITY LIVE DASHBOARD
# ============================================================

st.header("🌏 India Live AQI Dashboard")

cities = {

    "Delhi": (28.61,77.20),
    "Mumbai": (19.07,72.87),
    "Bangalore": (12.97,77.59),
    "Chennai": (13.08,80.27),
    "Kolkata": (22.57,88.36)

}

data = []

for city,(lat,lon) in cities.items():

    url=f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER}"

    res=requests.get(url).json()

    aqi=res["list"][0]["main"]["aqi"]*50

    data.append([city,aqi,lat,lon])

df_live=pd.DataFrame(data,columns=["City","AQI","lat","lon"])

st.dataframe(df_live)

fig=px.scatter_mapbox(df_live,
lat="lat",
lon="lon",
color="AQI",
size="AQI",
hover_name="City",
zoom=4)

fig.update_layout(mapbox_style="open-street-map")

st.plotly_chart(fig)

# ============================================================
# MAP SELECT
# ============================================================

st.header("📍 Select Any Location")

m=folium.Map(location=[20,78],zoom_start=5)

map_data=st_folium(m)

if not map_data or not map_data.get("last_clicked"):

    st.stop()

lat=map_data["last_clicked"]["lat"]
lon=map_data["last_clicked"]["lng"]

# ============================================================
# GET LIVE POLLUTION
# ============================================================

url=f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER}"

poll=requests.get(url).json()["list"][0]["components"]

pm25=poll["pm2_5"]
pm10=poll["pm10"]

st.metric("PM2.5",pm25)
st.metric("PM10",pm10)

# ============================================================
# AI PREDICTION
# ============================================================

row={}

for f in features:

    if "pm2.5" in f.lower():

        row[f]=pm25

    elif "pm10" in f.lower():

        row[f]=pm10

    else:

        row[f]=0

pred=model.predict(pd.DataFrame([row]))[0]

st.metric("Predicted AQI",round(pred,2))

# ============================================================
# SATELLITE SIMULATION
# ============================================================

st.header("🛰️ Satellite Pollution Simulation")

satellite_noise=np.random.normal(0,10)

satellite_aqi=pred+satellite_noise

st.metric("Satellite AQI",round(satellite_aqi,2))

# ============================================================
# HEATMAP
# ============================================================

st.header("🔥 Pollution Heatmap")

m2=folium.Map(location=[lat,lon],zoom_start=12)

HeatMap([[lat,lon,pred]]).add_to(m2)

st_folium(m2)

# ============================================================
# FUTURE FORECAST
# ============================================================

st.header("🔮 Future Forecast")

future=[pred+np.random.normal(0,5) for i in range(12)]

st.line_chart(future)

# ============================================================
# GEMINI AI ASSISTANT
# ============================================================

st.header("🤖 Google Gemini Smart City Assistant")

question=st.text_input("Ask about pollution, health, solutions")

if question:

    prompt=f"""
    AQI is {pred}
    PM2.5 is {pm25}
    PM10 is {pm10}

    Answer: {question}
    """

    response=gemini_model.generate_content(prompt)

    st.write(response.text)

# ============================================================
# SMART CITY SCORE
# ============================================================

st.header("🏙️ Smart City Score")

score=max(0,100-pred)

st.metric("City Sustainability Score",round(score,1))

# ============================================================
# SAVE HISTORY
# ============================================================

log=pd.DataFrame([[time_now,lat,lon,pred]],
columns=["time","lat","lon","aqi"])

if os.path.exists("history.csv"):

    log.to_csv("history.csv",mode="a",header=False,index=False)

else:

    log.to_csv("history.csv",index=False)

history=pd.read_csv("history.csv")

st.line_chart(history["aqi"])
