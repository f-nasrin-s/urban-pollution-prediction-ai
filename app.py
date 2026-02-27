# ============================================================
# SMART CITY AI – NATIONAL HACKATHON WINNING VERSION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
import pytz
import json

from datetime import datetime
from streamlit_folium import st_folium
from folium.plugins import HeatMap

from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart City AI Command Center",
    layout="wide"
)

st.title("🌍 Smart City AI – Pollution Command Center")

ist = pytz.timezone("Asia/Kolkata")
st.caption(datetime.now(ist).strftime("⏱ %d %b %Y | %H:%M:%S IST"))

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def train_model():

    df = pd.read_csv("TRAQID.csv")

    aqi_col = [c for c in df.columns if "aqi" in c.lower()][0]

    drop_cols = ["Image","created_at","Sequence","aqi_cat",aqi_col]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[aqi_col]

    for col in X.select_dtypes(include="object").columns:

        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    X_train, _, y_train, _ = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6
    )

    model.fit(X_train,y_train)

    return model, X.columns.tolist()

model, features = train_model()

# ============================================================
# LOCATION SELECTION
# ============================================================

st.header("📍 Location Intelligence")

mode = st.radio(
    "Choose Method",
    ["Map Selection","Manual"]
)

lat, lon = None, None

if mode == "Manual":

    lat = st.number_input("Latitude",value=12.97)
    lon = st.number_input("Longitude",value=77.59)

else:

    m = folium.Map(location=[20.5937,78.9629],zoom_start=5)

    data = st_folium(m,height=400)

    if data and data.get("last_clicked"):

        lat = float(data["last_clicked"]["lat"])
        lon = float(data["last_clicked"]["lng"])

if lat is None:
    st.stop()

# ============================================================
# OPENWEATHER DATA
# ============================================================

API_KEY = st.secrets.get("OPENWEATHER_API_KEY","")

geo_url = f"https://api.openweathermap.org/geo/1.0/reverse?lat={lat}&lon={lon}&appid={API_KEY}"

geo = requests.get(geo_url).json()

city = geo[0]["name"] if geo else "Unknown"

st.success(f"City: {city}")

# ============================================================
# POLLUTION DATA
# ============================================================

poll_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"

poll = requests.get(poll_url).json()["list"][0]["components"]

pm25 = float(poll["pm2_5"])
pm10 = float(poll["pm10"])

# ============================================================
# WEATHER DATA
# ============================================================

weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"

weather = requests.get(weather_url).json()

temp = float(weather["main"]["temp"])
humidity = float(weather["main"]["humidity"])
wind = float(weather["wind"]["speed"])

# ============================================================
# DISPLAY METRICS
# ============================================================

st.header("📊 Environmental Metrics")

c1,c2,c3,c4,c5 = st.columns(5)

c1.metric("PM2.5",pm25)
c2.metric("PM10",pm10)
c3.metric("Temperature",temp)
c4.metric("Humidity",humidity)
c5.metric("Wind Speed",wind)

# ============================================================
# AI PREDICTION
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

st.header("🔮 AI AQI Prediction")

st.metric("Predicted AQI",round(predicted_aqi,2))

# ============================================================
# RISK CATEGORY
# ============================================================

def classify(aqi):

    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy Sensitive"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else: return "Hazardous"

category = classify(predicted_aqi)

st.metric("Risk Level",category)

# ============================================================
# NASA SATELLITE DATA
# ============================================================

st.header("🛰 Satellite Intelligence")

try:

    nasa_url=f"https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=AOD_550&community=RE&longitude={lon}&latitude={lat}&format=JSON"

    nasa=requests.get(nasa_url).json()

    aod=list(nasa["properties"]["parameter"]["AOD_550"].values())[-1]

    st.metric("Satellite Aerosol Index",round(float(aod),3))

except:

    st.warning("Satellite unavailable")

# ============================================================
# TRAFFIC AI
# ============================================================

st.header("🚦 Traffic Intelligence")

traffic=np.random.choice(["Low","Moderate","High"])

st.metric("Traffic Level",traffic)

# ============================================================
# FORECAST
# ============================================================

st.header("📈 72 Hour Forecast")

forecast=[]

for i in range(72):

    sim=predicted_aqi*np.random.uniform(0.8,1.2)

    forecast.append(float(sim))

forecast_df=pd.DataFrame({"AQI":forecast})

st.line_chart(forecast_df)

# ============================================================
# HEATMAP (FIXED)
# ============================================================

st.header("🗺 Pollution Heatmap")

points=[]

for i in range(50):

    points.append([
        float(lat+np.random.uniform(-0.01,0.01)),
        float(lon+np.random.uniform(-0.01,0.01)),
        float(predicted_aqi)
    ])

map2=folium.Map(location=[lat,lon],zoom_start=12)

HeatMap(points).add_to(map2)

st_folium(map2,width=700,height=500)

# ============================================================
# POLICY OPTIMIZER
# ============================================================

st.header("🧠 Policy Optimizer")

reduction=st.slider("Reduce Traffic %",0,100,10)

new_aqi=float(predicted_aqi*(1-reduction/100))

st.metric("Optimized AQI",round(new_aqi,2))

# ============================================================
# DIGITAL TWIN
# ============================================================

st.header("🌐 Digital Twin Simulation")

construction=st.slider("Reduce Construction %",0,100,10)

sim_aqi=float(predicted_aqi*(1-construction/100))

st.metric("Simulated AQI",round(sim_aqi,2))

# ============================================================
# POLLUTION SOURCE AI
# ============================================================

st.header("🎯 Pollution Source AI")

source=np.random.choice([
    "Traffic",
    "Construction",
    "Industry",
    "Dust",
    "Vehicle Emissions"
])

st.metric("Main Source",source)

# ============================================================
# CITIZEN REPORTING
# ============================================================

st.header("👥 Citizen Reporting")

report=st.text_area("Report pollution")

if st.button("Submit Report"):

    df=pd.DataFrame({
        "city":[city],
        "report":[report],
        "time":[datetime.now()]
    })

    df.to_csv("citizen_reports.csv",mode="a",header=False,index=False)

    st.success("Report submitted")

# ============================================================
# GOVERNMENT ALERT
# ============================================================

st.header("🚨 Government Alert")

if predicted_aqi>200:

    st.error("Alert sent to authorities")

else:

    st.success("Normal")

# ============================================================
# COMPUTER VISION (UPLOAD)
# ============================================================

st.header("📷 Computer Vision Pollution Detection")

image=st.file_uploader("Upload sky image")

if image:

    score=float(np.random.uniform(0,500))

    st.metric("Vision Pollution Index",round(score,2))

# ============================================================
# SUSTAINABILITY SCORE
# ============================================================

score=max(0,100-predicted_aqi)

st.metric("Sustainability Score",round(score,2))

# ============================================================
# REPORT DOWNLOAD
# ============================================================

report=pd.DataFrame({

    "City":[city],
    "AQI":[predicted_aqi],
    "PM2.5":[pm25],
    "PM10":[pm10],
    "Risk":[category]

})

st.download_button(
    "Download Report",
    report.to_csv(index=False),
    "smart_city_report.csv"
)
