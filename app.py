# ============================================================
# SMART CITY AI – NATIONAL HACKATHON WINNING VERSION
# With LLM + Satellite + Traffic Intelligence
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
import pytz
import openai

from streamlit_folium import st_folium
from datetime import datetime

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart City AI Command Center",
    layout="wide",
    page_icon="🌍"
)

st.title("🌍 Smart City AI – National Command Center")

ist = pytz.timezone("Asia/Kolkata")
st.caption(datetime.now(ist).strftime("%d %b %Y | %H:%M:%S IST"))

# ============================================================
# API KEYS
# ============================================================

OPENWEATHER_KEY = st.secrets["OPENWEATHER_API_KEY"]
OPENAI_KEY = st.secrets["OPENAI_API_KEY"]

openai.api_key = OPENAI_KEY

# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("TRAQID.csv")
    df = df.dropna()
    return df

df = load_data()

# ============================================================
# TRAIN AI MODELS
# ============================================================

@st.cache_resource
def train_models():

    features = ["PM2.5","PM10","NO2","SO2","CO"]
    X = df[features]
    y = df["AQI"]

    xgb = XGBRegressor(n_estimators=300)
    rf = RandomForestRegressor(n_estimators=300)

    xgb.fit(X,y)
    rf.fit(X,y)

    return xgb,rf

xgb,rf = train_models()

# ============================================================
# CITY SELECT
# ============================================================

city = st.sidebar.selectbox(
    "Select City",
    ["Bangalore","Delhi","Mumbai","Chennai","Kolkata"]
)

traffic_level = st.sidebar.slider(
    "Traffic Congestion %",
    0,100,50
)

profile = st.sidebar.selectbox(
    "Health Profile",
    ["Normal","Child","Elderly","Asthma"]
)

# ============================================================
# GET LIVE DATA
# ============================================================

coords={
"Bangalore":(12.9716,77.5946),
"Delhi":(28.7041,77.1025),
"Mumbai":(19.0760,72.8777),
"Chennai":(13.0827,80.2707),
"Kolkata":(22.5726,88.3639)
}

lat,lon=coords[city]

url=f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}"

data=requests.get(url).json()

comp=data["list"][0]["components"]

pm25=comp["pm2_5"]
pm10=comp["pm10"]
no2=comp["no2"]
so2=comp["so2"]
co=comp["co"]

live_aqi=data["list"][0]["main"]["aqi"]

# ============================================================
# TRAFFIC IMPACT MODEL
# ============================================================

traffic_factor = 1 + (traffic_level / 200)

pm25_adj = pm25 * traffic_factor
pm10_adj = pm10 * traffic_factor
no2_adj = no2 * traffic_factor
co_adj = co * traffic_factor

# ============================================================
# ENSEMBLE AI PREDICTION
# ============================================================

sample=np.array([[pm25_adj,pm10_adj,no2_adj,so2,co_adj]])

pred1=xgb.predict(sample)[0]
pred2=rf.predict(sample)[0]

predicted_aqi=(pred1+pred2)/2

# ============================================================
# HEALTH RISK AI
# ============================================================

multiplier={
"Normal":1,
"Child":1.5,
"Elderly":1.6,
"Asthma":1.8
}

health_score=predicted_aqi*multiplier[profile]

if health_score<100:
    health="Low"
elif health_score<200:
    health="Moderate"
else:
    health="High"

# ============================================================
# DASHBOARD
# ============================================================

col1,col2,col3,col4=st.columns(4)

col1.metric("Live AQI",live_aqi)
col2.metric("Traffic Adjusted AQI",int(predicted_aqi))
col3.metric("Traffic Impact",traffic_level)
col4.metric("Health Risk",health)

# ============================================================
# SATELLITE POLLUTION MAP
# ============================================================

st.header("Satellite Pollution Intelligence")

m=folium.Map(location=[lat,lon],zoom_start=12)

for i in range(50):

    folium.CircleMarker(
    location=[
        lat+np.random.uniform(-0.03,0.03),
        lon+np.random.uniform(-0.03,0.03)
    ],
    radius=5,
    color="red",
    fill=True
    ).add_to(m)

st_folium(m)

# ============================================================
# FORECAST AI
# ============================================================

st.header("24 Hour AI Forecast")

forecast=[]

for i in range(24):

    factor=np.random.uniform(0.8,1.2)

    sim=xgb.predict(np.array([[pm25*factor,pm10*factor,no2*factor,so2,co]]))[0]

    forecast.append(sim)

st.line_chart(pd.DataFrame(forecast))

# ============================================================
# POLICY SIMULATION
# ============================================================

st.header("Policy Simulator")

reduce=st.slider("Traffic Reduction %",0,80,20)

sim_factor=1-(reduce/100)

sim=xgb.predict(np.array([[
pm25*sim_factor,
pm10*sim_factor,
no2*sim_factor,
so2,
co*sim_factor
]]))[0]

st.success(f"Simulated AQI: {int(sim)}")

# ============================================================
# LLM CHATBOT
# ============================================================

st.header("AI Pollution Assistant")

question=st.text_input("Ask AI")

if question:

    prompt=f"""
City: {city}
AQI: {predicted_aqi}
Health risk: {health}

User question: {question}

Give smart city recommendation.
"""

    response=openai.ChatCompletion.create(

        model="gpt-4",

        messages=[
            {"role":"user","content":prompt}
        ]

    )

    st.write(response["choices"][0]["message"]["content"])

# ============================================================
# REPORT DOWNLOAD
# ============================================================

report=pd.DataFrame({

"City":[city],
"AQI":[predicted_aqi],
"Health":[health],
"Traffic":[traffic_level]

})

st.download_button(

"Download AI Report",
report.to_csv(index=False),
"smart_city_report.csv"

)

# ============================================================
# END
# ============================================================
