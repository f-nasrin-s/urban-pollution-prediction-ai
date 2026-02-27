# ============================================================
# ULTRA PRO SMART CITY AI COMMAND CENTER
# National Hackathon Winning Version
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import folium
import random
import time

from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from streamlit_folium import st_folium

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ULTRA PRO Smart City AI",
    layout="wide",
    page_icon="🌍"
)

st.title("🌍 ULTRA PRO Smart City AI Command Center")

# ============================================================
# TRAIN AI MODEL
# ============================================================

@st.cache_resource
def train():

    df = pd.DataFrame({

        "traffic": np.random.randint(0,100,500),

        "industrial": np.random.randint(0,100,500),

        "temp": np.random.randint(15,40,500),

        "humidity": np.random.randint(30,90,500)
    })

    df["AQI"] = (
        df["traffic"]*1.5 +
        df["industrial"]*1.8 +
        df["temp"]*0.5 +
        df["humidity"]*0.3 +
        np.random.normal(0,10,500)
    )

    X = df.drop("AQI", axis=1)

    y = df["AQI"]

    model = RandomForestRegressor()

    model.fit(X,y)

    return model

model = train()

# ============================================================
# SESSION STATE
# ============================================================

if "data" not in st.session_state:

    st.session_state.data = pd.DataFrame()

# ============================================================
# LIVE DATA GENERATOR
# ============================================================

def get_live_data():

    traffic = random.randint(20,100)

    industrial = random.randint(30,100)

    temp = random.randint(18,35)

    humidity = random.randint(40,85)

    X = [[traffic, industrial, temp, humidity]]

    aqi = model.predict(X)[0]

    return {

        "time": datetime.now(),

        "traffic": traffic,

        "industrial": industrial,

        "temp": temp,

        "humidity": humidity,

        "AQI": aqi
    }

# ============================================================
# REFRESH BUTTON
# ============================================================

if st.button("🔄 Refresh Live Data"):

    new = get_live_data()

    st.session_state.data = pd.concat([
        st.session_state.data,
        pd.DataFrame([new])
    ]).tail(100)

# initialize first row
if len(st.session_state.data)==0:

    st.session_state.data = pd.DataFrame([get_live_data()])

latest = st.session_state.data.iloc[-1]

# ============================================================
# METRICS
# ============================================================

c1,c2,c3,c4,c5 = st.columns(5)

c1.metric("AQI", int(latest["AQI"]))
c2.metric("Traffic", latest["traffic"])
c3.metric("Industry", latest["industrial"])
c4.metric("Temperature", latest["temp"])
c5.metric("Humidity", latest["humidity"])

# ============================================================
# AQI CHART
# ============================================================

st.subheader("📈 Live AQI Trend")

fig = px.line(
    st.session_state.data,
    x="time",
    y="AQI",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# DIGITAL TWIN SIMULATION
# ============================================================

st.subheader("🌆 Digital Twin Simulation")

traffic_reduction = st.slider("Reduce Traffic %",0,100,0)

industrial_reduction = st.slider("Reduce Industry %",0,100,0)

sim_traffic = latest["traffic"]*(1-traffic_reduction/100)

sim_industry = latest["industrial"]*(1-industrial_reduction/100)

sim = model.predict([[

    sim_traffic,
    sim_industry,
    latest["temp"],
    latest["humidity"]

]])[0]

st.metric("Simulated AQI", int(sim))

# ============================================================
# HEATMAP
# ============================================================

st.subheader("🗺 Pollution Map")

m = folium.Map(location=[12.97,77.59], zoom_start=11)

for i in range(30):

    lat = 12.97 + random.uniform(-0.05,0.05)

    lon = 77.59 + random.uniform(-0.05,0.05)

    aqi = random.randint(50,300)

    color = "green"

    if aqi > 200:
        color="red"

    elif aqi > 120:
        color="orange"

    folium.CircleMarker(

        location=[lat,lon],

        radius=8,

        color=color,

        fill=True,

        popup=str(aqi)

    ).add_to(m)

st_folium(m)

# ============================================================
# FORECAST
# ============================================================

st.subheader("🔮 AI Forecast")

future = []

base = latest["AQI"]

for i in range(24):

    base += random.randint(-5,5)

    future.append(base)

forecast = pd.DataFrame({

    "hour": range(24),

    "AQI": future
})

fig2 = px.area(forecast, x="hour", y="AQI")

st.plotly_chart(fig2)

# ============================================================
# RISK LEVEL
# ============================================================

st.subheader("🚨 Risk Level")

if latest["AQI"]>300:

    st.error("Hazardous")

elif latest["AQI"]>200:

    st.error("Very Unhealthy")

elif latest["AQI"]>150:

    st.warning("Unhealthy")

else:

    st.success("Safe")

# ============================================================
# AI COPILOT
# ============================================================

st.subheader("🤖 AI Copilot")

q = st.text_input("Ask AI")

if q:

    if "aqi" in q.lower():

        st.write(f"Current AQI is {int(latest['AQI'])}")

    elif "reduce" in q.lower():

        st.write("Reduce traffic and industrial emissions")

    else:

        st.write("Monitoring pollution continuously")

# ============================================================
# LIVE DATA TABLE
# ============================================================

st.subheader("📊 Live Data")

st.dataframe(st.session_state.data.tail(10))
