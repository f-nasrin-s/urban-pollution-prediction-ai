# ============================================================
# URBAN POLLUTION AI - NEXT GEN LIVE DASHBOARD
# Unique Hackathon Winning Version
# Real-time changing data + animated charts + AI prediction
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import random

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Urban Pollution AI Live Dashboard",
    page_icon="🌍",
    layout="wide"
)

# ============================================================
# CUSTOM CSS (GLASS UI)
# ============================================================

st.markdown("""
<style>
.metric-card {
    background: rgba(255,255,255,0.1);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
}

.big-font {
    font-size: 28px !important;
    font-weight: bold;
}

.live {
    color: red;
    animation: blink 1s infinite;
}

@keyframes blink {
    50% { opacity: 0.3; }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE
# ============================================================

st.title("🌍 Urban Pollution AI Command Center")

st.markdown('<span class="live">● LIVE MONITORING</span>', unsafe_allow_html=True)

# ============================================================
# SESSION STATE INIT
# ============================================================

if "data" not in st.session_state:

    base = 120

    st.session_state.data = pd.DataFrame({
        "time": [datetime.now()],
        "AQI": [base],
        "traffic": [random.randint(30,80)],
        "industrial": [random.randint(40,90)],
        "temp": [random.randint(20,35)],
        "humidity": [random.randint(40,80)]
    })

# ============================================================
# AI MODEL TRAINING
# ============================================================

@st.cache_resource
def train_model():

    np.random.seed(42)

    df = pd.DataFrame({
        "traffic": np.random.randint(0,100,500),
        "industrial": np.random.randint(0,100,500),
        "temp": np.random.randint(15,40,500),
        "humidity": np.random.randint(30,90,500),
    })

    df["AQI"] = (
        df["traffic"] * 1.4 +
        df["industrial"] * 1.6 +
        df["temp"] * 0.5 +
        df["humidity"] * 0.4 +
        np.random.normal(0,10,500)
    )

    X = df.drop("AQI", axis=1)
    y = df["AQI"]

    model = RandomForestRegressor()
    model.fit(X,y)

    return model

model = train_model()

# ============================================================
# GENERATE LIVE DATA
# ============================================================

def generate_new_data():

    last = st.session_state.data.iloc[-1]

    new_row = {

        "time": datetime.now(),

        "traffic": max(0, min(100, last["traffic"] + random.randint(-5,5))),

        "industrial": max(0, min(100, last["industrial"] + random.randint(-3,3))),

        "temp": max(15, min(40, last["temp"] + random.randint(-2,2))),

        "humidity": max(30, min(90, last["humidity"] + random.randint(-3,3)))
    }

    X = [[
        new_row["traffic"],
        new_row["industrial"],
        new_row["temp"],
        new_row["humidity"]
    ]]

    new_row["AQI"] = model.predict(X)[0]

    st.session_state.data = pd.concat([
        st.session_state.data,
        pd.DataFrame([new_row])
    ]).tail(100)

# ============================================================
# AUTO UPDATE BUTTON
# ============================================================

col1,col2 = st.columns([1,4])

if col1.button("▶ Start Live"):

    for i in range(50):

        generate_new_data()
        time.sleep(0.5)

        st.rerun()

# ============================================================
# CURRENT METRICS
# ============================================================

latest = st.session_state.data.iloc[-1]

c1,c2,c3,c4 = st.columns(4)

c1.metric("AQI", int(latest["AQI"]), delta=random.randint(-5,5))
c2.metric("Traffic", int(latest["traffic"]))
c3.metric("Industry", int(latest["industrial"]))
c4.metric("Temperature", int(latest["temp"]))

# ============================================================
# LIVE AQI CHART
# ============================================================

st.subheader("📈 Live AQI Trend")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=st.session_state.data["time"],
    y=st.session_state.data["AQI"],
    mode='lines',
    line=dict(width=3),
    name="AQI"
))

fig.update_layout(
    template="plotly_dark",
    height=400
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# POLLUTION SOURCE PIE
# ============================================================

st.subheader("🏭 Pollution Source Contribution")

source = pd.DataFrame({

    "Source":["Traffic","Industry","Weather"],

    "Value":[
        latest["traffic"],
        latest["industrial"],
        (latest["temp"]+latest["humidity"])/2
    ]
})

fig2 = px.pie(source, names="Source", values="Value", hole=0.5)

st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# HEATMAP MAP
# ============================================================

st.subheader("🗺 Live Pollution Heatmap")

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
        popup=f"AQI: {aqi}"
    ).add_to(m)

st_folium(m, width=900, height=500)

# ============================================================
# FUTURE FORECAST
# ============================================================

st.subheader("🔮 AI Future Prediction")

future = []

base = latest["AQI"]

for i in range(24):

    base += random.randint(-5,5)
    future.append(base)

future_df = pd.DataFrame({
    "Hour": list(range(24)),
    "AQI": future
})

fig3 = px.area(future_df, x="Hour", y="AQI")

st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# INSIGHTS
# ============================================================

st.subheader("🧠 AI Insights")

if latest["AQI"] > 200:

    st.error("Severe pollution detected")

elif latest["AQI"] > 120:

    st.warning("Moderate pollution")

else:

    st.success("Air quality safe")

# ============================================================
# LIVE TABLE
# ============================================================

st.subheader("📊 Live Data Stream")

st.dataframe(st.session_state.data.tail(10), use_container_width=True)
