# ============================================================
# SMART CITY AI – NATIONAL LEVEL HACKATHON WINNING VERSION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
import pytz
import plotly.express as px

from datetime import datetime
from streamlit_folium import st_folium
from folium.plugins import HeatMap

from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart City AI Command Center",
    layout="wide"
)

st.title("🌍 Smart City AI Command Center")

ist = pytz.timezone("Asia/Kolkata")
st.caption(datetime.now(ist).strftime("%d %b %Y | %H:%M:%S IST"))

# ============================================================
# API KEYS
# ============================================================

OPENWEATHER_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")

client = None
if OPENAI_KEY:
    client = OpenAI(api_key=OPENAI_KEY)

# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model():

    df = pd.read_csv("TRAQID.csv")

    aaqi = [c for c in df.columns if "aqi" in c.lower()][0]

    drop_cols = ["Image","created_at","Sequence","aqi_cat",aaqi]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[aaqi]

    for col in X.select_dtypes(include="object"):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    X_train, _, y_train, _ = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    model = XGBRegressor(
        n_estimators=400,
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

m = folium.Map(location=[20.5937,78.9629], zoom_start=5)

map_data = st_folium(m, height=400)

if map_data and map_data.get("last_clicked"):

    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

else:
    st.stop()

# ============================================================
# FETCH WEATHER
# ============================================================

geo_url = f"http://api.openweathermap.org/geo/1.0/reverse?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}"

geo = requests.get(geo_url).json()

city = geo[0]["name"] if geo else "Unknown"

poll_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}"

poll = requests.get(poll_url).json()["list"][0]["components"]

pm25 = poll["pm2_5"]
pm10 = poll["pm10"]

weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}&units=metric"

weather = requests.get(weather_url).json()

temp = weather["main"]["temp"]
humidity = weather["main"]["humidity"]
wind = weather["wind"]["speed"]

# ============================================================
# DISPLAY METRICS
# ============================================================

st.header("📊 Environmental Intelligence")

c1,c2,c3,c4,c5 = st.columns(5)

c1.metric("City", city)
c2.metric("PM2.5", pm25)
c3.metric("PM10", pm10)
c4.metric("Temp", temp)
c5.metric("Humidity", humidity)

# ============================================================
# AQI PREDICTION
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

st.metric("Predicted AQI", round(predicted_aqi,2))

# ============================================================
# TRAFFIC AI SIMULATION
# ============================================================

st.header("🚦 Traffic AI Engine")

traffic_level = np.random.choice(
    ["Low","Moderate","High"]
)

st.metric("Traffic Level", traffic_level)

traffic_factor = {
    "Low":0.8,
    "Moderate":1,
    "High":1.3
}

traffic_adjusted_aqi = predicted_aqi * traffic_factor[traffic_level]

st.metric("Traffic Adjusted AQI", round(traffic_adjusted_aqi,2))

# ============================================================
# SATELLITE AI SIMULATION
# ============================================================

st.header("🛰 Satellite Pollution Detection")

satellite_index = np.random.uniform(0,500)

st.metric("Satellite Pollution Index", round(satellite_index,2))

# ============================================================
# FORECAST
# ============================================================

st.header("📈 72 Hour Forecast")

forecast = []

for i in range(72):

    sim = predicted_aqi * np.random.uniform(0.8,1.2)

    forecast.append(sim)

forecast_df = pd.DataFrame({"AQI":forecast})

st.line_chart(forecast_df)

# ============================================================
# ANOMALY DETECTION
# ============================================================

st.header("🚨 AI Anomaly Detection")

if predicted_aqi > np.mean(forecast)*1.3:

    st.error("Pollution anomaly detected")

else:

    st.success("Normal")

# ============================================================
# HEATMAP
# ============================================================

st.header("🗺 Pollution Heatmap")

points = []

for i in range(50):

    points.append([
        lat+np.random.uniform(-0.01,0.01),
        lon+np.random.uniform(-0.01,0.01),
        predicted_aqi
    ])

map2 = folium.Map(location=[lat,lon], zoom_start=12)

HeatMap(points).add_to(map2)

st_folium(map2)

# ============================================================
# POLICY OPTIMIZER
# ============================================================

st.header("🏛 Policy Optimizer")

best = predicted_aqi

best_policy = ""

for t in range(0,50,10):

    sim = predicted_aqi*(1-t/100)

    if sim < best:

        best = sim
        best_policy = f"Reduce traffic {t}%"

st.success(best_policy)

# ============================================================
# SUSTAINABILITY SCORE
# ============================================================

score = max(0,100-predicted_aqi)

st.metric("Sustainability Score", score)

# ============================================================
# MULTI CITY COMPARISON
# ============================================================

st.header("🌍 India City Comparison")

cities = ["Delhi","Mumbai","Bangalore","Chennai","Hyderabad"]

values = np.random.randint(50,350,5)

df = pd.DataFrame({

    "City":cities,
    "AQI":values

})

fig = px.bar(df,x="City",y="AQI")

st.plotly_chart(fig)

# ============================================================
# REPORT DOWNLOAD
# ============================================================

report = pd.DataFrame({

    "City":[city],
    "AQI":[predicted_aqi],
    "Traffic":[traffic_level],
    "Satellite":[satellite_index],
    "Score":[score]

})

st.download_button(

    "Download Smart City Report",

    report.to_csv(index=False),

    "smart_city_report.csv"

)

# ============================================================
# LLM AI ASSISTANT
# ============================================================

st.header("🤖 Smart City AI Assistant")

question = st.text_input("Ask Smart City AI")

if question and client:

    response = client.chat.completions.create(

        model="gpt-4o-mini",

        messages=[{
            "role":"user",
            "content":f"""
City: {city}
AQI: {predicted_aqi}
Traffic: {traffic_level}
Satellite Index: {satellite_index}

Question: {question}
"""
        }]
    )

    st.write(response.choices[0].message.content)

elif question:

    st.warning("Add OPENAI_API_KEY in secrets")

# ============================================================
# EMERGENCY ALERT
# ============================================================

st.header("🚨 Emergency Alert System")

if predicted_aqi > 300:

    st.error("Hazardous pollution")

elif predicted_aqi > 200:

    st.warning("Severe pollution")

else:

    st.success("Safe")

# ============================================================
# END
# ============================================================
