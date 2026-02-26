# ============================================================
# ISRO-LEVEL SMART CITY URBAN POLLUTION COMMAND CENTER
# Gemini AI Copilot + LSTM Prediction + National Heatmap
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
import pytz
import time

from datetime import datetime, timedelta
from streamlit_folium import st_folium
from folium.plugins import HeatMap, TimestampedGeoJson

from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import google.generativeai as genai

# ============================================================
# PAGE CONFIG (ISRO STYLE)
# ============================================================

st.set_page_config(
    page_title="ISRO UrbanGuard AI",
    layout="wide",
    page_icon="🛰️"
)

st.title("🛰️ ISRO UrbanGuard AI – National Pollution Intelligence System")

ist = pytz.timezone("Asia/Kolkata")
st.caption(datetime.now(ist).strftime("%d %b %Y | %H:%M:%S IST"))

# ============================================================
# LOAD API KEYS
# ============================================================

OPENWEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

genai.configure(api_key=GEMINI_API_KEY)

# ============================================================
# GEMINI COPILOT
# ============================================================

def gemini_response(prompt):

    model = genai.GenerativeModel("gemini-1.5-flash")

    response = model.generate_content(prompt)

    return response.text


# ============================================================
# TRAIN XGBOOST MODEL
# ============================================================

@st.cache_resource
def train_xgb():

    df = pd.read_csv("TRAQID.csv")

    aqi_col = [c for c in df.columns if "aqi" in c.lower()][0]

    X = df.drop(columns=[aqi_col])
    y = df[aqi_col]

    for col in X.select_dtypes(include="object"):

        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2
    )

    model = XGBRegressor()

    model.fit(X_train, y_train)

    return model, X.columns.tolist()

xgb_model, features = train_xgb()


# ============================================================
# TRAIN LSTM MODEL
# ============================================================

@st.cache_resource
def train_lstm():

    data = pd.read_csv("TRAQID.csv")

    aqi_col = [c for c in data.columns if "aqi" in c.lower()][0]

    series = data[aqi_col].values

    X = []
    y = []

    for i in range(10, len(series)):

        X.append(series[i-10:i])
        y.append(series[i])

    X = np.array(X)
    y = np.array(y)

    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()

    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile("adam", "mse")

    model.fit(X, y, epochs=3, verbose=0)

    return model, series

lstm_model, series = train_lstm()


# ============================================================
# LOCATION MAP
# ============================================================

st.subheader("📍 Select Location")

map1 = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

map_data = st_folium(map1)

if map_data and map_data.get("last_clicked"):

    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

else:

    st.stop()


# ============================================================
# GET LIVE POLLUTION DATA
# ============================================================

url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"

data = requests.get(url).json()

pm25 = data["list"][0]["components"]["pm2_5"]
pm10 = data["list"][0]["components"]["pm10"]

col1, col2 = st.columns(2)

col1.metric("PM2.5", pm25)
col2.metric("PM10", pm10)


# ============================================================
# AQI PREDICTION
# ============================================================

row = {f: 0 for f in features}

for f in features:

    if "pm2.5" in f.lower():

        row[f] = pm25

    if "pm10" in f.lower():

        row[f] = pm10

predicted_aqi = xgb_model.predict(pd.DataFrame([row]))[0]

st.metric("Predicted AQI", predicted_aqi)


# ============================================================
# FUTURE PREDICTION (LSTM)
# ============================================================

st.subheader("📈 7-Day Future AQI Forecast")

last_seq = series[-10:]

future = []

seq = last_seq.copy()

for i in range(7):

    pred = lstm_model.predict(
        seq.reshape(1,10,1),
        verbose=0
    )[0][0]

    future.append(pred)

    seq = np.append(seq[1:], pred)

future_df = pd.DataFrame({

    "Day": range(1,8),
    "Predicted AQI": future
})

st.line_chart(future_df.set_index("Day"))


# ============================================================
# NATIONAL HEATMAP
# ============================================================

st.subheader("🇮🇳 National Pollution Heatmap")

cities = [

    (28.6,77.2),
    (19.0,72.8),
    (13.0,80.2),
    (22.5,88.3),
    (17.3,78.4),
]

heat_data = []

for c in cities:

    try:

        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={c[0]}&lon={c[1]}&appid={OPENWEATHER_API_KEY}"

        d = requests.get(url).json()

        pm = d["list"][0]["components"]["pm2_5"]

        heat_data.append([c[0], c[1], pm])

    except:

        pass

map2 = folium.Map(location=[22,78], zoom_start=5)

HeatMap(heat_data).add_to(map2)

st_folium(map2)


# ============================================================
# SATELLITE ANIMATION
# ============================================================

st.subheader("🛰️ Satellite Pollution Animation")

features_anim = []

for i in range(10):

    features_anim.append({

        "type":"Feature",
        "geometry":{
            "type":"Point",
            "coordinates":[lon,lat]
        },
        "properties":{
            "time":(datetime.now()+timedelta(hours=i)).isoformat(),
            "style":{"color":"red"}
        }

    })

TimestampedGeoJson({

    "type":"FeatureCollection",
    "features":features_anim

}).add_to(map2)

st_folium(map2)


# ============================================================
# GEMINI COPILOT
# ============================================================

st.subheader("🤖 Gemini AI Copilot")

user_q = st.text_input("Ask AI about pollution")

if user_q:

    prompt = f"""

    AQI: {predicted_aqi}

    PM2.5: {pm25}

    PM10: {pm10}

    Question: {user_q}

    """

    answer = gemini_response(prompt)

    st.write(answer)
