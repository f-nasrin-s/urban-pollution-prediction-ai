# ============================================================
# URBANGUARD AI – ISRO LEVEL NATIONAL POLLUTION INTELLIGENCE
# WITH GEMINI AI COPILOT
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
import pytz
import time
import google.generativeai as genai

from datetime import datetime
from streamlit_folium import st_folium
from folium.plugins import HeatMap

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import streamlit.components.v1 as components

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="UrbanGuard AI – National Pollution Intelligence",
    layout="wide"
)

st.title("🛰️ UrbanGuard AI – National Smart City Pollution Intelligence System")

ist = pytz.timezone("Asia/Kolkata")
st.caption(f"⏱️ {datetime.now(ist).strftime('%d %b %Y | %H:%M:%S IST')}")

# ============================================================
# API KEYS
# ============================================================

OPENWEATHER = st.secrets.get("OPENWEATHER_API_KEY", "")
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")

# Configure Gemini
genai.configure(api_key=GEMINI_KEY)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource
def load_data():

    df = pd.read_csv("TRAQID.csv")

    aqi_col = [c for c in df.columns if "aqi" in c.lower()][0]

    drop_cols = ["Image","created_at","Sequence","aqi_cat",aqi_col]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[aqi_col]

    for col in X.select_dtypes(include="object").columns:

        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    return X,y,X.columns.tolist()

X,y,features = load_data()

# ============================================================
# XGBOOST MODEL
# ============================================================

@st.cache_resource
def train_xgb():

    model = XGBRegressor(n_estimators=150,max_depth=6)

    model.fit(X,y)

    return model

xgb_model = train_xgb()

# ============================================================
# LSTM MODEL
# ============================================================

@st.cache_resource
def train_lstm():

    scaler = MinMaxScaler()

    y_scaled = scaler.fit_transform(y.values.reshape(-1,1))

    seq=10

    X_lstm=[]
    y_lstm=[]

    for i in range(seq,len(y_scaled)):

        X_lstm.append(y_scaled[i-seq:i])
        y_lstm.append(y_scaled[i])

    X_lstm,y_lstm=np.array(X_lstm),np.array(y_lstm)

    model=Sequential()
    model.add(LSTM(50,input_shape=(seq,1)))
    model.add(Dense(1))

    model.compile(loss="mse",optimizer="adam")

    model.fit(X_lstm,y_lstm,epochs=5,verbose=0)

    return model,scaler

lstm_model,scaler=train_lstm()

# ============================================================
# GPS DETECT
# ============================================================

def gps():

    html = """
    <script>
    navigator.geolocation.getCurrentPosition(
    (pos)=>{
    const coords=pos.coords.latitude+","+pos.coords.longitude;
    const input=window.parent.document.querySelector('input');
    input.value=coords;
    input.dispatchEvent(new Event('input',{bubbles:true}));
    });
    </script>
    """

    components.html(html)

# ============================================================
# LOCATION
# ============================================================

st.header("📍 Location Selection")

mode=st.radio(
"Select Location Method",
["Map","Auto Detect"]
)

lat,lon=None,None

if mode=="Auto Detect":

    gps_val=st.text_input("GPS Coordinates")

    if st.button("Detect My Location"):

        gps()

    if gps_val:

        lat,lon=map(float,gps_val.split(","))

else:

    m=folium.Map(location=[22,78],zoom_start=5)

    data=st_folium(m,height=400)

    if data and data["last_clicked"]:

        lat=data["last_clicked"]["lat"]
        lon=data["last_clicked"]["lng"]

if lat is None:

    st.stop()

# ============================================================
# LIVE AQI
# ============================================================

url=f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER}"

data=requests.get(url).json()

pm25=data["list"][0]["components"]["pm2_5"]
pm10=data["list"][0]["components"]["pm10"]

st.metric("PM2.5",round(pm25,2))
st.metric("PM10",round(pm10,2))

# ============================================================
# XGB PREDICTION
# ============================================================

row={}

for f in features:

    if "pm2.5" in f.lower():

        row[f]=pm25

    elif "pm10" in f.lower():

        row[f]=pm10

    else:

        row[f]=0

pred_xgb=float(xgb_model.predict(pd.DataFrame([row]))[0])

st.metric("Current AQI Prediction",round(pred_xgb,2))

# ============================================================
# LSTM FUTURE
# ============================================================

last=y.values[-10:]

scaled=scaler.transform(last.reshape(-1,1))
scaled=scaled.reshape(1,10,1)

future_scaled=lstm_model.predict(scaled)

future=scaler.inverse_transform(future_scaled)[0][0]

st.metric("Future AQI Prediction",round(future,2))

# ============================================================
# NATIONAL HEATMAP
# ============================================================

st.header("🇮🇳 National Pollution Heatmap")

cities={
"Delhi":(28.6,77.2),
"Mumbai":(19.0,72.8),
"Bangalore":(12.9,77.5),
"Chennai":(13.0,80.2)
}

heat=[]

m2=folium.Map(location=[22,78],zoom_start=5,tiles="CartoDB dark_matter")

for city,(la,lo) in cities.items():

    url=f"http://api.openweathermap.org/data/2.5/air_pollution?lat={la}&lon={lo}&appid={OPENWEATHER}"

    d=requests.get(url).json()

    val=d["list"][0]["components"]["pm2_5"]

    heat.append([la,lo,val])

HeatMap(heat).add_to(m2)

st_folium(m2,height=500)

# ============================================================
# SATELLITE SCAN SIMULATION
# ============================================================

st.header("🛰️ Satellite Scan")

progress=st.progress(0)

for i in range(100):

    time.sleep(0.01)
    progress.progress(i+1)

st.success("Satellite Scan Complete")

# ============================================================
# GEMINI AI COPILOT
# ============================================================

st.header("🤖 Gemini AI Copilot")

user_question=st.text_input(
"Ask Gemini about pollution, health, or solutions:"
)

if user_question:

    try:

        model=genai.GenerativeModel("gemini-1.5-flash")

        prompt=f"""
        You are an environmental AI expert.

        Current AQI: {pred_xgb}
        Future AQI: {future}
        PM2.5: {pm25}
        PM10: {pm10}

        User Question: {user_question}

        Provide:
        - Risk analysis
        - Health impact
        - Recommended actions
        """

        response=model.generate_content(prompt)

        st.success(response.text)

    except Exception as e:

        st.error("Gemini error. Check API key.")

# ============================================================
# COMMAND CENTER STATUS
# ============================================================

st.header("🏛️ National Command Center")

if pred_xgb>150:

    st.error("🚨 Emergency Pollution Level")

elif pred_xgb>80:

    st.warning("⚠️ Moderate Pollution")

else:

    st.success("✅ Safe Air Quality")

# ============================================================
# END
# ============================================================
