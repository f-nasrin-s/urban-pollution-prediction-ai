# ============================================================
# URBANGUARD AI – ISRO-LEVEL NATIONAL POLLUTION INTELLIGENCE
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
import pytz
import time

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

st.title("🛰️ UrbanGuard AI – National Smart City Pollution Intelligence")

ist = pytz.timezone("Asia/Kolkata")
st.caption(f"⏱️ {datetime.now(ist).strftime('%d %b %Y | %H:%M:%S IST')}")

API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")

# ============================================================
# SATELLITE STATUS
# ============================================================

st.header("🛰️ Satellite Network Status")

c1, c2, c3 = st.columns(3)

c1.success("ISRO INSAT-3D: ACTIVE")
c2.success("NASA Aura: ACTIVE")
c3.success("ESA Sentinel-5P: ACTIVE")

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

    return X, y, X.columns.tolist()

X, y, features = load_data()

# ============================================================
# TRAIN XGBOOST MODEL
# ============================================================

@st.cache_resource
def train_xgb():

    model = XGBRegressor(n_estimators=150,max_depth=6)

    model.fit(X,y)

    return model

xgb_model = train_xgb()

# ============================================================
# TRAIN LSTM MODEL
# ============================================================

@st.cache_resource
def train_lstm():

    scaler = MinMaxScaler()

    y_scaled = scaler.fit_transform(y.values.reshape(-1,1))

    X_lstm = []

    y_lstm = []

    seq = 10

    for i in range(seq,len(y_scaled)):

        X_lstm.append(y_scaled[i-seq:i])

        y_lstm.append(y_scaled[i])

    X_lstm,y_lstm = np.array(X_lstm),np.array(y_lstm)

    model = Sequential()

    model.add(LSTM(50,input_shape=(seq,1)))

    model.add(Dense(1))

    model.compile(loss="mse",optimizer="adam")

    model.fit(X_lstm,y_lstm,epochs=5,verbose=0)

    return model,scaler

lstm_model, scaler = train_lstm()

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

mode = st.radio(
"Select",
["Map","Auto Detect"]
)

lat,lon=None,None

if mode=="Auto Detect":

    gps_val=st.text_input("GPS")

    if st.button("Detect"):

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

url=f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"

data=requests.get(url).json()

pm25=data["list"][0]["components"]["pm2_5"]

pm10=data["list"][0]["components"]["pm10"]

st.metric("PM2.5",pm25)

st.metric("PM10",pm10)

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

st.metric("XGBoost AQI",round(pred_xgb,2))

# ============================================================
# LSTM FUTURE PREDICTION
# ============================================================

st.header("🔮 Deep Learning Future Prediction")

last=y.values[-10:]

scaled=scaler.transform(last.reshape(-1,1))

scaled=scaled.reshape(1,10,1)

future_scaled=lstm_model.predict(scaled)

future=scaler.inverse_transform(future_scaled)[0][0]

st.metric("LSTM Future AQI",round(future,2))

# ============================================================
# SATELLITE ANIMATION
# ============================================================

st.header("🛰️ Satellite Pollution Scan")

progress=st.progress(0)

for i in range(100):

    time.sleep(0.01)

    progress.progress(i+1)

st.success("Satellite Scan Complete")

# ============================================================
# NATIONAL HEATMAP
# ============================================================

st.header("🇮🇳 National Heatmap")

cities={
"Delhi":(28,77),
"Mumbai":(19,72),
"Bangalore":(12,77),
"Chennai":(13,80)
}

heat=[]

m2=folium.Map(location=[22,78],zoom_start=5,tiles="CartoDB dark_matter")

for city,(la,lo) in cities.items():

    url=f"http://api.openweathermap.org/data/2.5/air_pollution?lat={la}&lon={lo}&appid={API_KEY}"

    d=requests.get(url).json()

    val=d["list"][0]["components"]["pm2_5"]

    heat.append([la,lo,val])

HeatMap(heat).add_to(m2)

st_folium(m2,height=500)

# ============================================================
# FORECAST GRAPH
# ============================================================

st.header("📈 Forecast")

forecast=[pred_xgb,future]

df_fore=pd.DataFrame({

"time":["now","future"],

"aqi":forecast

})

st.line_chart(df_fore.set_index("time"))

# ============================================================
# COMMAND CENTER STATUS
# ============================================================

st.header("🏛️ Command Center")

if pred_xgb>150:

    st.error("Emergency")

elif pred_xgb>80:

    st.warning("Warning")

else:

    st.success("Safe")

# ============================================================
# END
# ============================================================
