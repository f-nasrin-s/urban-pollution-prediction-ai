# 🏙️ Smart City Pollution AI

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25-orange?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red)

**Real-time Urban Pollution Monitoring & Prediction Platform**  
Monitor AQI, analyze pollution sources, simulate interventions, and get AI recommendations for health and policy.

---

## 🌟 Features

- 📌 **Auto-detect location** or select on interactive map  
- 🔮 **AQI Prediction** using XGBoost ML model  
- 🧪 **Factor Attribution** – Dominant pollutants & sources  
- ❤️ **Health Risk Analysis** – Children, Elderly, Asthma  
- 🧠 **What-If Simulation** – Traffic & Construction reduction impact  
- 🗺️ **Hotspot Map** – Interactive heatmap visualization  
- 🤖 **AI Recommendations** – Policy & citizen guidance  
- 💬 **Interactive Chatbot** – Ask about AQI, PM2.5, PM10, and health

---

## 📊 Dataset

`TRAQID.csv` contains historical pollution data:

| Column | Description |
|--------|-------------|
| created_at | Timestamp of record |
| Temperature | Local temperature (°C) |
| Humidity | Humidity (%) |
| PM2.5 | PM2.5 concentration (µg/m³) |
| PM10 | PM10 concentration (µg/m³) |
| aqi | Air Quality Index |
| aqi_cat | AQI Category (Good/Moderate/Unhealthy) |
| ... | Other features (Season, Day/Night, Sequence, Image) |

---

## 💻 Installation

1. Clone the repo:

```bash
git clone https://github.com/f-nasrin-s/smart-city-pollution-ai.git
cd smart-city-pollution-ai
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Set OpenWeather API Key:

toml
Copy code
# .streamlit/secrets.toml
OPENWEATHER_API_KEY = "83350e70e4de15a991533bdd03e028ab"
🚀 Run the App
bash
Copy code
streamlit run app.py
Choose Auto Detect or Map Selection

View live AQI, risk analysis, hotspot maps, and run what-if simulations

⚙️ Tech Stack
Python: streamlit, pandas, numpy, xgboost, scikit-learn, requests, folium, pytz, joblib

API: OpenWeather Air Pollution API

Visualization: Folium maps with HeatMap overlays

🤖 AI Recommendations
AQI	Advisory
>180	Severe: Emergency measures, traffic control, stop construction
121-180	Moderate-High: Remote work, hotspot monitoring
≤120	Low: Normal activities, promote green mobility

🎯 Future Enhancements
Real-time multi-city pollution feeds

Mobile-friendly citizen alerts

Advanced NLP chatbot for natural language queries

Meteorological impact integration for better AQI predictions

