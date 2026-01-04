import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.title("Urban Pollution Prediction 🚦")

# Load dataset
df = pd.read_csv("TRAQID.csv")
st.write("Dataset Preview:")
st.dataframe(df.head())

# ML Model Code (your existing code)
# Make sure to replace print() with st.write() or st.pyplot()
# Example template:
if "AQI" in df.columns:
    X = df.drop(columns=["aqi"])
    y = df["aqi"]

    # Encode categorical columns
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Sample Predictions")
    st.write(y_pred[:5])

    # SHAP explainability
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    st.subheader("Feature Importance (SHAP)")
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(bbox_inches='tight')
else:
    st.warning("AQI column not found in dataset!")
