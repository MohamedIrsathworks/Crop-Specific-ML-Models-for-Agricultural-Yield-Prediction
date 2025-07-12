import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from data_preprocessing import PreprocessingPipeline
from model_inference import CropYieldPredictor

# Load cleaned data
df = pd.read_csv("crop_yield_cleaned.csv")

# Initialize
pipeline = PreprocessingPipeline()
pipeline.fit(df)
predictor = CropYieldPredictor()

# UI settings
st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="centered")
st.title("🌾 Crop Yield Prediction System")

option = st.sidebar.selectbox("Choose Option", ("Prediction", "Model Comparison"))

if option == "Prediction":
    st.header("🔮 Predict Yield")

    crop = st.selectbox("Crop", sorted(df["Crop"].unique()))
    fertilizer = st.number_input("Fertilizer Usage (kg)", min_value=0.0)
    pesticide = st.number_input("Pesticide Usage (kg)", min_value=0.0)
    area = st.number_input("Area (hectares)", min_value=0.1)
    annual_rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0)
    state = st.selectbox("State", sorted(df["State"].unique()))
    season = st.selectbox("Season", sorted(df["Season"].unique()))

    if st.button("Predict Yield"):
        # Check if all inputs are zero
        if fertilizer == 0.0 and pesticide == 0.0 and annual_rainfall == 0.0:
            st.warning("⚠️ Invalid Inputs. Cannot predict yield.")
        else:
            input_df = pd.DataFrame([{
                "Crop": crop,
                "Fertilizer": fertilizer,
                "Pesticide": pesticide,
                "Area": area,
                "Annual_Rainfall": annual_rainfall,
                "State": state,
                "Season": season
            }])

            # Predict directly using raw input_df
            predicted_yield, feature_names = predictor.predict(crop, input_df)

            # Calculate total production
            predicted_production = predicted_yield * area

            # Show results
            st.success(f"✅ Predicted Yield: {predicted_yield:.2f} tons/hectare")
            st.info(f"📦 Estimated Total Production: {predicted_production:.2f} tons")

elif option == "Model Comparison":
    st.header("📊 Compare Models for a Crop")

    model_df = pd.read_csv("model_performance_with_kfold.csv")
    crop_list = sorted(model_df["Crop"].unique())
    selected_crop = st.selectbox("Select Crop for Comparison", crop_list)

    crop_models = model_df[model_df["Crop"] == selected_crop]

    if not crop_models.empty:
        st.subheader("R² Scores")
        fig_r2 = px.bar(crop_models, x="Model", y="R²", color="Model", text_auto=True)
        st.plotly_chart(fig_r2)

        st.subheader("MSE Scores")
        fig_mse = px.bar(crop_models, x="Model", y="MSE", color="Model", text_auto=True)
        st.plotly_chart(fig_mse)

        st.subheader("MAE Scores")
        fig_mae = px.bar(crop_models, x="Model", y="MAE", color="Model", text_auto=True)
        st.plotly_chart(fig_mae)

        st.subheader("RMSE Scores")
        fig_rmse = px.bar(crop_models, x="Model", y="RMSE", color="Model", text_auto=True)
        st.plotly_chart(fig_rmse)
    else:
        st.warning("⚠️ No data available for this crop.")
