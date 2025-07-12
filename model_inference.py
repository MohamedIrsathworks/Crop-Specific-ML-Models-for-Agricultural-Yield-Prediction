import joblib
import json
import pandas as pd

class CropYieldPredictor:
    def __init__(self, model_folder="models"):
        self.model_folder = model_folder
        self.loaded_models = {}
        self.loaded_scalers = {}
        self.feature_map = {}

    def load_model(self, crop_name):
        if crop_name in self.loaded_models:
            return self.loaded_models[crop_name]

        crop_safe = crop_name.replace("/", "_")
        model_path = f"{self.model_folder}/{crop_safe}.pkl"
        model = joblib.load(model_path)
        self.loaded_models[crop_name] = model
        return model

    def load_scaler(self, crop_name):
        if crop_name in self.loaded_scalers:
            return self.loaded_scalers[crop_name]

        crop_safe = crop_name.replace("/", "_")
        scaler_path = f"{self.model_folder}/{crop_safe}_scaler.pkl"
        scaler = joblib.load(scaler_path)
        self.loaded_scalers[crop_name] = scaler
        return scaler

    def get_feature_order(self, crop_name):
        if crop_name in self.feature_map:
            return self.feature_map[crop_name]

        crop_safe = crop_name.replace("/", "_")
        with open(f"{self.model_folder}/{crop_safe}_features.json", "r") as f:
            features = json.load(f)
        self.feature_map[crop_name] = features
        return features

    def predict(self, crop_name, features_df):
        model = self.load_model(crop_name)
        scaler = self.load_scaler(crop_name)
        expected_columns = self.get_feature_order(crop_name)

        # Feature engineering
        features_df = features_df.copy()
        features_df["Fertilizer_Usage_per_Area"] = features_df["Fertilizer"] / (features_df["Area"] + 1e-6)
        features_df["Pesticide_Usage_per_Area"] = features_df["Pesticide"] / (features_df["Area"] + 1e-6)

        # One-hot encoding
        features_df = pd.get_dummies(features_df, columns=["State", "Season"], drop_first=True)

        # Add missing columns and reorder
        for col in expected_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        features_df = features_df[expected_columns]

        # Scale per crop (using the stored scaler)
        features_df[["Fertilizer_Usage_per_Area", "Pesticide_Usage_per_Area", "Annual_Rainfall"]] = scaler.transform(
            features_df[["Fertilizer_Usage_per_Area", "Pesticide_Usage_per_Area", "Annual_Rainfall"]]
        )

        # Predict
        prediction = model.predict(features_df)[0]

        return prediction, expected_columns
