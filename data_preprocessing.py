import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

class PreprocessingPipeline:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.state_cols = []
        self.season_cols = []

    def fit(self, df):
        df = df.copy()
        # Calculate Fertilizer and Pesticide usage per area
        df["Fertilizer_Usage_per_Area"] = df["Fertilizer"] / (df["Area"] + 1e-6)
        df["Pesticide_Usage_per_Area"] = df["Pesticide"] / (df["Area"] + 1e-6)

        # One-hot encoding for categorical variables
        df = pd.get_dummies(df, columns=["State", "Season"], drop_first=True)

        # Save column names for later use
        self.state_cols = [col for col in df.columns if col.startswith("State_")]
        self.season_cols = [col for col in df.columns if col.startswith("Season_")]

        # Select relevant features for scaling (excluding "Area")
        features = ["Fertilizer_Usage_per_Area", "Pesticide_Usage_per_Area", "Annual_Rainfall"]
        self.scaler.fit(df[features])

    def transform(self, df):
        df = df.copy()
        # Calculate Fertilizer and Pesticide usage per area
        df["Fertilizer_Usage_per_Area"] = df["Fertilizer"] / (df["Area"] + 1e-6)
        df["Pesticide_Usage_per_Area"] = df["Pesticide"] / (df["Area"] + 1e-6)

        # One-hot encoding for categorical variables (same as in training)
        df = pd.get_dummies(df, columns=["State", "Season"], drop_first=True)

        # Ensure all necessary columns are present
        # Add missing state and season columns with 0 if they don't exist in the input data
        for col in self.state_cols + self.season_cols:
            if col not in df.columns:
                df[col] = 0

        # Ensure columns are in the correct order (matching the training data order)
        feature_order = ["Fertilizer_Usage_per_Area", "Pesticide_Usage_per_Area", "Annual_Rainfall"] + self.state_cols + self.season_cols
        df = df.reindex(columns=feature_order)

        # Only scale the relevant features (excluding "Area")
        features = ["Fertilizer_Usage_per_Area", "Pesticide_Usage_per_Area", "Annual_Rainfall"]
        df[features] = self.scaler.transform(df[features])

        return df
