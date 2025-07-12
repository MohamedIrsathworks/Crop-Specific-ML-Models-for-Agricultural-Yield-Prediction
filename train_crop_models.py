import pandas as pd
import os
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('crop_yield_cleaned.csv')

# Make output folder for models
os.makedirs("models", exist_ok=True)

# Model dictionary
model_dict = {
    "Random Forest": RandomForestRegressor,
    "Gradient Boosting": GradientBoostingRegressor,
    "XGBoost": XGBRegressor,
    "CatBoost": CatBoostRegressor,
    "SVM": SVR,
    "MLP": MLPRegressor
}

# Load best hyperparameters from a file
with open("best_models_params.json", "r") as f:
    best_models_params = json.load(f)

# Train models for each crop
for crop in sorted(df["Crop"].unique()):
    print(f"🚜 Training {crop}...")

    crop_df = df[df["Crop"] == crop].copy()

    # Feature engineering (calculate fertilizer/pesticide usage per area)
    crop_df["Fertilizer_Usage_per_Area"] = crop_df["Fertilizer"] / (crop_df["Area"] + 1e-6)
    crop_df["Pesticide_Usage_per_Area"] = crop_df["Pesticide"] / (crop_df["Area"] + 1e-6)

    # Prepare feature columns, remove 'Area' from the features
    X = crop_df[["Fertilizer_Usage_per_Area", "Pesticide_Usage_per_Area", "Annual_Rainfall", "State", "Season"]]
    y = crop_df["Yield"]

    # One-hot encoding for categorical variables (State and Season)
    X = pd.get_dummies(X, columns=["State", "Season"], drop_first=True)

    # Save feature names after one-hot encoding
    feature_names = X.columns.tolist()
    crop_safe = crop.replace("/", "_")
    with open(f"models/{crop_safe}_features.json", "w") as f:
        json.dump(feature_names, f)

    # Scaling numerical features per crop
    scaler = StandardScaler()
    X[["Fertilizer_Usage_per_Area", "Pesticide_Usage_per_Area", "Annual_Rainfall"]] = scaler.fit_transform(
        X[["Fertilizer_Usage_per_Area", "Pesticide_Usage_per_Area", "Annual_Rainfall"]]
    )

    # Save the scaler
    scaler_path = f"models/{crop_safe}_scaler.pkl"
    joblib.dump(scaler, scaler_path)

    # Get model and its best parameters
    model_name = best_models_params[crop]["model"]
    params = best_models_params[crop]["params"]

    # Instantiate and train model
    model_cls = model_dict[model_name]
    model = model_cls(**params)
    model.fit(X, y)

    # Save trained model
    model_path = f"models/{crop_safe}.pkl"
    joblib.dump(model, model_path)

    print(f"✅ {crop} model trained and saved.")

print("\n✅ All crop models trained and saved inside the 'models/' folder.")
