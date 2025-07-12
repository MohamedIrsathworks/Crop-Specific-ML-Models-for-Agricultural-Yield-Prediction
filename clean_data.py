# clean_data.py

import pandas as pd

# Load dataset
df = pd.read_csv("crop_yield.csv")

# Strip spaces
df['Season'] = df['Season'].str.strip()
df['Crop'] = df['Crop'].str.strip()
df['State'] = df['State'].str.strip()

# 1. Remove small Crop-State-Season groups (< 3 entries)
group_counts = df.groupby(["Crop", "State", "Season"]).size()
small_groups = group_counts[group_counts < 3].index
df = df[~df.set_index(["Crop", "State", "Season"]).index.isin(small_groups)]

# 2. Replace outliers within each Crop-State-Season group
def replace_outliers_with_mean(df, column_name):
    grouped = df.groupby(["Crop", "State", "Season"])
    for (crop, state, season), group in grouped:
        if len(group) < 3:
            continue

        Q1 = group[column_name].quantile(0.25)
        Q3 = group[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        valid = group[(group[column_name] >= lower) & (group[column_name] <= upper)][column_name]

        if valid.empty:
            continue

        mean_valid = valid.mean()
        mask = (
            (df["Crop"] == crop) & (df["State"] == state) & (df["Season"] == season) &
            ((df[column_name] < lower) | (df[column_name] > upper))
        )
        df.loc[mask, column_name] = mean_valid

    return df

numerical_cols = ["Yield", "Fertilizer", "Pesticide", "Annual_Rainfall"]
for col in numerical_cols:
    df = replace_outliers_with_mean(df, col)

# 3. Combine crops with < 100 samples as "Other_Crops"
crop_counts = df["Crop"].value_counts()
other_crops = crop_counts[crop_counts < 100].index
df["Crop"] = df["Crop"].replace(other_crops, "Other_Crops")

# 4. Save cleaned dataset
df.to_csv("crop_yield_cleaned.csv", index=False)

print("✅ Data cleaned and saved to crop_yield_cleaned.csv")
