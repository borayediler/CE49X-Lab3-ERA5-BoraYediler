#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lab 3 — ERA5 Weather Data Analysis (Berlin & Munich)
Version 0: Load and Explore Datasets (without argparse)
BORA YEDİLER - 2021403186 - CE49X LAB 3

"""

# First lets import the necessary libraries
import sys
from pathlib import Path
import pandas as pd
import numpy as np
# Below two is used for ploting graphs and charts
import matplotlib.pyplot as plt
import seaborn as sns

# This function reads the ERA5 CSV files and performs basic cleaning.
def read_era5_csv(path: Path) -> pd.DataFrame:
    """
    Read an ERA5 hourly CSV file and perform basic cleaning.
    Expected columns: time (or timestamp), u10m, v10m  (optional: t2m)
    """
    # 1- File check if it exists
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # 2- Read CSV file
    df = pd.read_csv(path)

    # ---- normalize time column name ----
    # If the file has 'timestamp' instead of 'time', rename it to 'time'.
    # We did this because AI suggested that converting the timestamp into time format would be more flexible for the future.
    if "time" not in df.columns and "timestamp" in df.columns:
        df.rename(columns={"timestamp": "time"}, inplace=True) # inplace True means that the changes are made in the original dataframe.

    # 3- Check required columns if they are in the file
    required = {"time", "u10m", "v10m"} # Sşnce we changed the timestamp into time, we will work with time variable instead of timestamp.
    missing = required - set(df.columns) # If the columns are not in the file, raise an error.
    if missing:
        raise ValueError(f"Missing column(s) in {path.name}: {sorted(missing)}") # If the columns are not in the file, raise an error.  

    # 4- Convert to correct data types
    # If there is improper data, to_* with errors='coerce' will turn them into NaT/NaN. 
    # to_datetime converts the time column to datetime format
    # to_numeric converts the u10m and v10m columns to numeric format
    # to_numeric converts the t2m column to numeric format if it exists
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for c in ["u10m", "v10m"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "t2m" in df.columns:
        df["t2m"] = pd.to_numeric(df["t2m"], errors="coerce")

    # 5- Clean invalid times and duplicates
    df = (
        df.dropna(subset=["time"])
          .sort_values("time")
          .drop_duplicates(subset=["time"])
          .reset_index(drop=True)
    )

    # 6- Handle missing values (small gaps) for wind components
    df[["u10m", "v10m"]] = df[["u10m", "v10m"]].interpolate(limit=3)  # linear interpolation
    df[["u10m", "v10m"]] = df[["u10m", "v10m"]].bfill().ffill()       # backfill & forward fill

    return df

# --- Summary statistics manually calculated ---
def print_summary(df: pd.DataFrame, city_name: str):
    print(f"\n=== Summary Statistics for {city_name} ===")
    for col in ["u10m", "v10m", "t2m"]:
        if col in df.columns:
            col_data = df[col].dropna()
            print(f"{col}:")
            print(f"  Min:   {col_data.min():.3f}")
            print(f"  Max:   {col_data.max():.3f}")
            print(f"  Mean:  {col_data.mean():.3f}")
            print(f"  Median:{col_data.median():.3f}")
            print(f"  Std:   {col_data.std():.3f}\n")

# === Load and summarize the datasets ===
script_dir = Path(__file__).parent.resolve() # Finds the python file we are currently working on.

# datasets klasörü repo kökünde (labs ile aynı seviyede)
berlin_path = script_dir.parents[1] / "datasets" / "berlin_era5_wind_20241231_20241231.csv" # Get 1 file from the python file we are currently working on.
munich_path = script_dir.parents[1] / "datasets" / "munich_era5_wind_20241231_20241231.csv" # Get 1 file from the python file we are currently working on.

try: # Here reads the csv 2 files with the function we created.
    df_b = read_era5_csv(berlin_path)
    df_m = read_era5_csv(munich_path)
except Exception as e:
    print(f"[ERROR] {e}", file=sys.stderr)
    sys.exit(1)

# Print summary statistics for both datasets
print_summary(df_b, "Berlin")
print_summary(df_m, "Munich")

def compute_wind_speed(df: pd.DataFrame) -> pd.DataFrame: # Calculates the wind speed 
    # Compute wind speed (m/s) from u10m and v10m components.
    # Adds a new column 'wind_speed' to the dataframe.
    # Formula: sqrt(u10m^2 + v10m^2)
    if not {"u10m", "v10m"}.issubset(df.columns): # This checks if the columns u10m and v10m are in the dataframe.
        raise ValueError("Both u10m and v10m columns are required to compute wind speed.")

    df["wind_speed"] = np.sqrt(df["u10m"]**2 + df["v10m"]**2)
    return df

# --- Compute wind speed for both datasets ---
df_b = compute_wind_speed(df_b)
df_m = compute_wind_speed(df_m)

# --- Calculate monthly averages for wind speed and temperature (if available) ---
# Returns the month of the year from the time column as a number (1-12).
df_b["month"] = df_b["time"].dt.month
df_m["month"] = df_m["time"].dt.month

# Groups the data by month and calculates the mean of the wind speed. t2m part means thatit uses the temperature information if exists.
monthly_avg_b = df_b.groupby("month")[["wind_speed"] + (["t2m"] if "t2m" in df_b.columns else [])].mean()
monthly_avg_m = df_m.groupby("month")[["wind_speed"] + (["t2m"] if "t2m" in df_m.columns else [])].mean()

print("\n=== Monthly Averages — Berlin ===")
print(monthly_avg_b)

print("\n=== Monthly Averages — Munich ===")
print(monthly_avg_m)

# --- Define helper for seasons ---
def month_to_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

df_b["season"] = df_b["month"].apply(month_to_season)
df_m["season"] = df_m["month"].apply(month_to_season)

seasonal_avg_b = df_b.groupby("season")[["wind_speed"]].mean()
seasonal_avg_m = df_m.groupby("season")[["wind_speed"]].mean()

print("\n=== Seasonal Average Wind Speeds — Berlin ===")
print(seasonal_avg_b)

print("\n=== Seasonal Average Wind Speeds — Munich ===")
print(seasonal_avg_m)

# Now we will plot the graphs.
# --- Time Series Plot of Monthly Average Wind Speeds ---
plt.figure(figsize=(10, 5))
plt.plot(monthly_avg_b.index, monthly_avg_b["wind_speed"], marker="o", label="Berlin")
plt.plot(monthly_avg_m.index, monthly_avg_m["wind_speed"], marker="o", label="Munich")

plt.title("Monthly Average Wind Speed (2024)")
plt.xlabel("Month")
plt.ylabel("Wind Speed (m/s)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# --- Bar Chart of Seasonal Averages ---
seasonal_compare = pd.DataFrame({
    "Berlin": seasonal_avg_b["wind_speed"],
    "Munich": seasonal_avg_m["wind_speed"]
})

seasonal_compare.plot(kind="bar", figsize=(8, 5))
plt.title("Seasonal Average Wind Speeds (Berlin vs Munich)")
plt.xlabel("Season")
plt.ylabel("Wind Speed (m/s)")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# --- Compute wind direction (in degrees) ---
df_b["wind_dir"] = np.degrees(np.arctan2(df_b["v10m"], df_b["u10m"]))
df_m["wind_dir"] = np.degrees(np.arctan2(df_m["v10m"], df_m["u10m"]))

def direction_category(angle):
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int(((angle + 360) % 360) // 45)
    return dirs[idx]

df_b["dir_cat"] = df_b["wind_dir"].apply(direction_category)
df_m["dir_cat"] = df_m["wind_dir"].apply(direction_category)

# --- Wind Direction Distribution (Wind Rose-like bar plot) ---
plt.figure(figsize=(8, 5))
sns.countplot(x="dir_cat", data=df_b, order=["N","NE","E","SE","S","SW","W","NW"], color="skyblue", label="Berlin")
sns.countplot(x="dir_cat", data=df_m, order=["N","NE","E","SE","S","SW","W","NW"], color="salmon", alpha=0.6, label="Munich")

plt.title("Wind Direction Distribution (Berlin vs Munich)")
plt.xlabel("Direction")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()
