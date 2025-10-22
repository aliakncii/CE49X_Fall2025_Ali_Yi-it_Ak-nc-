"""
Lab 3 – ERA5 Weather Data Analysis (Berlin vs Munich)
Run from repo root:
    python labs/lab3/lab3_era5_analysis.py

This script (simple & clear):
- Loads 2 CSV files (Berlin & Munich) with ERA5-like columns: timestamp,u10m,v10m,lat,lon
- Cleans data (parse time, drop missing)
- Computes wind speed (ws) and wind direction (dir)
- Makes monthly, seasonal, and diurnal (hourly) aggregations
- Plots 3+ figures
- Prints descriptive stats and “extreme wind days”
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- general plot style ----------
sns.set(style="whitegrid")


# ---------- helpers ----------
def season_from_month(m: int) -> str:
    """Return season name for a given month number."""
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Spring"
    if m in (6, 7, 8):
        return "Summer"
    return "Autumn"


def ensure_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure we have a datetime column named 'time'.
    Your CSV has 'timestamp', so we convert that to 'time'.
    If a file already has 'time', we keep it.
    """
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    elif "timestamp" in df.columns:
        df["time"] = pd.to_datetime(df["timestamp"])
        # keep original columns but drop raw timestamp to avoid confusion
        df.drop(columns=["timestamp"], inplace=True)
    else:
        raise KeyError("CSV must include 'timestamp' or 'time' column.")
    return df


def compute_ws_dir(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute wind speed (m/s) and wind direction (deg).
    Note: dir is meteorological-like: 0/360 = north, 90 = east.
    """
    # u10m: zonal (west<->east), v10m: meridional (south<->north)
    df["ws"] = np.sqrt(df["u10m"] ** 2 + df["v10m"] ** 2)
    # direction: from where the wind comes; here a simple arctan2 transform
    # Using arctan2(u, v) and converting to degrees [0, 360)
    df["dir_deg"] = (np.degrees(np.arctan2(df["u10m"], df["v10m"])) + 360) % 360
    return df


def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Add year, month, hour, season columns for grouping."""
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["hour"] = df["time"].dt.hour
    df["season"] = df["month"].map(season_from_month)
    return df


def load_city_csv(path: Path, city: str) -> pd.DataFrame:
    """
    Load a city CSV safely.
    Expected columns: timestamp,u10m,v10m,lat,lon (or 'time' instead of 'timestamp')
    """
    print(f"[LOAD] {city}: {path}")
    df = pd.read_csv(path)
    # minimal sanity: keep only needed cols if present
    keep = [c for c in ["timestamp", "time", "u10m", "v10m", "lat", "lon"] if c in df.columns]
    df = df[keep].copy()
    df = ensure_time_column(df)
    # drop rows with missing essential values
    df = df.dropna(subset=["u10m", "v10m", "time"]).reset_index(drop=True)
    df = compute_ws_dir(df)
    df = add_time_parts(df)
    return df


def basic_summary(df: pd.DataFrame, city: str) -> None:
    """Print quick descriptive info (numeric only)."""
    print(f"\n=== BASIC INFO: {city} ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    # select numeric columns to avoid pandas version differences
    num_cols = df.select_dtypes(include="number")
    print(num_cols.describe().round(3))


def monthly_avg_ws(df: pd.DataFrame) -> pd.Series:
    """Monthly mean wind speed."""
    return df.groupby("month")["ws"].mean()


def seasonal_avg_ws(df: pd.DataFrame) -> pd.Series:
    """Seasonal mean wind speed."""
    # order seasons for nicer plot
    order = ["Winter", "Spring", "Summer", "Autumn"]
    s = df.groupby("season")["ws"].mean()
    return s.reindex(order)


def diurnal_avg_ws(df: pd.DataFrame) -> pd.Series:
    """Mean wind speed by hour-of-day (0..23)."""
    return df.groupby("hour")["ws"].mean()


def find_extreme_days(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """
    Find top-k days with highest daily max wind speed.
    Returns a small table with date, max_ws.
    """
    daily_max = df.set_index("time").resample("D")["ws"].max().dropna()
    top = daily_max.sort_values(ascending=False).head(k).round(3).reset_index()
    top.columns = ["date", "max_ws"]
    return top


# ---------- plotting ----------
def plot_monthly_comparison(dfA: pd.DataFrame, dfB: pd.DataFrame, cityA: str, cityB: str, outdir: Path):
    """Line chart: monthly average wind speed for two cities."""
    a = monthly_avg_ws(dfA)
    b = monthly_avg_ws(dfB)

    plt.figure(figsize=(8, 5))
    sns.lineplot(x=a.index, y=a.values, marker="o", label=cityA)
    sns.lineplot(x=b.index, y=b.values, marker="o", label=cityB)
    plt.title("Monthly Average Wind Speed")
    plt.xlabel("Month")
    plt.ylabel("Wind speed (m/s)")
    plt.xticks(range(1, 13))
    plt.legend()
    plt.tight_layout()
    (outdir / "fig_monthly_ws.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / "fig_monthly_ws.png", dpi=160)
    plt.show()


def plot_seasonal_bars(dfA: pd.DataFrame, dfB: pd.DataFrame, cityA: str, cityB: str, outdir: Path):
    """Bar chart: seasonal average wind speed."""
    a = seasonal_avg_ws(dfA)
    b = seasonal_avg_ws(dfB)
    seasons = a.index

    data = pd.DataFrame({cityA: a.values, cityB: b.loc[seasons].values}, index=seasons)

    plt.figure(figsize=(8, 5))
    data.plot(kind="bar")
    plt.title("Seasonal Average Wind Speed")
    plt.xlabel("Season")
    plt.ylabel("Wind speed (m/s)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(outdir / "fig_seasonal_ws.png", dpi=160)
    plt.show()


def plot_diurnal(dfA: pd.DataFrame, dfB: pd.DataFrame, cityA: str, cityB: str, outdir: Path):
    """Line chart: average wind speed by hour-of-day."""
    a = diurnal_avg_ws(dfA)
    b = diurnal_avg_ws(dfB)

    plt.figure(figsize=(8, 5))
    sns.lineplot(x=a.index, y=a.values, marker="o", label=cityA)
    sns.lineplot(x=b.index, y=b.values, marker="o", label=cityB)
    plt.title("Diurnal Pattern of Wind Speed")
    plt.xlabel("Hour of day")
    plt.ylabel("Wind speed (m/s)")
    plt.xticks(range(0, 24, 2))
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fig_diurnal_ws.png", dpi=160)
    plt.show()


def print_skyrim_note():
    """Short B1-level description for the report about Skyrim repo."""
    print(
        "\n[Skyrim – short description]\n"
        "Skyrim is an open-source tool to run modern large weather models with a single interface.\n"
        "It can fetch initial conditions (e.g., GFS/IFS), run models like GraphCast/FourCastNet/Pangu,\n"
        "and help visualize forecasts. This is useful for civil/environmental projects that need local\n"
        "weather predictions (e.g., wind resource, flood risk, scheduling).\n"
    )


# ---------- main ----------
def main():
    # paths
    repo_root = Path(__file__).resolve().parents[2]  # go to repo root
    data_dir = repo_root / "datasets"
    out_dir = repo_root / "labs" / "lab3" / "figs"

    # filenames (as given by the instructor)
    berlin_csv = data_dir / "berlin_era5_wind_20241231_20241231.csv"
    munich_csv = data_dir / "munich_era5_wind_20241231_20241231.csv"

    # load
    try:
        df_berlin = load_city_csv(berlin_csv, "Berlin")
        df_munich = load_city_csv(munich_csv, "Munich")
    except Exception as e:
        print(f"[ERROR] Data load failed: {e}")
        return

    # summaries
    basic_summary(df_berlin, "Berlin")
    basic_summary(df_munich, "Munich")

    # extremes
    print("\n=== EXTREME WIND DAYS (Top 3) ===")
    print("Berlin:\n", find_extreme_days(df_berlin, 3))
    print("Munich:\n", find_extreme_days(df_munich, 3))

    # plots (3 figures)
    plot_monthly_comparison(df_berlin, df_munich, "Berlin", "Munich", out_dir)
    plot_seasonal_bars(df_berlin, df_munich, "Berlin", "Munich", out_dir)
    plot_diurnal(df_berlin, df_munich, "Berlin", "Munich", out_dir)

    # short Skyrim note for the report
    print_skyrim_note()

    print("\n[OK] Finished. Figures saved under:", out_dir)


if __name__ == "__main__":
    main()
