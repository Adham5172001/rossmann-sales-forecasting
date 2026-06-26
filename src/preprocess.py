"""
Rossmann Store Sales — Feature Engineering Pipeline
Author: Adham Aboulkheir | University of Essex
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, List


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from Date column."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df.get("Date", pd.date_range("2013-01-01", periods=len(df))))
    df["Year"]        = df["Date"].dt.year
    df["Month"]       = df["Date"].dt.month
    df["Week"]        = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfYear"]   = df["Date"].dt.dayofyear
    df["Quarter"]     = df["Date"].dt.quarter
    df["IsWeekend"]   = (df["Date"].dt.dayofweek >= 5).astype(int)
    df["MonthStart"]  = (df["Date"].dt.day <= 7).astype(int)
    df["MonthEnd"]    = (df["Date"].dt.day >= 24).astype(int)
    df["SinMonth"]    = np.sin(2 * np.pi * df["Month"] / 12)
    df["CosMonth"]    = np.cos(2 * np.pi * df["Month"] / 12)
    df["SinDayOfWeek"]= np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["CosDayOfWeek"]= np.cos(2 * np.pi * df["DayOfWeek"] / 7)
    return df


def create_lag_features(df: pd.DataFrame, target_col: str = "Sales",
                         lags: List[int] = [7, 14, 30]) -> pd.DataFrame:
    """Create lag features for time series."""
    df = df.copy().sort_values("Date")
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame, target_col: str = "Sales",
                              windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
    """Create rolling window statistics."""
    df = df.copy().sort_values("Date")
    for w in windows:
        df[f"{target_col}_roll_mean_{w}"] = df[target_col].rolling(w, min_periods=1).mean()
        df[f"{target_col}_roll_std_{w}"]  = df[target_col].rolling(w, min_periods=1).std().fillna(0)
        df[f"{target_col}_roll_max_{w}"]  = df[target_col].rolling(w, min_periods=1).max()
    return df


def encode_categoricals(df: pd.DataFrame,
                          cat_cols: List[str] = None) -> Tuple[pd.DataFrame, dict]:
    """Label-encode categorical columns."""
    if cat_cols is None:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    encoders = {}
    df = df.copy()
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def generate_rossmann_data(n: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic Rossmann-like dataset."""
    np.random.seed(seed)
    dates = pd.date_range("2013-01-01", periods=n, freq="D")
    
    df = pd.DataFrame({
        "Date":                dates,
        "Store":               np.random.randint(1, 100, n),
        "DayOfWeek":           dates.dayofweek,
        "Promo":               np.random.randint(0, 2, n),
        "StateHoliday":        np.random.choice(["0", "a", "b", "c"], n, p=[0.85, 0.05, 0.05, 0.05]),
        "SchoolHoliday":       np.random.randint(0, 2, n),
        "StoreType":           np.random.choice(["a", "b", "c", "d"], n),
        "Assortment":          np.random.choice(["a", "b", "c"], n),
        "CompetitionDistance": np.random.exponential(3000, n),
        "Promo2":              np.random.randint(0, 2, n),
    })
    
    # Generate realistic sales with seasonality and promo effects
    base_sales = 5000
    seasonal = 2000 * np.sin(2 * np.pi * dates.dayofyear / 365)
    weekly   = 1000 * np.sin(2 * np.pi * dates.dayofweek / 7)
    promo_effect = 1500 * df["Promo"].values
    noise = np.random.normal(0, 400, n)
    
    df["Sales"] = (base_sales + seasonal + weekly + promo_effect + noise).clip(0)
    df["Customers"] = (df["Sales"] / np.random.uniform(4, 8, n)).astype(int)
    df["Open"] = (df["DayOfWeek"] < 6).astype(int)
    df.loc[df["Open"] == 0, "Sales"] = 0
    
    return df


if __name__ == "__main__":
    print("Rossmann Feature Engineering Demo")
    df = generate_rossmann_data(n=5000)
    df = create_temporal_features(df)
    df = create_lag_features(df, lags=[7, 14])
    df = create_rolling_features(df, windows=[7, 30])
    df, encoders = encode_categoricals(df, ["StateHoliday", "StoreType", "Assortment"])
    df = df.dropna()
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {[c for c in df.columns if c not in ['Date', 'Sales', 'Customers'][:5]]}")
    print(f"Sales stats: mean={df.Sales.mean():.0f}, std={df.Sales.std():.0f}")
