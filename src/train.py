"""
Rossmann Sales Forecasting — Training Pipeline
Author: Adham Aboulkheir | University of Essex
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import generate_rossmann_data, create_temporal_features
from src.preprocess import create_lag_features, create_rolling_features, encode_categoricals


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Percentage Error."""
    mask = y_true != 0
    return float(np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)))


FEATURE_COLS = [
    "Store", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday",
    "StoreType", "Assortment", "CompetitionDistance", "Promo2",
    "Year", "Month", "Week", "DayOfYear", "Quarter",
    "IsWeekend", "MonthStart", "MonthEnd",
    "SinMonth", "CosMonth", "SinDayOfWeek", "CosDayOfWeek",
]


def train_model(df: pd.DataFrame, feature_cols: list = None,
                target_col: str = "Sales") -> dict:
    """
    Train the sales forecasting model.
    
    Returns dict with model, scaler, metrics, and feature importances.
    """
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    
    # Train/test split (temporal)
    split_date = df["Date"].quantile(0.8)
    train_mask = df["Date"] <= split_date
    
    X_train = df.loc[train_mask, feature_cols].fillna(0).values
    y_train = df.loc[train_mask, target_col].values
    X_test  = df.loc[~train_mask, feature_cols].fillna(0).values
    y_test  = df.loc[~train_mask, target_col].values
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    
    print(f"  Training: {len(X_train)} samples | Test: {len(X_test)} samples")
    
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=20, random_state=42
    )
    model.fit(X_train_s, y_train)
    
    y_pred = model.predict(X_test_s)
    y_pred = np.clip(y_pred, 0, None)
    
    mae_val   = mean_absolute_error(y_test, y_pred)
    rmspe_val = rmspe(y_test, y_pred)
    r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
    
    importances = sorted(zip(feature_cols, model.feature_importances_),
                          key=lambda x: -x[1])
    
    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "metrics": {"mae": mae_val, "rmspe": rmspe_val, "r2": r2},
        "feature_importances": importances,
        "y_test": y_test,
        "y_pred": y_pred,
    }


if __name__ == "__main__":
    print("Rossmann Sales Forecasting — Training")
    print("=" * 45)
    
    df = generate_rossmann_data(n=10000)
    df = create_temporal_features(df)
    df, _ = encode_categoricals(df, ["StateHoliday", "StoreType", "Assortment"])
    df = df.dropna()
    
    print(f"Dataset: {len(df)} samples")
    results = train_model(df)
    
    m = results["metrics"]
    print(f"\nResults:")
    print(f"  MAE:   {m['mae']:.0f}")
    print(f"  RMSPE: {m['rmspe']:.4f}")
    print(f"  R²:    {m['r2']:.4f}")
    
    print("\nTop 5 Feature Importances:")
    for feat, imp in results["feature_importances"][:5]:
        print(f"  {feat:<30} {imp:.4f}")
