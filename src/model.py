"""
Rossmann Store Sales Forecasting
Author: Adham Aboulkheir | University of Essex
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor


def create_features(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df.get("Date", pd.date_range("2013-01-01", periods=len(df))))
    df["Year"]      = df["Date"].dt.year
    df["Month"]     = df["Date"].dt.month
    df["Week"]      = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Quarter"]   = df["Date"].dt.quarter
    df["IsWeekend"] = (df["Date"].dt.dayofweek >= 5).astype(int)
    return df


def rmspe(y_true, y_pred):
    mask = y_true != 0
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))


if __name__ == "__main__":
    print("Rossmann Store Sales Forecasting Demo")
    np.random.seed(42)
    n = 5000
    dates = pd.date_range("2013-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "Date": dates, "Store": np.random.randint(1, 50, n),
        "DayOfWeek": dates.dayofweek, "Promo": np.random.randint(0, 2, n),
        "StoreType": np.random.choice(["a","b","c","d"], n),
        "Assortment": np.random.choice(["a","b","c"], n),
        "CompetitionDistance": np.random.exponential(3000, n),
    })
    df["Sales"] = (5000 + 2000*np.sin(2*np.pi*dates.dayofyear/365)
                   + 1500*df["Promo"] + np.random.normal(0, 400, n)).clip(0)
    df = create_features(df)
    for col in ["StoreType", "Assortment"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    feature_cols = ["Store","DayOfWeek","Promo","StoreType","Assortment",
                    "CompetitionDistance","Year","Month","Week","DayOfYear","Quarter","IsWeekend"]
    X, y = df[feature_cols].values, df["Sales"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    model = GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42)
    model.fit(scaler.fit_transform(X_train), y_train)
    y_pred = model.predict(scaler.transform(X_test))
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.0f}")
    print(f"RMSPE: {rmspe(y_test, y_pred):.4f}")
    r2 = 1 - np.sum((y_test-y_pred)**2)/np.sum((y_test-np.mean(y_test))**2)
    print(f"R2: {r2:.4f}")
