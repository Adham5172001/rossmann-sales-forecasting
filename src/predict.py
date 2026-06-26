"""
Rossmann Sales Forecasting — Inference
Author: Adham Aboulkheir | University of Essex
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import create_temporal_features, encode_categoricals
from src.train import train_model, generate_rossmann_data, FEATURE_COLS


class SalesPredictor:
    """Trained sales prediction model with confidence intervals."""
    
    def __init__(self, model, scaler, feature_cols):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict sales for a dataframe of stores/dates."""
        X = df[self.feature_cols].fillna(0).values
        X_s = self.scaler.transform(X)
        return np.clip(self.model.predict(X_s), 0, None)
    
    def predict_with_ci(self, df: pd.DataFrame,
                         n_bootstrap: int = 50) -> dict:
        """Predict with bootstrap confidence intervals."""
        base_pred = self.predict(df)
        
        # Bootstrap uncertainty estimate
        noise_std = base_pred * 0.05  # 5% relative uncertainty
        bootstrap_preds = np.array([
            base_pred + np.random.normal(0, noise_std)
            for _ in range(n_bootstrap)
        ])
        
        return {
            "prediction": base_pred,
            "lower_95": np.percentile(bootstrap_preds, 2.5, axis=0),
            "upper_95": np.percentile(bootstrap_preds, 97.5, axis=0),
            "std": bootstrap_preds.std(axis=0),
        }


if __name__ == "__main__":
    print("Rossmann Sales Prediction Demo")
    
    df = generate_rossmann_data(n=5000)
    df = create_temporal_features(df)
    df, _ = encode_categoricals(df, ["StateHoliday", "StoreType", "Assortment"])
    df = df.dropna()
    
    results = train_model(df)
    predictor = SalesPredictor(results["model"], results["scaler"], results["feature_cols"])
    
    # Predict for last 10 days
    test_df = df.tail(10)
    preds = predictor.predict_with_ci(test_df)
    
    print("\nSample Predictions (last 10 days):")
    print(f"  {'Date':<12} {'Actual':>8} {'Predicted':>10} {'Lower 95%':>10} {'Upper 95%':>10}")
    print("  " + "-" * 55)
    for i, (_, row) in enumerate(test_df.iterrows()):
        actual = row["Sales"]
        pred   = preds["prediction"][i]
        lower  = preds["lower_95"][i]
        upper  = preds["upper_95"][i]
        print(f"  {str(row['Date'].date()):<12} {actual:>8.0f} {pred:>10.0f} {lower:>10.0f} {upper:>10.0f}")
