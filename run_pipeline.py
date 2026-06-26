"""
Rossmann Sales Forecasting — End-to-End Pipeline
Author: Adham Aboulkheir | University of Essex
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocess import generate_rossmann_data, create_temporal_features, encode_categoricals
from src.train import train_model


def main():
    print("=" * 55)
    print("ROSSMANN STORE SALES FORECASTING PIPELINE")
    print("Author: Adham Aboulkheir | University of Essex")
    print("=" * 55)
    
    os.makedirs("outputs", exist_ok=True)
    
    print("\n[1/4] Generating dataset...")
    df = generate_rossmann_data(n=10000)
    df = create_temporal_features(df)
    df, _ = encode_categoricals(df, ["StateHoliday", "StoreType", "Assortment"])
    df = df.dropna()
    print(f"  {len(df)} samples, {df.shape[1]} columns")
    
    print("\n[2/4] Training model...")
    results = train_model(df)
    m = results["metrics"]
    print(f"  MAE: {m['mae']:.0f} | RMSPE: {m['rmspe']:.4f} | R²: {m['r2']:.4f}")
    
    print("\n[3/4] Generating visualisations...")
    y_test = results["y_test"]
    y_pred = results["y_pred"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
    
    # Actual vs Predicted
    n_show = min(200, len(y_test))
    axes[0].plot(y_test[:n_show], color="#58a6ff", linewidth=1.2, label="Actual", alpha=0.8)
    axes[0].plot(y_pred[:n_show], color="#00c9b1", linewidth=1.2, linestyle="--", label="Predicted")
    axes[0].set_title("Actual vs Predicted Sales", color="white")
    axes[0].set_xlabel("Day", color="white")
    axes[0].set_ylabel("Sales (€)", color="white")
    axes[0].legend(facecolor="#161b22", labelcolor="white", fontsize=8)
    axes[0].tick_params(colors="white")
    axes[0].grid(alpha=0.3, color="#21262d")
    
    # Residuals
    residuals = y_test - y_pred
    axes[1].hist(residuals, bins=40, color="#00c9b1", alpha=0.8, edgecolor="none")
    axes[1].axvline(x=0, color="#f4a261", linewidth=2)
    axes[1].set_title("Residual Distribution", color="white")
    axes[1].set_xlabel("Residual (€)", color="white")
    axes[1].set_ylabel("Count", color="white")
    axes[1].tick_params(colors="white")
    axes[1].grid(alpha=0.3, color="#21262d")
    
    # Feature importance
    top_features = results["feature_importances"][:8]
    names = [f[0] for f in top_features]
    imps  = [f[1] for f in top_features]
    axes[2].barh(names[::-1], imps[::-1], color="#00c9b1", alpha=0.85)
    axes[2].set_title("Feature Importance", color="white")
    axes[2].set_xlabel("Importance", color="white")
    axes[2].tick_params(colors="white")
    axes[2].grid(axis="x", alpha=0.3, color="#21262d")
    
    plt.tight_layout()
    plt.savefig("outputs/rossmann_results.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("  Saved: outputs/rossmann_results.png")
    
    print("\n[4/4] Summary:")
    print(f"  Model: Gradient Boosting (300 estimators, depth=6)")
    print(f"  MAE:   {m['mae']:.0f} EUR")
    print(f"  RMSPE: {m['rmspe']:.4f} (competition metric)")
    print(f"  R²:    {m['r2']:.4f}")
    print("\n✓ Pipeline complete!")


if __name__ == "__main__":
    main()
