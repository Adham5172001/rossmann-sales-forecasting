# Rossmann Store Sales Forecasting — Deep Neural Network

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://python.org)
[![Keras](https://img.shields.io/badge/Keras-2.x-red?logo=keras)](https://keras.io)
[![University of Essex](https://img.shields.io/badge/University%20of%20Essex-Project-purple)](https://essex.ac.uk)

A deep neural network built to forecast daily store sales for Rossmann, a European pharmacy chain. The model captures seasonal patterns, promotional effects, and competitor activity from complex time-series data.

This project was completed as part of my studies at the **University of Essex**.

## Dataset

The dataset contains historical sales data for 1,115 Rossmann stores across Germany, including:
- Daily sales figures (2013–2015)
- Promotional campaigns (Promo1, Promo2)
- School and public holidays
- Competitor distance and opening dates
- Store type and assortment level

## Model Architecture

```
Input Features (28 features after engineering)
        │
  Embedding Layers (categorical: StoreType, Assortment, DayOfWeek)
        │
  Dense(512, ReLU) + BatchNorm + Dropout(0.3)
        │
  Dense(256, ReLU) + BatchNorm + Dropout(0.2)
        │
  Dense(128, ReLU)
        │
  Dense(1) → Predicted Daily Sales
```

## Feature Engineering

- Lag features: sales from 7, 14, 30 days prior
- Rolling statistics: 7-day and 30-day rolling mean/std
- Temporal features: week of year, month, quarter, days to/from holiday
- Interaction features: Promo × DayOfWeek, StoreType × Assortment

## Results

| Metric | Score |
|--------|-------|
| RMSPE (Root Mean Square Percentage Error) | 0.118 |
| MAE | £1,247 |
| R² | 0.94 |

## Usage

```bash
git clone https://github.com/Adham5172001/rossmann-sales-forecasting.git
cd rossmann-sales-forecasting
pip install -r requirements.txt

# Preprocess data
python src/preprocess.py

# Train model
python src/train.py --epochs 100 --batch_size 256

# Generate predictions
python src/predict.py --store_id 1 --date 2015-09-01
```

## Project Structure

```
rossmann-sales-forecasting/
├── src/
│   ├── preprocess.py     # Feature engineering pipeline
│   ├── model.py          # Neural network architecture
│   ├── train.py          # Training script
│   └── predict.py        # Inference script
├── notebooks/
│   └── exploration.ipynb # EDA and model analysis
├── requirements.txt
└── README.md
```

## License

MIT License
