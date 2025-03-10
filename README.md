![image](https://github.com/user-attachments/assets/9d41132f-5e0a-4162-a0fd-384a413c5065)


# Forex-Buy-Sell-Signal-Predictor
This project is a Dash-based web application for predicting buy and sell signals in Forex trading, powered by historical Forex data and machine learning techniques. The system fetches real-time Forex data, computes various technical indicators, and uses an XGBoost model to predict buy or sell signals.

## Features
- Currency Pair Selection: Choose from major Forex pairs like EURUSD, GBPUSD, USDJPY, AUDUSD, and USDCHF.
- Time Interval Selection: Choose the time interval for the Forex data (e.g., 5 minutes, 15 minutes, 1 hour).
- Technical Indicators: Displays key indicators such as SMA, EMA, RSI, MACD, Bollinger Bands, and Stochastic Oscillator.
- Machine Learning Model: Uses an XGBoost classifier to predict buy/sell signals based on the computed indicators.
- Real-time Updates: Automatically updates data and signals every minute.
  
## Requirements
To run this project locally, you need to install the following Python packages:

1. ``dash`` - Web framework for building the application.
2. ``yfinance`` - Fetches historical Forex data.
3. ``xgboost`` - Machine learning library for classification.
4. ``sklearn`` - Used for model training and evaluation.
5. ``plotly`` - Data visualization.
6. ``diskcache`` - Caching data to optimize performance.

## How It Works
- Data Fetching
The app uses the yfinance library to fetch historical Forex data for the selected currency pair. The data is cached for efficiency, avoiding repeated requests to the Yahoo Finance API.

- Technical Indicators
The app computes multiple technical indicators such as Simple Moving Average (SMA), Exponential Moving Average (EMA), Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), Bollinger Bands, and Stochastic Oscillator.
These indicators are used as features for the machine learning model.

- Model Training
If no pre-trained model is found, a new XGBoost classifier is trained on the historical data. It uses the technical indicators to predict buy (1) or sell (0) signals.

- Predictions
The trained model predicts whether the market will go up (Buy) or down (Sell) based on the current Forex data and indicators.

- Plotting
The app visualizes the Forex data in a candlestick chart and overlays buy/sell signals on the chart using Plotly.

- Real-time Updates
The graph and signals are updated every minute, ensuring that users get the most recent Forex market data and predictions.

## Caching
```Forex data is cached for improved performance. ```

The cache expires after 10 minutes ``(CACHE_EXPIRY = 600 seconds)``, after which fresh data is fetched.

## Model
The model is trained using the XGBoost algorithm, and its hyperparameters are optimized using RandomizedSearchCV. 

The trained model is saved as ``forex_model.pkl`` to avoid retraining on each run.

If the model file is not found or is corrupted, the app will retrain the model using the latest data.
