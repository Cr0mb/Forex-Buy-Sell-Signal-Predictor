import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import time
import diskcache
from dash.exceptions import PreventUpdate

MODEL_PATH = "forex_model.pkl"
CACHE_EXPIRY = 600
cache = diskcache.Cache('cache', size_limit=1e9)

print("Made by github.com/Cr0mb")

def fetch_forex_data(pair="EURUSD", period="7d", interval="5m"):
    cache_key = f"{pair}_{interval}"
    cached_data = cache.get(cache_key)
    
    if cached_data and time.time() - cached_data['timestamp'] < CACHE_EXPIRY:
        return cached_data['data']
    
    try:
        data = yf.download(f'{pair}=X', period=period, interval=interval)
        if not data.empty:
            data = data[['Open', 'High', 'Low', 'Close']]
            data.columns = ['open', 'high', 'low', 'close']
            data['timestamp'] = data.index
            cache.set(cache_key, {'data': data, 'timestamp': time.time()})
            return data
    except Exception as e:
        print(f"Error fetching data for {pair}: {e}")
    
    return pd.DataFrame()

def compute_technical_indicators(df):
    print("Computing technical indicators...")
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df['bollinger_upper'] = df['SMA_20'] + (df['close'].rolling(window=20).std() * 2)
    df['bollinger_lower'] = df['SMA_20'] - (df['close'].rolling(window=20).std() * 2)

    df['stochastic'] = 100 * (df['close'] - df['low'].rolling(window=14).min()) / (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min())
    
    print("Technical indicators computed.")
    return df

def preprocess_data(df, lag=5):
    print("Preprocessing data...")
    df['price_change'] = df['close'].pct_change()
    df['target'] = np.where(df['price_change'] > 0, 1, 0)

    for i in range(1, lag + 1):
        df[f'lag_{i}'] = df['close'].shift(i)

    df.ffill(inplace=True)
    df.dropna(inplace=True)

    if df.empty or df.shape[0] < 10:
        print("Insufficient data for training.")
        return pd.DataFrame(), pd.Series()

    indicators = ['SMA_20', 'EMA_20', 'RSI', 'MACD', 'MACD_signal', 'bollinger_upper', 'bollinger_lower', 'stochastic']
    
    features = ['open', 'high', 'low', 'close'] + [f'lag_{i}' for i in range(1, lag + 1)] + indicators
    print("Data preprocessing complete.")
    return df[features], df['target']

def train_model(X, y):
    print("Training model...")
    model = XGBClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 20, None],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    
    search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=3, verbose=1, n_jobs=-1, n_iter=5)
    search.fit(X, y)
    print(f"Model trained with best parameters: {search.best_params_}")
    return search.best_estimator_

def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        try:
            print("Loading saved model...")
            return joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}. Retraining...")
            os.remove(MODEL_PATH)
    
    print("Training a new model...")
    forex_data = fetch_forex_data("EURUSD")
    if forex_data.empty:
        print("Failed to fetch data.")
        return None

    forex_data = compute_technical_indicators(forex_data)
    X, y = preprocess_data(forex_data)
    if X.empty or y.empty:
        print("Not enough data for training.")
        return None

    model = train_model(X, y)
    joblib.dump(model, MODEL_PATH)
    print("Model training complete and saved.")
    return model

model = load_or_train_model()

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1("Forex Buy/Sell Signal Predictor", style={'textAlign': 'center', 'color': '#4A90E2', 'fontFamily': 'Arial, sans-serif', 'fontWeight': 'bold'}),
    ], style={'backgroundColor': '#f4f4f9', 'padding': '20px'}),
    
    html.Div([
        dcc.Dropdown(
            id='currency-pair',
            options=[{'label': pair, 'value': pair} for pair in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF"]],
            value='EURUSD',
            style={'width': '50%', 'margin': 'auto', 'fontSize': '18px', 'padding': '10px'}
        ),
    ], style={'width': '60%', 'margin': '20px auto', 'padding': '10px'}),
    
    html.Div([
        dcc.Dropdown(
            id='time-interval',
            options=[{'label': '5 Minutes', 'value': '5m'}, {'label': '15 Minutes', 'value': '15m'}, {'label': '1 Hour', 'value': '1h'}],
            value='5m',
            style={'width': '50%', 'margin': 'auto', 'fontSize': '18px', 'padding': '10px'}
        ),
    ], style={'width': '60%', 'margin': '20px auto', 'padding': '10px'}),
    
    html.Div([
        dcc.Graph(id='forex-graph', style={'height': '80vh', 'borderRadius': '10px'}),
    ], style={'margin': '20px auto'}),
    
    dcc.Interval(id='interval-update', interval=60*1000, n_intervals=0)
])

def plot_forex(df, pair):
    print("Plotting data...")
    buy_signal = df[df['signal'] == 'Buy']
    sell_signal = df[df['signal'] == 'Sell']

    trace_price = go.Candlestick(
        x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        increasing=dict(line=dict(color='green', width=2), fillcolor='rgba(0,255,0,0.3)'),
        decreasing=dict(line=dict(color='red', width=2), fillcolor='rgba(255,0,0,0.3)'),
        hovertext=df.apply(lambda row: 
                           f"<b>Timestamp:</b> {row['timestamp']}<br>"
                           f"<b>Close Price:</b> {row['close']:.2f}<br>"
                           f"<b>SMA (20):</b> {row['SMA_20']:.2f}<br>"
                           f"<b>EMA (20):</b> {row['EMA_20']:.2f}<br>"
                           f"<b>RSI:</b> {row['RSI']:.2f}<br>"
                           f"<b>MACD:</b> {row['MACD']:.2f}<br>"
                           f"<b>Bollinger Upper:</b> {row['bollinger_upper']:.2f}<br>"
                           f"<b>Bollinger Lower:</b> {row['bollinger_lower']:.2f}<br>"
                           f"<b>Stochastic Oscillator:</b> {row['stochastic']:.2f}", axis=1),
        hoverinfo='text'
    )

    trace_buy = go.Scatter(
        x=buy_signal['timestamp'], y=buy_signal['close'], mode='markers',
        marker=dict(symbol='triangle-up', color='lime', size=12, line=dict(width=2, color='black')), name='Buy Signal'
    )

    trace_sell = go.Scatter(
        x=sell_signal['timestamp'], y=sell_signal['close'], mode='markers',
        marker=dict(symbol='triangle-down', color='darkred', size=12, line=dict(width=2, color='black')), name='Sell Signal'
    )

    print(f"Buy signals: {len(buy_signal)} | Sell signals: {len(sell_signal)}")

    return {
        'data': [trace_price, trace_buy, trace_sell],
        'layout': go.Layout(
            title=f'{pair} Price with Buy/Sell Signals',
            xaxis=dict(
                title='Time',
                tickangle=45,
                showgrid=True,
                zeroline=False,
                tickformat='%H:%M:%S',
            ),
            yaxis=dict(
                title='Price',
                showgrid=True,
                zeroline=True
            ),
            hovermode='x unified',
            plot_bgcolor='#f4f4f9',
            paper_bgcolor='#ffffff',
            font=dict(family='Arial, sans-serif', size=12, color='#7f7f7f'),
            margin=dict(l=50, r=50, t=40, b=50),
            legend=dict(
                x=0.01, y=0.99,
                bgcolor='rgba(255, 255, 255, 0.7)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=12)
            )
        )
    }


@app.callback(
    dash.Output('forex-graph', 'figure'),
    [dash.Input('currency-pair', 'value'), dash.Input('time-interval', 'value'), dash.Input('interval-update', 'n_intervals')]
)
def update_graph(pair, interval, _):
    print(f"Fetching data for {pair} with interval {interval}...")
    df = fetch_forex_data(pair, interval=interval)
    
    if df.empty or model is None or df.shape[0] < 10:
        print("No valid data to process.")
        raise PreventUpdate

    df = compute_technical_indicators(df)
    X, y = preprocess_data(df)
    
    if X.empty or y.empty:
        print("No valid features to process.")
        raise PreventUpdate

    df['predicted_signal'] = model.predict(X)
    df['signal'] = np.where(df['predicted_signal'] == 1, 'Buy', 'Sell')

    print("Buy/Sell signal update complete.")
    return plot_forex(df, pair)

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8051)


