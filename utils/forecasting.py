import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go

# --- Prophet ---
def train_prophet(df, regressors=None, horizon=30, growth='linear', daily=True, weekly=True, yearly=True, seasonality_mode='additive', changepoint_prior_scale=0.05):
    """Train a Prophet model with full controls."""
    df = df.copy()

    if growth == 'logistic':
        df['cap'] = df['y'].max() * 1.5

    model = Prophet(
        growth=growth,
        daily_seasonality=daily,
        weekly_seasonality=weekly,
        yearly_seasonality=yearly,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale
    )

    if regressors:
        for reg in regressors:
            model.add_regressor(reg)

    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)

    model.fit(df)

    future = model.make_future_dataframe(periods=horizon)
    if growth == 'logistic':
        future['cap'] = df['cap'].iloc[-1]

    if regressors:
        for reg in regressors:
            future[reg] = df[reg].iloc[-1]

    forecast = model.predict(future)
    return model, forecast

def plot_prophet_forecast(model, forecast, df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    fig.update_layout(title="Actual vs Forecast (Prophet)", xaxis_title="Date", yaxis_title="Value")
    return fig

def plot_prophet_components(model, forecast):
    return plot_components_plotly(model, forecast)

# --- ARIMA ---
def train_arima(df, horizon=30, order=(5,1,0)):
    """Train an ARIMA model with customizable (p,d,q) order."""
    model = ARIMA(df['y'], order=order).fit()
    forecast = model.forecast(steps=horizon)
    return model, forecast

def plot_arima_forecast(df, forecast_arima, horizon):
    future_dates = pd.date_range(df['ds'].iloc[-1], periods=horizon+1, freq='D')[1:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast_arima, mode='lines', name='ARIMA Forecast'))
    fig.update_layout(title="Forecast (ARIMA)", xaxis_title="Date", yaxis_title="Value")
    return fig

# --- LSTM ---
def train_lstm(df, horizon=30, seq_len=10):
    """Train an LSTM model for forecasting."""
    data = df['y'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(seq_len, len(data_scaled)):
        X.append(data_scaled[i-seq_len:i])
        y.append(data_scaled[i])

    X, y = np.array(X), np.array(y)
    X_train, y_train = X[:-horizon], y[:-horizon]
    X_test = X[-horizon:]

    model = Sequential([
        LSTM(64, activation='relu', input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=16, callbacks=[EarlyStopping(patience=5)], verbose=0)

    pred = model.predict(X_test)
    pred = scaler.inverse_transform(pred)

    return model, pred.flatten()

def plot_lstm_forecast(df, pred_lstm, horizon):
    lstm_dates = pd.date_range(df['ds'].iloc[-1], periods=horizon+1, freq='D')[1:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=lstm_dates, y=pred_lstm, mode='lines', name='LSTM Forecast'))
    fig.update_layout(title="Forecast (LSTM)", xaxis_title="Date", yaxis_title="Value")
    return fig


