from prophet import Prophet
from sklearn.model_selection import ParameterGrid
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Prophet Hyperparameter Tuning
def tune_prophet(df, param_grid, horizon=30):
    best_rmse = float('inf')
    best_params = {}

    for params in ParameterGrid(param_grid):
        try:
            model = Prophet(
                changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
                seasonality_mode=params.get('seasonality_mode', 'additive')
            )
            model.fit(df)
            future = model.make_future_dataframe(periods=horizon)
            forecast = model.predict(future)

            y_true = df['y'].iloc[-horizon:].values
            y_pred = forecast['yhat'].iloc[-horizon:].values[-horizon:]

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
        except Exception as e:
            continue

    return best_params, best_rmse