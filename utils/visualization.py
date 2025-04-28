# utils/visualization.py

import plotly.graph_objs as go

def plot_forecast(df, forecast):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    fig.update_layout(title="Actual vs Forecast", xaxis_title="Date", yaxis_title="Value")
    return fig
