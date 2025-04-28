import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocessing import preprocess_data
from utils.eda import run_eda
from utils.anomaly import detect_anomalies
from utils.forecasting import (
    train_prophet, plot_prophet_forecast, plot_prophet_components,
    train_arima, plot_arima_forecast,
    train_lstm, plot_lstm_forecast
)
from utils.explainability import explain_model, plot_feature_importance
from utils.tuning import tune_prophet
from prophet.diagnostics import cross_validation, performance_metrics

# --- App Configuration ---
st.set_page_config(page_title="📈 Forecasting App ", layout="wide")
st.title("🔮 Forecasting App")

# --- Session State Initialization ---
for key in ["prophet_model", "prophet_forecast", "arima_model", "arima_forecast", "lstm_model", "lstm_forecast"]:
    if key not in st.session_state:
        st.session_state[key] = None

# --- Sidebar: File Upload and Settings ---
with st.sidebar:
    st.header("📂 Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

        date_cols = [col for col in df_raw.columns if 'date' in col.lower() or 'time' in col.lower()]
        num_cols = df_raw.select_dtypes(include='number').columns.tolist()

        date_col = st.selectbox("Select Date Column", date_cols if date_cols else df_raw.columns)
        target_col = st.selectbox("Select Target Column", num_cols if num_cols else df_raw.columns)
        other_cols = st.multiselect("External Regressors (Optional)", [col for col in df_raw.columns if col not in [date_col, target_col]])

        resample_needed = st.checkbox("🔄 Resample High Frequency Data?", value=False)
        resample_freq = st.selectbox("Select Resample Frequency", ["D", "W", "M"]) if resample_needed else None

# --- Main Tabs ---
if uploaded_file:
    df = preprocess_data(df_raw, date_col, target_col, external_cols=other_cols, resample_freq=resample_freq)

    tabs = st.tabs([":chart_with_upwards_trend: Data Overview", ":rotating_light: Anomaly Detection", ":crystal_ball: Forecasting", ":mag: Explainability"])

    # --- Data Overview ---
    with tabs[0]:
        run_eda(df)

    # --- Anomaly Detection ---
    with tabs[1]:
        st.header(":rotating_light: Anomaly Detection")
        contamination = st.slider("Outlier Contamination Fraction", 0.0, 0.1, 0.01)
        df_anomaly = detect_anomalies(df, contamination=contamination)

        st.subheader("Detected Anomalies")
        st.write(df_anomaly[df_anomaly['is_anomaly']])

        st.subheader("Anomaly Scatter Plot")
        fig = px.scatter(df_anomaly, x='ds', y='y', color='is_anomaly', title="Anomaly Detection Results")
        st.plotly_chart(fig, use_container_width=True)

        if st.checkbox("Remove Anomalies Before Forecasting"):
            df = df_anomaly[df_anomaly['is_anomaly'] == False]
            st.success("✅ Anomalies removed for modeling.")

    # --- Forecasting ---
    with tabs[2]:
        st.header(":crystal_ball: Forecasting")

        st.subheader("Choose Models to Train:")
        use_prophet = st.checkbox("✅ Prophet")
        use_arima = st.checkbox("✅ ARIMA")
        use_lstm = st.checkbox("✅ LSTM")

        horizon = st.number_input("Forecast Horizon (future periods)", min_value=1, value=30)

        # --- Prophet Specific Options
        growth = st.selectbox("Growth Type (for Prophet)", ["linear", "logistic"])
        seasonality_mode = st.selectbox("Seasonality Mode (for Prophet)", ["additive", "multiplicative"])
        changepoint_prior_scale = st.slider("Changepoint Prior Scale (for Prophet)", min_value=0.001, max_value=0.5, value=0.05, step=0.001)

        # --- ARIMA Specific Options
        p = st.number_input("ARIMA p (autoregressive term)", min_value=0, value=5)
        d = st.number_input("ARIMA d (differencing term)", min_value=0, value=1)
        q = st.number_input("ARIMA q (moving average term)", min_value=0, value=0)

        if st.button("🚀 Train Selected Models"):
            with st.spinner("Training Models... Please wait 🚀"):
                if use_prophet:
                    prophet_model, forecast = train_prophet(
                        df,
                        regressors=other_cols,
                        horizon=horizon,
                        growth=growth,
                        seasonality_mode=seasonality_mode,
                        changepoint_prior_scale=changepoint_prior_scale
                    )
                    st.session_state.prophet_model = prophet_model
                    st.session_state.prophet_forecast = forecast
                    st.success("✅ Prophet Model Trained!")
                if use_arima:
                    arima_model, forecast_arima = train_arima(df, horizon=horizon, order=(p, d, q))
                    st.session_state.arima_model = arima_model
                    st.session_state.arima_forecast = forecast_arima
                    st.success("✅ ARIMA Model Trained!")
                if use_lstm:
                    lstm_model, pred_lstm = train_lstm(df, horizon=horizon)
                    st.session_state.lstm_model = lstm_model
                    st.session_state.lstm_forecast = pred_lstm
                    st.success("✅ LSTM Model Trained!")

        if st.session_state.prophet_forecast is not None:
            st.subheader(":chart_with_upwards_trend: Prophet Forecast")
            st.plotly_chart(plot_prophet_forecast(st.session_state.prophet_model, st.session_state.prophet_forecast, df), use_container_width=True)

            if st.checkbox("Show Prophet Components"):
                st.plotly_chart(plot_prophet_components(st.session_state.prophet_model, st.session_state.prophet_forecast), use_container_width=True)

            if st.checkbox("🧪 Run Prophet Cross Validation"):
                try:
                    df_len = len(df)
                    initial = f"{int(df_len * 0.5)} days"
                    period = f"{int(df_len * 0.1)} days"
                    horizon_cv = f"{int(df_len * 0.2)} days"

                    df_cv = cross_validation(st.session_state.prophet_model, initial=initial, period=period, horizon=horizon_cv)
                    df_p = performance_metrics(df_cv)
                    st.dataframe(df_p)

                    st.metric("MAE", round(df_p['mae'].mean(), 2))
                    st.metric("RMSE", round(df_p['rmse'].mean(), 2))
                    st.metric("MAPE", f"{round(df_p['mape'].mean() * 100, 2)}%")
                except Exception as e:
                    st.warning(f"⚠️ Cross-validation could not run: {e}")

            with st.expander("🌟 Prophet Hyperparameter Tuning (Optional)"):
                run_tuning = st.checkbox("Enable Tuning")
                if run_tuning:
                    param_grid = {
                        'changepoint_prior_scale': [0.001, 0.01, 0.1],
                        'seasonality_mode': ['additive', 'multiplicative']
                    }
                    st.info("Running grid search across parameters... (This may take a few minutes)")
                    best_params, best_rmse = tune_prophet(df, param_grid, horizon=horizon)
                    st.success(f"✅ Best Parameters: {best_params}")
                    st.metric("Best RMSE", round(best_rmse, 2))

        if st.session_state.arima_forecast is not None:
            st.subheader(":chart_with_upwards_trend: ARIMA Forecast")
            st.plotly_chart(plot_arima_forecast(df, st.session_state.arima_forecast, horizon), use_container_width=True)

        if st.session_state.lstm_forecast is not None:
            st.subheader(":chart_with_upwards_trend: LSTM Forecast")
            st.plotly_chart(plot_lstm_forecast(df, st.session_state.lstm_forecast, horizon), use_container_width=True)

    # --- Explainability ---
    with tabs[3]:
        st.header(":mag: Explainability")
        if other_cols:
            st.info("🔎 Calculating Feature Importance...")
            importance_df = explain_model(df, other_cols)
            st.dataframe(importance_df)

            top_n = st.slider("Select Top N Features to Display", 1, min(10, len(importance_df)), 5)
            fig = plot_feature_importance(importance_df, top_n=top_n)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No external regressors selected to explain!")

# ✅ Now your app is fully professional, polished and changepoint tunable!
