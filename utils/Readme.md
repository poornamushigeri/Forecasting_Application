📈 Forecasting App
A powerful Streamlit application for Time Series Forecasting across various domains — sales, finance, energy, weather, and more.

🚀 Features

🔮 Forecast with Prophet, ARIMA, and LSTM models
🆚 Compare multiple models side by side
➕ Multivariate Forecasting with external regressors
🚨 Anomaly Detection using Isolation Forest
🛠️ Automatic Data Preprocessing and Resampling
📊 Exploratory Data Analysis (Fast EDA, Sweetviz, YData Profiling)
✅ Prophet Cross-validation and evaluation metrics (MAE, RMSE, MAPE)
🎯 Hyperparameter Tuning for Prophet
🧠 Model Explainability via Feature Importance

🛠️ Tech Stack

Frontend: Streamlit, Plotly
Forecasting Models: Prophet, ARIMA (statsmodels), LSTM (TensorFlow/Keras)
Data Analysis: Sweetviz, YData Profiling
Machine Learning: Scikit-learn

📂 Project Structure

forecasting_app/
│
├── app.py                  # Main Streamlit App
├── requirements.txt        # Python dependencies
│
└── utils/
    ├── preprocessing.py    # Data cleaning and resampling
    ├── eda.py               # Exploratory Data Analysis tools
    ├── anomaly.py           # Anomaly detection methods
    ├── forecasting.py       # Forecasting models (Prophet, ARIMA, LSTM)
    ├── explainability.py    # Feature importance visualization
    └── tuning.py            # Hyperparameter tuning modules

📦 Installation Guide
Clone the Repository
git clone https://github.com/poornamushigeri/Forecasting_Application.git
cd forecasting-app

Install Dependencies
pip install -r requirements.txt

Run the App
streamlit run forecasting_app/app.py

🖥️ How to Use
Upload your CSV time series file
Select the Date and Target columns
Perform EDA (Built-in EDA, Sweetviz, or YData Profiling)
Detect and Handle Anomalies (optional)
Train Models (Prophet, ARIMA, LSTM)
Validate Models (Prophet cross-validation, error metrics)
Tune Hyperparameters (Prophet tuning)
Visualize Forecasts and Analyze Feature Importance

📋 Example Datasets
✈️ Monthly Airline Passengers
📈 Stock Market Prices
⚡ Electricity Demand
🛒 Retail Sales Data
🌡️ Weather Time Series

📛 Badges
<p align="center"> <img src="https://img.shields.io/badge/Streamlit-Application-orange?logo=streamlit" alt="Streamlit"> <img src="https://img.shields.io/badge/Prophet-Forecasting-blue?logo=facebook" alt="Prophet"> <img src="https://img.shields.io/badge/Tensorflow-LSTM-red?logo=tensorflow" alt="Tensorflow"> <img src="https://img.shields.io/badge/Plotly-Visualization-brightgreen?logo=plotly" alt="Plotly"> </p>


✅ End









