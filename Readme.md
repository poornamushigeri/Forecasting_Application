# 📈 Forecasting App

A powerful **Streamlit application** for **Time Series Forecasting** across various domains — sales, finance, energy, weather, and more.

---

## 🚀 Features

- Forecast with **Prophet**, **ARIMA**, and **LSTM** models  
- Compare multiple models side by side  
- Multivariate forecasting with external regressors  
- Anomaly detection using Isolation Forest  
- Automatic data preprocessing and resampling  
- Exploratory Data Analysis (Fast EDA, Sweetviz, YData Profiling)  
- Prophet cross-validation and evaluation metrics (MAE, RMSE, MAPE)  
- Hyperparameter tuning for Prophet  
- Model explainability via feature importance

---

## 🛠️ Technologies Used

- Streamlit
- Prophet
- ARIMA (statsmodels)
- LSTM (TensorFlow/Keras)
- Sweetviz
- YData Profiling
- Scikit-learn
- Plotly

---

## 📂 Project Structure

```
forecasting_app/
│
├── app.py                  # Main Streamlit App
├── requirements.txt        # Python dependencies
│
└── utils/
    ├── preprocessing.py    # Data cleaning and resampling
    ├── eda.py               # Exploratory Data Analysis
    ├── anomaly.py           # Anomaly detection
    ├── forecasting.py       # Forecasting models (Prophet, ARIMA, LSTM)
    ├── explainability.py    # Feature importance calculation
    └── tuning.py            # Hyperparameter tuning
```

---

## 📦 Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/poornamushigeri/Forecasting_Application.git
    cd Forecasting_Application
    ```

2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run forecasting_app/app.py
    ```

---

## 🖥️ Usage Instructions

- Upload your CSV file  
- Select Date and Target columns  
- Perform EDA (Built-in EDA, Sweetviz, or YData Profiling)  
- Detect and remove anomalies (optional)  
- Train Prophet, ARIMA, and LSTM models  
- Perform Prophet cross-validation and hyperparameter tuning  
- Visualize forecast results  
- Analyze feature importance (if regressors are used)

---

## 📸 App Screenshot

<p align="center">
  <img src="forecasting_app/screenshots/app_scr.jpg" width="700">
</p>

---

## 📋 Example Datasets

- Monthly Airline Passengers  
- Stock Market Data  
- Electricity Demand  
- Retail Sales Data  
- Weather Time Series  

---

## 📛 Badges

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-Application-orange?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Prophet-Forecasting-blue?logo=facebook" alt="Prophet">
  <img src="https://img.shields.io/badge/Tensorflow-LSTM-red?logo=tensorflow" alt="Tensorflow">
  <img src="https://img.shields.io/badge/Plotly-Visualization-brightgreen?logo=plotly" alt="Plotly">
</p>

---

# ✅ End
