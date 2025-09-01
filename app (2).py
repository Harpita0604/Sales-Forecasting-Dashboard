import streamlit as st
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import os

st.title("Sales Forecasting Dashboard")

# Define the path to the zip file and extracted file
zip_file_path = "/content/sale forecasting.zip"
extracted_file_name = "stores_sales_forecasting.csv"
extracted_file_path = f"/content/{extracted_file_name}"

# Check if the zip file exists before attempting to extract
if not os.path.exists(zip_file_path):
    st.error(f"Error: The file {zip_file_path} was not found.")
else:
    # Extract the file if the zip exists
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall("/content/")

    # Load data
    try:
        data = pd.read_csv(extracted_file_path)
    except UnicodeDecodeError:
        data = pd.read_csv(extracted_file_path, encoding='latin-1')

    st.write("Data Head:", data.head())

    # Data processing
    data['Order Date'] = pd.to_datetime(data['Order Date'])
    data = data.set_index('Order Date')
    data = data.resample('MS').sum()
    data['Sales'] = data['Sales'].fillna(data['Sales'].mean())

    # Plot historical sales trends
    fig1, ax1 = plt.subplots()
    ax1.plot(data['Sales'])
    ax1.set_title("Historical Sales Trends")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Sales")
    st.pyplot(fig1)

    # ARIMA model fitting and forecasting
    # Add a check for sufficient data before fitting the model
    if len(data['Sales']) > 1:
        model = ARIMA(data['Sales'], order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=6)

        # Plot forecast
        fig2, ax2 = plt.subplots()
        ax2.plot(data['Sales'], label='Historical Sales')
        ax2.plot(forecast.index, forecast, label='Forecast', color='red')
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Sales")
        ax2.set_title("Sales Forecasting Dashboard")
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.warning("Not enough data to fit the ARIMA model and generate a forecast.")

