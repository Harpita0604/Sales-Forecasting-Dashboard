
import pandas as pd
import zipfile

# Load data
zip_file_path = "/content/sale forecasting.zip"


with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall("/content/")


extracted_file_name = "stores_sales_forecasting.csv" # Replace with the actual extracted file name
extracted_file_path = f"/content/{extracted_file_name}"

try:
    data = pd.read_csv(extracted_file_path)
    print(data.head())
except UnicodeDecodeError:
    print(f"UnicodeDecodeError: Could not read {extracted_file_name} with default encoding. Trying with 'latin-1' encoding.")
    data = pd.read_csv(extracted_file_path, encoding='latin-1')
    print(data.head())

import matplotlib.pyplot as plt
data.plot(title="Historical Sales Trends")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()

data=data.asfreq('MS')
data['Sales']=data['Sales'].fillna(data['Sales'].mean())

from statsmodels.tsa.arima.model import ARIMA
model=ARIMA(data['Sales'],order=(1,1,1))
model_fit=model.fit()

forecast=model_fit.forecast(steps=6)
print(forecast)
plt.plot(data['Sales'],label='Historical Sales')
plt.plot(forecast.index,forecast,label='Forecast',color='red')
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Sales Forecasting Dashboard")
plt.legend()
plt.show()
