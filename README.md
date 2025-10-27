# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\sem 3\time series\AirPassengers.csv")
data['Month'] = pd.to_datetime(data['Month'], format='%Y-%m')
data.set_index('Month', inplace=True)
data.rename(columns={'#Passengers': 'passengers'}, inplace=True)

# Visualize the time series data
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['passengers'])
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.title('Air Passenger Time Series')
plt.show()

# Check for stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

check_stationarity(data['passengers'])

# ACF and PACF plots
plot_acf(data['passengers'])
plt.show()
plot_pacf(data['passengers'])
plt.show()

# Split the data into train and test sets (80% train, 20% test)
train_size = int(len(data) * 0.8)
train, test = data['passengers'][:train_size], data['passengers'][train_size:]

# SARIMA model configuration
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Forecasting
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate RMSE
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.title('SARIMA Model Forecasting for Air Passengers')
plt.legend()
plt.show()
```
### OUTPUT:

<img width="1295" height="667" alt="Screenshot 2025-10-27 153535" src="https://github.com/user-attachments/assets/968c5ddf-74b1-4246-9572-8e73fe5ac8be" />

<img width="487" height="165" alt="Screenshot 2025-10-27 153541" src="https://github.com/user-attachments/assets/3f3d8d50-6ef5-4feb-9d60-f5089cedaf11" />

<img width="915" height="552" alt="Screenshot 2025-10-27 153548" src="https://github.com/user-attachments/assets/428089dc-081e-4ff7-b4b4-8ab31d1b5bc9" />

<img width="885" height="541" alt="Screenshot 2025-10-27 153557" src="https://github.com/user-attachments/assets/56a39258-6cc5-4659-91e9-44397b2f40e4" />

<img width="1176" height="687" alt="Screenshot 2025-10-27 153609" src="https://github.com/user-attachments/assets/80744ee6-32a3-431b-b387-87697736dc8c" />


### RESULT:
Thus the program run successfully based on the SARIMA model.
