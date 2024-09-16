import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

ticker = 'TRAK'

data = yf.download(ticker, start='2010-01-01', end='2024-09-13')

# Check for missing values
print(data.isnull().sum())

plt.figure(figsize=(14,7))
plt.plot(data['Close'])
plt.title('Closing Price of {}'.format(ticker))
plt.xlabel('Date')
plt.ylabel('Price USD ($)')
plt.show()

data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()

plt.figure(figsize=(14,7))
plt.plot(data['Close'], label='Close Price')
plt.plot(data['MA20'], label='20-Day MA')
plt.plot(data['MA50'], label='50-Day MA')
plt.title('Moving Averages of {}'.format(ticker))
plt.xlabel('Date')
plt.ylabel('Price USD ($)')
plt.legend()
plt.show()

data['Daily Return'] = data['Close'].pct_change()

plt.figure(figsize=(14,7))
sns.distplot(data['Daily Return'].dropna(), bins=50, color='purple')
plt.title('Distribution of Daily Returns')
plt.show()

df = data[['Close']]
df.dropna(inplace=True)

from sklearn.model_selection import train_test_split

# Using 80% of the data for training and 20% for testing
train_data, test_data = train_test_split(df, test_size=0.2, shuffle=False)

from statsmodels.tsa.stattools import adfuller

def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary.")
    else:
        print("Weak evidence against null hypothesis, time series is non-stationary.")

adf_test(train_data['Close'])

train_diff = train_data['Close'].diff().dropna()
adf_test(train_diff)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.figure(figsize=(14,7))
plot_acf(train_diff, lags=40)
plt.show()

plt.figure(figsize=(14,7))
plot_pacf(train_diff, lags=40)
plt.show()

from statsmodels.tsa.arima.model import ARIMA

# Define the model (p,d,q)
model = ARIMA(train_data['Close'], order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())

# Forecast
start = len(train_data)
end = len(train_data) + len(test_data) - 1
predictions = model_fit.predict(start=start, end=end, typ='levels')

# Plot the results
plt.figure(figsize=(14,7))
plt.plot(train_data['Close'], label='Training Data')
plt.plot(test_data['Close'], label='Actual Price')
plt.plot(test_data.index, predictions, label='Predicted Price', color='red')
plt.title('ARIMA Model Predictions')
plt.xlabel('Date')
plt.ylabel('Price USD ($)')
plt.legend()
plt.show()

# Calculate RMSE
from sklearn.metrics import mean_squared_error
import numpy as np

rmse_arima = np.sqrt(mean_squared_error(test_data['Close'], predictions))
print('ARIMA RMSE: {}'.format(rmse_arima))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df)

# Training data
train_data_len = int(np.ceil(len(scaled_data) * 0.8))
train_data = scaled_data[0:train_data_len, :]

# Testing data
test_data = scaled_data[train_data_len - 60:, :]

def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i - time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(train_data)
X_test, y_test = create_dataset(test_data)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=64)
