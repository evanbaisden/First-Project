import yfinance as yf
import pandas as pd
import numpy as np

# Fetch Bitcoin data
btc = yf.Ticker("BTC-USD")
df = btc.history(period="2y")

# Calculate daily returns
df['Returns'] = df['Close'].pct_change()

# Calculate 30-day rolling volatility
df['Volatility'] = df['Returns'].rolling(window=30).std() * np.sqrt(252)

print(df[['Close', 'Volatility']].tail())

# You can now use this data to build a prediction model