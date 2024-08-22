import yfinance as yf
import pandas as pd

start = '2024-01-01'
end = '2024-08-13'

eth_df = yf.download('ETH-USD', start=start, end=end)
btc_df = yf.download('BTC-USD', start=start, end=end)

aligned_df = pd.concat([eth_df['Close'], btc_df['Close']], axis=1, keys=['ETH', 'BTC']).dropna()

# Calculate the ratio of 1 ETH to 1 BTC
aligned_df['ETH/BTC Ratio'] = aligned_df['ETH'] / aligned_df['BTC']

print(aligned_df.head())
