import pandas as pd
import yfinance as yf
from datetime import datetime
from fredapi import Fred
import matplotlib.pyplot as plt

# Constants
SP500 = "S&P 500"
M2 = "M2 (Billions)"

# FRED API Key
fred_api_key = "ecc0c4e92352e1eff5869528238a3bff"

def yfinance_data(ticker, start, end):
    try:
        return yf.download(ticker, start=start, end=end)
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

def fred_data(series_id):
    fred = Fred(api_key=fred_api_key)
    try:
        data = fred.get_series(series_id)
        return pd.DataFrame(data, columns=[M2])
    except Exception as e:
        print(f"Error fetching data from FRED: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

# Main execution
start_date = "1990-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")

# Fetch financial data
sp500_df = yfinance_data("^GSPC", start=start_date, end=end_date)

# Fetch and process M2 data
m2_df = fred_data("M2SL")
m2_df.index = m2_df.index - pd.offsets.MonthEnd(1)

# Resample S&P 500 data to monthly frequency, keeping the last available value of each month
sp500_df = sp500_df.resample("M").last()

# Merge M2 and S&P 500 data, dropping NaNs early to avoid unnecessary operations
m2_sp500_df = pd.concat([m2_df, sp500_df["Close"]], axis=1).dropna()
m2_sp500_df.columns = [M2, SP500]

# Plotting with matplotlib
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot M2 on the left y-axis
ax1.plot(m2_sp500_df.index, m2_sp500_df[M2], color="blue", label=M2)
ax1.set_xlabel("Date")
ax1.set_ylabel(M2, color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

# Create a second y-axis for S&P 500
ax2 = ax1.twinx()
ax2.plot(m2_sp500_df.index, m2_sp500_df[SP500], color="green", label=SP500)
ax2.set_ylabel(SP500, color="green")
ax2.tick_params(axis="y", labelcolor="green")

# Title and show plot
plt.title(f"{SP500} vs {M2}")
fig.tight_layout()
plt.show()
