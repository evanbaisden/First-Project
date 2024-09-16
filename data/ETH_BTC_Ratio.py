import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ETH_BTC_RATIO = "ETH/BTC Ratio"

def yfinance_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch financial data from Yahoo Finance.
    
    Args:
        ticker (str): The stock ticker symbol
        start (str): Start date in 'YYYY-MM-DD' format
        end (str): End date in 'YYYY-MM-DD' format
    
    Returns:
        pd.DataFrame: DataFrame containing the financial data
    """
    try:
        return yf.download(ticker, start=start, end=end)
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def calculate_eth_btc_ratio(eth_df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the ETH/BTC ratio.
    
    Args:
        eth_df (pd.DataFrame): DataFrame containing ETH data
        btc_df (pd.DataFrame): DataFrame containing BTC data
    
    Returns:
        pd.DataFrame: DataFrame containing the ETH/BTC ratio
    """
    ratio_df = pd.concat([eth_df['Close'], btc_df['Close']], axis=1, keys=['ETH', 'BTC']).dropna()
    ratio_df[ETH_BTC_RATIO] = ratio_df['ETH'] / ratio_df['BTC']
    return ratio_df

def plot_eth_btc_ratio(ratio_df: pd.DataFrame, save_path: str = None):
    """
    Plot the ETH/BTC ratio over time.
    
    Args:
        ratio_df (pd.DataFrame): DataFrame containing the ETH/BTC ratio
        save_path (str, optional): Path to save the plot. If None, the plot will be displayed.
    """
    plt.figure(figsize=(12, 6))
    sns.set_style("darkgrid")
    sns.lineplot(data=ratio_df, x=ratio_df.index, y=ETH_BTC_RATIO, color="purple")

    plt.xlabel("Date")
    plt.ylabel(ETH_BTC_RATIO)
    plt.title(f"{ETH_BTC_RATIO} Over Time")

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Plot saved to {save_path}")
    else:
        plt.show()

def main():
    start_date = "2015-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

    logging.info("Fetching ETH data...")
    eth_df = yfinance_data("ETH-USD", start=start_date, end=end_date)
    
    logging.info("Fetching BTC data...")
    btc_df = yfinance_data("BTC-USD", start=start_date, end=end_date)

    if eth_df.empty or btc_df.empty:
        logging.error("Failed to fetch data. Exiting.")
        return

    logging.info("Calculating ETH/BTC ratio...")
    eth_btc_ratio_df = calculate_eth_btc_ratio(eth_df, btc_df)

    logging.info("Plotting ETH/BTC ratio...")
    plot_eth_btc_ratio(eth_btc_ratio_df, save_path="eth_btc_ratio.png")

if __name__ == "__main__":
    main()
