import pandas as pd
import yfinance as yf
from datetime import datetime
from vizro import Vizro
import vizro.models as vm
import vizro.plotly.express as px


def fetch_data(ticker, start, end):
    try:
        return yf.download(ticker, start=start, end=end)
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error


def calculate_ratio(eth_df, btc_df):
    aligned_df = pd.concat(
        [eth_df["Close"], btc_df["Close"]], axis=1, keys=["ETH", "BTC"]
    ).dropna()
    aligned_df["ETH/BTC Ratio"] = aligned_df["ETH"] / aligned_df["BTC"]
    return aligned_df


def create_chart(data):
    return px.line(
        data, x=data.index, y="ETH/BTC Ratio", title="ETH/BTC Ratio Over Time"
    )


def create_dashboard(chart):
    graph_component = vm.Graph(figure=chart)
    page = vm.Page(
        title="ETH/BTC Ratio Dashboard", components=[graph_component], controls=[]
    )
    return Vizro().build(vm.Dashboard(pages=[page]))


def main():
    start_date = "2020-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")
    eth_ticker = "ETH-USD"
    btc_ticker = "BTC-USD"

    eth_df = fetch_data(eth_ticker, start=start_date, end=end_date)
    btc_df = fetch_data(btc_ticker, start=start_date, end=end_date)

    if eth_df.empty or btc_df.empty:
        print("Data fetch failed. Exiting.")
        return

    ratio_df = calculate_ratio(eth_df, btc_df)

    chart = create_chart(ratio_df)

    dashboard = create_dashboard(chart)
    dashboard.run()


main()
