import dash
from dash import dcc, html
import plotly.graph_objs as go
import yfinance as yf
import pandas as pd

app = dash.Dash(__name__)

start = '2020-01-01'
end = '2024-08-13'
eth_df = yf.download('ETH-USD', start=start, end=end)
btc_df = yf.download('BTC-USD', start=start, end=end)
aligned_df = pd.concat([eth_df['Close'], btc_df['Close']], axis=1, keys=['ETH', 'BTC']).dropna()
aligned_df['ETH/BTC Ratio'] = aligned_df['ETH'] / aligned_df['BTC']

# Create the dashboard layout
app.layout = html.Div([
    html.H1("ETH/BTC Ratio Dashboard"),
    dcc.Graph(
        id='eth-btc-ratio',
        figure={
            'data': [
                go.Scatter(
                    x=aligned_df.index,
                    y=aligned_df['ETH/BTC Ratio'],
                    mode='lines',
                    name='ETH/BTC Ratio'
                )
            ],
            'layout': go.Layout(
                title='ETH/BTC Ratio Over Time',
                xaxis={'title': 'Date'},
                yaxis={'title': 'ETH/BTC Ratio'},
            )
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
