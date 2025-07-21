from datetime import datetime
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

def get_ytd_comparison_chart(ticker):
    start_date = f"{datetime.now().year}-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    spy = yf.Ticker("SPY").history(start=start_date, end=end_date)["Close"]
    stock = yf.Ticker(ticker).history(start=start_date, end=end_date)["Close"]

    if spy.empty or stock.empty:
        return None

    spy_pct = (spy / spy.iloc[0] - 1) * 100
    stock_pct = (stock / stock.iloc[0] - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spy_pct.index, y=spy_pct.values, name="S&P 500", line=dict(color='gray')))
    fig.add_trace(go.Scatter(x=stock_pct.index, y=stock_pct.values, name=ticker.upper(), line=dict(color='blue')))

    fig.update_layout(
        title=f"{ticker.upper()} vs S&P 500 - YTD Growth",
        xaxis_title="Date",
        yaxis_title="YTD Growth (%)",
        template="plotly_white",
        height=450
    )
    return fig
