
# Re-run after kernel reset to regenerate app.py
import streamlit as st
import pandas as pd
from real_engine import analyze_stock
from visual_engine import get_ytd_comparison_chart

st.set_page_config(page_title="📈 StockMonkey Analyzer", layout="wide")
st.title("🦍 StockMonkey Stock Analyzer")

# Sidebar input
with st.sidebar:
    ticker = st.text_input("Enter Stock Ticker", value="AAPL")
    submit = st.button("Run Analysis")

# Run analysis
if submit:
    with st.spinner("Analyzing..."):
        result = analyze_stock(ticker)
        fig = get_ytd_comparison_chart(ticker)

    if not result or isinstance(result, str):
        st.error("Analysis failed or data not available.")
    else:
        # Score Summary
        st.subheader("📊 Score Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Tech Score", result.get("Tech Score", "N/A"))
        col2.metric("Fund Score", result.get("Fund Score", "N/A"))
        col3.metric("Total Score", result.get("Total Score", "N/A"))

        # Fundamentals Table
        st.subheader("🧮 Fundamentals")
        fund_keys = [
            "EPS", "EPS Growth (YoY)", "P/E", "PEG", "P/S", 
            "ROE", "Revenue Growth", "Debt-to-Equity", "FCF Yield", 
            "Forward Dividend Yield", "Trailing Dividend Yield", "Ex-Div Date"
        ]
        fund_data = {k: result[k] for k in fund_keys if k in result}
        st.table(pd.DataFrame(fund_data.items(), columns=["Metric", "Value"]))

        # Technicals Table
        st.subheader("📈 Technical Indicators")
        tech_keys = [
            "RSI", "Momentum", "Price > MA20 (%)", 
            "Open Price", "Close Price", "Price Change (%)", 
            "52W High", "Current vs 52W High (%)", 
            "Trend Signal", "Market Mode"
        ]
        tech_data = {k: result[k] for k in tech_keys if k in result}
        st.table(pd.DataFrame(tech_data.items(), columns=["Metric", "Value"]))

        # Recommendation Section
        st.subheader("🧭 Summary & Recommendation")
        rec_keys = [
            "Stock Type", "Valuation", "Strong Momentum Exception Rule", 
            "Short-Term Rec", "Long-Term Rec"
        ]
        rec_data = {k: result[k] for k in rec_keys if k in result}
        st.table(pd.DataFrame(rec_data.items(), columns=["Metric", "Value"]))

        # YTD Growth Chart
        if fig:
            st.subheader("📉 YTD Growth Comparison")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Chart data not available.")
else:
    st.info("Enter a ticker and click 'Run Analysis' to begin.")
