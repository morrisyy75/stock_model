
# Re-run after kernel reset to regenerate app.py
import streamlit as st
import pandas as pd
from real_engine import analyze_stock
from visual_engine import get_ytd_comparison_chart

st.set_page_config(page_title="📈 StockMonkey Analyzer", layout="wide")

# 🔷 Head bar with logo and ticker input
col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("## 🦍 StockMonkey Analyzer")
with col2:
    with st.form(key="ticker_form"):
        ticker = st.text_input("Enter Ticker", value="AAPL")
        submit = st.form_submit_button("Run Analysis")

# 🔷 Sidebar navigation
section = st.sidebar.radio("📂 View Section", ["📊 Fundamentals", "📈 Technicals"])

# 🔷 Run analysis and display output
if submit:
    with st.spinner("Analyzing..."):
        result = analyze_stock(ticker)
        fig = get_ytd_comparison_chart(ticker)

    if not result or isinstance(result, str):
        st.error("Analysis failed or data not available.")
    else:
        # 🔸 Live Trading Chart
        st.subheader("📺 Live Trading Chart")
        tradingview_code = f"""
<div class="tradingview-widget-container" style="margin-top: 20px;">
  <div id="tradingview_chart" style="height:500px;"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
    new TradingView.widget({{
      "autosize": true,
      "symbol": "{ticker.upper()}",
      "interval": "D",
      "timezone": "Etc/UTC",
      "theme": "dark",
      "style": "1",
      "locale": "en",
      "toolbar_bg": "#f1f3f6",
      "enable_publishing": false,
      "allow_symbol_change": false,
      "container_id": "tradingview_chart"
    }});
  </script>
</div>
"""
        st.components.v1.html(tradingview_code, height=520)

        # 🔸 Score Summary
        st.subheader("📊 Score Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Technical Score", result.get("Tech Score", "N/A"))
        col2.metric("Fundamental Score", result.get("Fund Score", "N/A"))
        col3.metric("Overall Score", result.get("Total Score", "N/A"))

        # 🔸 Recommendation
        st.subheader("🧭 Recommendation Summary")
        rec_keys = [
            "Stock Type", "Valuation", "Strong Momentum Exception Rule", 
            "Short-Term Rec", "Long-Term Rec"
        ]
        rec_data = {k: result[k] for k in rec_keys if k in result}
        st.table(pd.DataFrame(rec_data.items(), columns=["Metric", "Value"]))

        ## YTD Growth Chart
         if fig:
            st.subheader("📉 YTD Growth Comparison")
            st.plotly_chart(fig, use_container_width=True)
         else:
            st.warning("Chart data not available.")

        # 🔸 Sidebar-controlled Details
        if section == "📊 Fundamentals":
            fund_keys = [
                "EPS", "EPS Growth (YoY)", "P/E", "PEG", "P/S", 
                "ROE", "Revenue Growth", "Debt-to-Equity", "FCF Yield", 
                "Forward Dividend Yield", "Trailing Dividend Yield", "Ex-Div Date"
            ]
            fund_data = {k: result[k] for k in fund_keys if k in result}
            st.subheader("📊 Fundamentals")
            st.table(pd.DataFrame(fund_data.items(), columns=["Metric", "Value"]))

        elif section == "📈 Technicals":
            tech_keys = [
                "RSI", "Momentum", "Price > MA20 (%)", 
                "Open Price", "Close Price", "Price Change (%)", 
                "52W High", "Current vs 52W High (%)", 
                "Trend Signal", "Market Mode"
            ]
            tech_data = {k: result[k] for k in tech_keys if k in result}
            st.subheader("📈 Technical Indicators")
            st.table(pd.DataFrame(tech_data.items(), columns=["Metric", "Value"]))

# Default prompt before any submission
else:
    st.info("Enter a ticker and click 'Run Analysis' to begin.")
