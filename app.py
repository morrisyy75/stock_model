
import streamlit as st
from real_engine import analyze_stock

st.set_page_config(page_title="📈 Stock Forecast", layout="wide")
st.title("🦍 StockMonkey Stock Analyzer")

with st.sidebar:
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL")
    period = st.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "2y"])
    submit = st.button("Run Analysis")

if submit:
    with st.spinner("Analyzing..."):
        result = analyze_stock(ticker, period)

    if not result or isinstance(result, str):
        st.error("Analysis failed or no data.")
    else:
        st.success("Analysis Complete ✅")
        st.subheader("📊 Summary:")
        for key, val in result.items():
            st.markdown(f"**{key}:** {val}")
else:
    st.info("Enter a ticker and click 'Run Analysis' to begin.")
