# Install required packages
#Import libraries
import yfinance as yf
import pandas as pd
import numpy as np
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from tabulate import tabulate
import traceback
from tqdm import tqdm
import time
from datetime import datetime
from typing import Literal

nltk.download('vader_lexicon')

# Sector mappings
sector_etf_map = {
    'Technology': 'QQQ', 'Financial Services': 'XLF', 'Healthcare': 'XLV',
    'Consumer Cyclical': 'XLY', 'Consumer Defensive': 'XLP', 'Energy': 'XLE',
    'Industrials': 'XLI', 'Basic Materials': 'XLB', 'Communication Services': 'XLC',
    'Utilities': 'XLU', 'Real Estate': 'XLRE'
}

# Fetch S&P 500 tickers and sectors from Wikipedia
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_table = tables[0]
        tickers_sectors = sp500_table[['Symbol', 'GICS Sector']].copy()
        tickers_sectors.columns = ['Ticker', 'Sector']
        tickers_sectors['Ticker'] = tickers_sectors['Ticker'].str.replace('.', '-', regex=False)
        return tickers_sectors
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        return pd.DataFrame(columns=['Ticker', 'Sector'])

# Stock analysis functions
def classify_stock_type(info):
    growth = info.get('earningsGrowth') or 0
    rev_growth = info.get('revenueGrowth') or 0
    pe = info.get('trailingPE') or 0
    pb = info.get('priceToBook') or 0
    gross_margin = info.get('grossMargins') or 0

    # Speculative scoring logic
    speculative_score = 0
    speculative_score += pe == 0 or pe > 100
    speculative_score += growth < 0
    speculative_score += rev_growth < 0
    if speculative_score >= 2:
        return 'speculative'

    # Value stock logic (scoring)
    value_score = 0
    value_score += pe < 20
    value_score += pb < 2
    value_score += growth < 0.10
    value_score += rev_growth < 0.10
    if value_score >= 3:
        return 'value'

    # High growth stock logic (scoring)
    growth_score = 0
    growth_score += growth > 0.12
    growth_score += rev_growth > 0.12
    growth_score += gross_margin > 0.25
    growth_score += pe < 80
    if growth_score >= 3:
        return 'high_growth'

    return 'stable'

def fundamental_score(ticker, stock_info, history):
    if not ticker or not isinstance(ticker, str):
        raise ValueError("Invalid ticker provided")

    try:
        stock_type = classify_stock_type(stock_info)
        weights_by_type = {
    'high_growth': {'valuation': 0.15, 'rev_growth': 0.4, 'eps_growth': 0.15, 'fcf_yield': 0.1, 'debt_eq': 0.1, 'roe': 0.1},
    'value': {'valuation': 0.3, 'fcf_yield': 0.25, 'debt_eq': 0.2, 'roe': 0.25},
    'speculative': {'rev_growth': 0.5, 'valuation': 0.15, 'fcf_yield': 0.15},
    'stable': {'valuation': 0.25, 'rev_growth': 0.2, 'eps_growth': 0.1, 'fcf_yield': 0.2, 'roe': 0.15, 'debt_eq': 0.1}
        }

        weights = weights_by_type.get(stock_type, weights_by_type['stable'])
        score = 0
        indicators = {}

        # Valuation score
        val_score, _, val_details = valuation_score(ticker, stock_info)
        score += val_score * weights.get('valuation', 0)
        indicators['pe_ratio'] = val_details.get('pe', np.nan)
        indicators['ps_ratio'] = val_details.get('ps', np.nan)
        indicators['peg_ratio'] = val_details.get('peg', np.nan)

        # EPS and EPS Growth
        eps = stock_info.get('trailingEps')
        eps_growth = stock_info.get('earningsGrowth')
        indicators['eps'] = eps if eps is not None else np.nan
        indicators['eps_growth'] = eps_growth if eps_growth is not None else np.nan

        if eps_growth is not None:
            if eps_growth > 0.30: score += 100 * weights.get('eps_growth', 0)
            elif eps_growth > 0.15: score += 80 * weights.get('eps_growth', 0)
            elif eps_growth > 0.05: score += 60 * weights.get('eps_growth', 0)
            elif eps_growth >= 0: score += 40 * weights.get('eps_growth', 0)
            else: score += 20 * weights.get('eps_growth', 0)


        # Return on Equity
        roe = stock_info.get('returnOnEquity')
        indicators['roe'] = roe if roe is not None else np.nan
        if roe is not None and roe >= 0:
            if roe > 0.25: score += 100 * weights.get('roe', 0)
            elif roe > 0.15: score += 80 * weights.get('roe', 0)
            elif roe > 0.05: score += 60 * weights.get('roe', 0)
            else: score += 40 * weights.get('roe', 0)

        # Revenue Growth
        rev_growth = stock_info.get('revenueGrowth')
        indicators['revenue_growth'] = rev_growth if rev_growth is not None else np.nan
        if rev_growth is not None:
            if rev_growth > 0.30: score += 100 * weights.get('rev_growth', 0)
            elif rev_growth > 0.15: score += 80 * weights.get('rev_growth', 0)
            elif rev_growth > 0: score += 60 * weights.get('rev_growth', 0)
            else: score += 40 * weights.get('rev_growth', 0)

        # Debt-to-Equity
        debt_eq = stock_info.get('debtToEquity')
        if debt_eq is not None:
            debt_eq *= 0.01  # Convert percentage to decimal
        indicators['debt_to_equity'] = debt_eq if debt_eq is not None else np.nan
        if debt_eq is not None and debt_eq >= 0:
            if debt_eq < 0.5: score += 100 * weights.get('debt_eq', 0)
            elif debt_eq < 1.0: score += 80 * weights.get('debt_eq', 0)
            elif debt_eq < 2.0: score += 60 * weights.get('debt_eq', 0)
            else: score += 40 * weights.get('debt_eq', 0)

        # Free Cash Flow Yield
        fcf = stock_info.get('freeCashflow')
        market_cap = stock_info.get('marketCap')
        indicators['fcf_yield'] = np.nan
        if fcf and market_cap and fcf > 0:
            if fcf < market_cap * 0.05:
                fcf *= 4
            fcf_yield = fcf / market_cap
            indicators['fcf_yield'] = fcf_yield
            if fcf_yield > 0.08: score += 100 * weights.get('fcf_yield', 0)
            elif fcf_yield > 0.04: score += 80 * weights.get('fcf_yield', 0)
            elif fcf_yield > 0.01: score += 60 * weights.get('fcf_yield', 0)
            else: score += 40 * weights.get('fcf_yield', 0)

        return round(score, 2), indicators

    except Exception as e:
        print(f"Error fetching fundamental score for {ticker}: {e}")
        return 0, {}
def technical_score(ticker, history, period):
    try:
        required_columns = ['Close', 'High', 'Low', 'Volume']
        if not isinstance(history, pd.DataFrame) or not all(col in history.columns for col in required_columns):
            print(f"Error: Invalid DataFrame for {ticker}")
            return 0, history
        if len(history) < 50:
            print(f"Insufficient data ({len(history)} rows) for {ticker}. Minimum 50 required.")
            return 20, history

        score = 0
        weights = {'ma_crossover': 0.25, 'rsi': 0.25, 'macd': 0.25, 'volume_trend': 0.15, 'bb_adx': 0.1}

        stock = yf.Ticker(ticker)
        sector = stock.info.get('sector', 'Unknown')
        benchmark_ticker = sector_etf_map.get(sector, 'SPY')
        benchmark_history = yf.Ticker(benchmark_ticker).history(period=period)
        if benchmark_history.empty or len(benchmark_history) < 50:
            benchmark_history = history
            print(f"Using stock data as benchmark for {ticker}")

        history = history.sort_index()
        benchmark_history = benchmark_history.sort_index()

        # Moving Average Crossover
        history['MA20'] = history['Close'].rolling(window=20).mean()
        history['MA50'] = history['Close'].rolling(window=50).mean()
        latest_close = history['Close'].iloc[-1].item()
        latest_ma20 = history['MA20'].iloc[-1].item()
        latest_ma50 = history['MA50'].iloc[-1].item()

        if pd.isna(latest_ma20) or pd.isna(latest_ma50):
            score_ma = 0
        elif latest_close > 1.05 * latest_ma20 and latest_ma20 > latest_ma50:
            score_ma = 20 * weights['ma_crossover']
        elif latest_close > latest_ma20:
            score_ma = 15 * weights['ma_crossover']
        else:
            score_ma = 10 * weights['ma_crossover']
        score += score_ma

        # RSI
        def calculate_rsi(data, periods=14):
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=periods, min_periods=1).mean()
            avg_loss = loss.rolling(window=periods, min_periods=1).mean()
            avg_loss = avg_loss.replace(0, 1e-10)
            rs = avg_gain / avg_loss
            rsi_series = 100 - (100 / (1 + rs))
            return rsi_series.iloc[-1].item()

        rsi_val = calculate_rsi(history)
        benchmark_rsi_val = calculate_rsi(benchmark_history)
        history['RSI'] = calculate_rsi(history)
        is_uptrend = history['Close'].iloc[-1].item() > history['Close'].iloc[-3].item() if len(history) >= 3 else False

        if pd.isna(rsi_val):
            score_rsi = 0
        elif rsi_val < 30 and is_uptrend:
            score_rsi = 20 * weights['rsi']
        elif rsi_val > 70 and (pd.isna(benchmark_rsi_val) or rsi_val <= benchmark_rsi_val):
            score_rsi = 15 * weights['rsi']
        elif 30 <= rsi_val <= 70:
            score_rsi = 15 * weights['rsi']
        else:
            score_rsi = 5 * weights['rsi']
        score += score_rsi

        # MACD
        history['EMA12'] = history['Close'].ewm(span=12, adjust=False).mean()
        history['EMA26'] = history['Close'].ewm(span=26, adjust=False).mean()
        history['MACD'] = history['EMA12'] - history['EMA26']
        history['Signal'] = history['MACD'].ewm(span=9, adjust=False).mean()
        history['MACD_Hist'] = history['MACD'] - history['Signal']
        benchmark_history['EMA12'] = benchmark_history['Close'].ewm(span=12, adjust=False).mean()
        benchmark_history['EMA26'] = benchmark_history['Close'].ewm(span=26, adjust=False).mean()
        benchmark_history['MACD'] = benchmark_history['EMA12'] - benchmark_history['EMA26']

        if len(history) < 26:
            score_macd = 0
        else:
            macd_cross = history['MACD'].iloc[-2] < history['Signal'].iloc[-2] and history['MACD'].iloc[-1] > history['Signal'].iloc[-1]
            hist_positive = history['MACD_Hist'].iloc[-1] > 0
            hist_negative = history['MACD_Hist'].iloc[-1] < 0
            benchmark_macd = benchmark_history['MACD'].iloc[-1]

            if macd_cross and hist_positive:
                score_macd = 20 * weights['macd']
            elif macd_cross or (hist_positive and history['MACD'].iloc[-1] > benchmark_macd):
                score_macd = 15 * weights['macd']
            elif not hist_negative:
                score_macd = 10 * weights['macd']
            else:
                score_macd = 5 * weights['macd']
        score += score_macd

        # Volume Trend
        history['Vol_MA20'] = history['Volume'].rolling(window=20).mean()
        benchmark_history['Vol_MA20'] = benchmark_history['Volume'].rolling(window=20).mean()
        latest_volume = history['Volume'].iloc[-1]
        latest_vol_ma20 = history['Vol_MA20'].iloc[-1]
        benchmark_volume = benchmark_history['Volume'].iloc[-1]
        benchmark_vol_ma20 = benchmark_history['Vol_MA20'].iloc[-1]

        volume_ratio = latest_volume / latest_vol_ma20 if latest_vol_ma20 != 0 else 1
        benchmark_volume_ratio = benchmark_volume / benchmark_vol_ma20 if benchmark_vol_ma20 != 0 else 1
        price_up_today = history['Close'].iloc[-1] > history['Close'].iloc[-2]

        if volume_ratio > 1.5 and price_up_today and volume_ratio > benchmark_volume_ratio:
            score_volume = 20 * weights['volume_trend']
        elif volume_ratio > benchmark_volume_ratio or (volume_ratio > 1.0 and price_up_today):
            score_volume = 15 * weights['volume_trend']
        elif volume_ratio < 0.7:
            score_volume = 5 * weights['volume_trend']
        else:
            score_volume = 10 * weights['volume_trend']
        score += score_volume

        # Bollinger Bands + ADX
        history['stddev'] = history['Close'].rolling(window=20).std()
        history['UpperBB'] = history['MA20'] + (2 * history['stddev'])
        history['LowerBB'] = history['MA20'] - (2 * history['stddev'])
        history['BB_Width'] = history['UpperBB'] - history['LowerBB']
        benchmark_history['stddev'] = benchmark_history['Close'].rolling(window=20).std()
        benchmark_history['BB_Width'] = benchmark_history['stddev'] * 4

        latest_bb_width = history['BB_Width'].iloc[-1]
        bb_exp = latest_bb_width > history['BB_Width'].iloc[-5] if len(history) >= 5 else False
        benchmark_width = benchmark_history['BB_Width'].iloc[-1] if not pd.isna(benchmark_history['BB_Width'].iloc[-1]) else latest_bb_width
        near_upper = history['Close'].iloc[-1] > 0.95 * history['UpperBB'].iloc[-1]

        high = history['High']
        low = history['Low']
        close = history['Close']
        history['TR'] = np.maximum((high - low), np.maximum(abs(high - close.shift()), abs(low - close)))
        history['ATR'] = history['TR'].ewm(span=14, adjust=False).mean()
        history['UpMove'] = high.diff()
        history['DownMove'] = low.diff().abs()
        history['+DM'] = np.where((history['UpMove'] > history['DownMove']) & (history['UpMove'] > 0), history['UpMove'], 0)
        history['-DM'] = np.where((history['DownMove'] > history['UpMove']) & (history['DownMove'] > 0), history['DownMove'], 0)
        plus_dm_avg = pd.Series(history['+DM']).ewm(span=14, adjust=False).mean()
        minus_dm_avg = pd.Series(history['-DM']).ewm(span=14, adjust=False).mean()
        avg_atr = history['ATR']
        history['+DI'] = 100 * (plus_dm_avg / avg_atr.replace(0, np.nan))
        history['-DI'] = 100 * (minus_dm_avg / avg_atr.replace(0, np.nan))
        sum_di = history['+DI'] + history['-DI']
        history['DX'] = (abs(history['+DI'] - history['-DI']) / sum_di.replace(0, np.nan)) * 100
        history['ADX'] = history['DX'].ewm(span=14, adjust=False).mean()
        adx_trend = history['ADX'].iloc[-1] > 25 if not pd.isna(history['ADX'].iloc[-1]) else False

        if near_upper and bb_exp and adx_trend and latest_bb_width > benchmark_width:
            score_bb = 20 * weights['bb_adx']
        elif bb_exp or adx_trend:
            score_bb = 15 * weights['bb_adx']
        else:
            score_bb = 10 * weights['bb_adx']
        score += score_bb

        raw_max_score = 20 * sum(weights.values())
        normalized_score = (score / raw_max_score) * 100 if raw_max_score != 0 else 0
        return round(normalized_score, 2), history

    except Exception as e:
        print(f"❌ Error in technical_score({ticker}): {e}")
        traceback.print_exc()
        return 0, history

def news_sentiment_score(ticker):
    url = f"https://news.google.com/search?q={ticker}%20stock&hl=en-US&gl=US&ceid=US:en"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('article')[:10]
    except requests.RequestException as e:
        print(f"Error fetching news for {ticker}: {e}")
        return 50

    if len(articles) < 3:
        print(f"Fewer than 3 articles for {ticker}, returning neutral score.")
        return 50

    analyzer = SentimentIntensityAnalyzer()
    headlines = [article.get_text() for article in articles if article.get_text()]
    if not headlines:
        return 50

    sentiment_scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
    avg_sentiment = np.mean(sentiment_scores)

    if avg_sentiment > 0.5: score_sent = 20
    elif avg_sentiment > 0.2: score_sent = 15
    elif avg_sentiment > -0.2: score_sent = 10
    else: score_sent = 5

    volume = len(headlines)
    if volume >= 8: score_vol = 20
    elif volume >= 5: score_vol = 15
    else: score_vol = 10

    recent_scores = sentiment_scores[:5]
    older_scores = sentiment_scores[5:] if len(sentiment_scores) > 5 else [0]
    trend = np.mean(recent_scores) - np.mean(older_scores)
    if trend > 0.2: score_trend = 20
    elif trend > 0: score_trend = 15
    elif trend > -0.2: score_trend = 10
    else: score_trend = 5

    pos_keywords = ["growth", "strong", "beat", "record", "upgraded", "positive", "buy", "expansion"]
    neg_keywords = ["lawsuit", "scandal", "miss", "downgraded", "negative", "sell", "fraud", "decline"]
    pos_hits = sum(any(word in h.lower() for word in pos_keywords) for h in headlines)
    neg_hits = sum(any(word in h.lower() for word in neg_keywords) for h in headlines)
    total_hits = pos_hits + neg_hits
    if total_hits == 0: score_kw = 10
    else:
        pos_ratio = pos_hits / total_hits
        if pos_ratio > 0.7: score_kw = 20
        elif pos_ratio > 0.5: score_kw = 15
        elif pos_ratio > 0.3: score_kw = 10
        else: score_kw = 5

    positive_event = any(word in h.lower() for h in headlines for word in ["earnings beat", "merger", "acquisition", "partnership"])
    negative_event = any(word in h.lower() for h in headlines for word in ["regulation", "fraud", "investigation", "lawsuit"])
    if positive_event and not negative_event: score_event = 20
    elif negative_event and not positive_event: score_event = 5
    elif positive_event and negative_event: score_event = 10
    else: score_event = 15

    total_score = (score_sent + score_vol + score_trend + score_kw + score_event) / 100 * 100
    return round(total_score, 2)

def valuation_score(ticker, stock_info):
    try:
        pe = stock_info.get('trailingPE', np.nan)
        ps = stock_info.get('priceToSalesTrailing12Months', np.nan)
        peg = stock_info.get('pegRatio', np.nan)
        sector = stock_info.get('sector', 'Technology')
        earnings_growth = stock_info.get('earningsGrowth', np.nan)

        sector_median_pe = 40 if sector == 'Technology' else 35 if sector == 'Consumer Cyclical' else 20
        sector_median_ps = 15 if sector == 'Technology' else 8 if sector == 'Consumer Cyclical' else 5

        if pd.isna(peg) and not pd.isna(pe) and not pd.isna(earnings_growth) and earnings_growth > 0:
            peg = pe / (earnings_growth * 100)
        if peg and 0 < peg < 10:
            if peg < 0.3: peg_score = 85
            elif peg < 0.6: peg_score = 75
            elif peg < 1.0: peg_score = 65
            elif peg < 1.5: peg_score = 55
            elif peg < 2.5: peg_score = 45
            elif peg < 4: peg_score = 35
            else: peg_score = 25
        else:
            peg_score = 50

        if pe and not np.isnan(pe):
            pe_score = 100 - min(50 * np.log1p(pe / sector_median_pe), 100)
        else:
            pe_score = 50
        if ps and not np.isnan(ps):
            ps_score = 100 - min(50 * np.log1p(ps / sector_median_ps), 100)
        else:
            ps_score = 50

        sector_score = (pe_score + ps_score) / 2
        final_score = 0.6 * peg_score + 0.4 * sector_score

        if final_score >= 70: label = "Undervalued"
        elif final_score <= 40: label = "Overvalued"
        else: label = "Neutral"

        return round(final_score, 2), label, {"peg": peg, "pe": pe, "ps": ps}
    except Exception as e:
        print(f"Error fetching valuation score for {ticker}: {e}")
        return 50, "Neutral", {"peg": None, "pe": None, "ps": None}

def get_precise_dividend_yield(ticker):
    try:
        stock = yf.Ticker(ticker)
        dividends = stock.dividends
        if dividends.empty:
            return "N/A"
        recent_dividends = dividends[-4:]  # assume quarterly dividend
        ttm_div = recent_dividends.sum()
        current_price = stock.history(period="1d")['Close'][-1]
        yield_percent = ttm_div / current_price * 100
        return round(yield_percent, 2)
    except Exception as e:
        print(f"Error fetching precise dividend yield for {ticker}: {e}")
        return "N/A"




def is_overbought(ticker, history, current_price=None):
    try:
        if history.empty or len(history) < 20:
            print(f"⚠️ Insufficient history data for {ticker} in is_overbought")
            return False, {'rsi': 50, 'price_above_ma20': 0, 'near_all_time_high': False}

        delta = history['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        avg_loss = avg_loss.replace(0, 1e-10)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_latest = rsi.iloc[-1].item() if not rsi.empty and pd.notna(rsi.iloc[-1]) else 50

        ma20_series = history['Close'].rolling(window=20).mean()
        ma20 = ma20_series.iloc[-1].item() if not ma20_series.empty and pd.notna(ma20_series.iloc[-1]) else history['Close'].iloc[-1].item()

        latest_price = history['Close'].iloc[-1].item() if not history.empty else np.nan
        price_above_ma20 = (latest_price / ma20 - 1) * 100 if pd.notna(latest_price) and pd.notna(ma20) and ma20 != 0 else 0

        high_all_time = history['Close'].max()
        near_all_time_high = current_price >= 0.98 * high_all_time if current_price and pd.notna(high_all_time) else False

        overbought = rsi_latest > 70 or (rsi_latest > 65 and (price_above_ma20 > 15 or near_all_time_high))
        return overbought, {'rsi': rsi_latest, 'price_above_ma20': price_above_ma20, 'near_all_time_high': near_all_time_high}
    except Exception as e:
        print(f"Error in is_overbought({ticker}): {e}")
        return False, {'rsi': 50, 'price_above_ma20': 0, 'near_all_time_high': False}

def is_underbought(ticker, history, current_price=None):
    try:
        if history.empty or len(history) < 20:
            print(f"⚠️ Insufficient history data for {ticker} in is_underbought")
            return False, {'rsi': 50, 'price_below_ma20': 0, 'near_52w_low': False}

        # === RSI Calculation ===
        delta = history['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean().replace(0, 1e-10)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_latest = rsi.iloc[-1].item() if not rsi.empty and pd.notna(rsi.iloc[-1]) else 50

        # === MA20 and Price Gap ===
        ma20_series = history['Close'].rolling(window=20).mean()
        ma20 = ma20_series.iloc[-1].item() if not ma20_series.empty and pd.notna(ma20_series.iloc[-1]) else history['Close'].iloc[-1].item()
        latest_price = history['Close'].iloc[-1].item() if not history.empty else np.nan
        price_below_ma20 = (1 - latest_price / ma20) * 100 if pd.notna(latest_price) and pd.notna(ma20) and ma20 != 0 else 0

        # === Near 52-Week Low ===
        low_52w = history['Close'].min()
        near_52w_low = current_price <= 1.02 * low_52w if current_price and pd.notna(low_52w) else False

        underbought = rsi_latest < 30 or (rsi_latest < 35 and (price_below_ma20 > 10 or near_52w_low))
        return underbought, {'rsi': rsi_latest, 'price_below_ma20': price_below_ma20, 'near_52w_low': near_52w_low}

    except Exception as e:
        print(f"Error in is_underbought({ticker}): {e}")
        return False, {'rsi': 50, 'price_below_ma20': 0, 'near_52w_low': False}

def get_dividend_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="1d")
        current_price = history['Close'].iloc[-1] if not history.empty else info.get("currentPrice")

        # === Forward Dividend Yield ===
        forward_yield = info.get('dividendYield')
        if forward_yield is not None:
            if forward_yield > 1:  # Already in percentage (e.g., 6.64)
                forward_yield *= 0.01
            forward_yield_str = f"{forward_yield * 100:.2f}%"
        else:
            forward_yield_str = "N/A"

        # === Trailing Dividend Yield ===
        dividends = stock.dividends
        if dividends is not None and not dividends.empty and current_price:
            last_4_divs = dividends[-4:].sum()
            trailing_yield = last_4_divs / current_price * 100
            trailing_yield_str = f"{trailing_yield:.2f}%"
        else:
            trailing_yield_str = "N/A"

        # === Ex-Dividend Date ===
        ex_dividend_date = "N/A"
        if 'exDividendDate' in info and info['exDividendDate']:
            ts = info['exDividendDate']
            if isinstance(ts, (int, float)):
                ex_dividend_date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')

        return forward_yield_str, trailing_yield_str, ex_dividend_date

    except Exception as e:
        print(f"⚠️ Error getting dividend info for {ticker}: {e}")
        return "N/A", "N/A", "N/A"


# Let's debug `get_trend_signals()` thoroughly with weights and short/long-term split.
def get_trend_signals(ticker, history):
    try:
        if not all(col in history.columns for col in ['MACD', 'Signal', 'RSI', 'MA20', 'MA50']):
            print(f"⚠️ Missing columns for {ticker} in get_trend_signals")
            return "Neutral", "N/A"

        macd = history['MACD'].iloc[-1]
        macd_prev = history['MACD'].iloc[-2]
        signal_line = history['Signal'].iloc[-1]
        signal_prev = history['Signal'].iloc[-2]
        rsi = history['RSI'].iloc[-1]
        rsi_prev = history['RSI'].iloc[-2]
        price = history['Close'].iloc[-1]

        ma20 = history['MA20'].iloc[-1]
        ma50 = history['MA50'].iloc[-1]
        ma200 = history['Close'].rolling(window=200).mean().iloc[-1] if len(history) >= 200 else None

        # === Short-Term Weighted Signal ===
        short_score = 0

        # MACD Crossover (weight 2)
        if macd_prev < signal_prev and macd > signal_line:
            short_score += 2
        elif macd_prev > signal_prev and macd < signal_line:
            short_score -= 2

        # RSI Momentum (weight 1)
        if rsi > 50 and rsi > rsi_prev:
            short_score += 1
        elif rsi < 50 and rsi < rsi_prev:
            short_score -= 1

        # MA20 Position (weight 1)
        if price > ma20:
            short_score += 1
        elif price < ma20:
            short_score -= 1

        # Short-term Classification
        if short_score >= 3:
            short_term_signal = "Strong Bullish"
        elif short_score == 1.5:
            short_term_signal = "Bullish"
        elif short_score == 1:
            short_term_signal = "Moderate Bullish"
        elif short_score == 0:
            short_term_signal = "Neutral"
        elif short_score == -1:
            short_term_signal = "Moderate Bearish"
        elif short_score == -1.5:
            short_term_signal = "Bearish"
        else:
            short_term_signal = "Strong Bearish"

        # === Long-Term Weighted Signal ===
        if ma200 is None or pd.isna(ma200):
            long_term_signal = "N/A"
        else:
            long_score = 0

            # Trend Structure (MA50 vs MA200) (weight 2)
            if ma50 > ma200 and price > ma200:
                long_score += 2
            elif ma50 < ma200 and price < ma200:
                long_score -= 2

            # MACD Direction (weight 1)
            if macd > 0:
                long_score += 1
            elif macd < 0:
                long_score -= 1

            # RSI Level (weight 1)
            if rsi > 55:
                long_score += 1
            elif rsi < 45:
                long_score -= 1

            # Long-term Classification
            if long_score >= 3:
                long_term_signal = "Strong Bullish"
            elif long_score == 1.5:
                long_term_signal = "Bullish"
            elif long_score == 1:
                long_term_signal = "Slight Bullish"
            elif long_score == 0:
                long_term_signal = "Neutral"
            elif long_score == -1:
                long_term_signal = "Slight Bearish"
            elif long_score == -1.5:
                long_term_signal = "Bearish"
            else:
                long_term_signal = "Strong Bearish"

        return short_term_signal, long_term_signal

    except Exception as e:
        print(f"❌ Error determining trend signals for {ticker}: {e}")
        return "Neutral", "N/A"




def is_overvalued(ticker, stock_info):
    label, score, details = valuation_score(ticker, stock_info)
    return label == 'Overvalued', details

def is_undervalued(ticker, stock_info):
    label, score, details = valuation_score(ticker, stock_info)
    return label == 'Undervalued', details

def get_two_action_recommendation(
    tech_score, fund_score, sentiment_score, overvalued, overbought,
    undervalued, underbought, val_details, stock_type, current_price,
    high_52_week, is_high_risk=False, market_mode="neutral"
):
    thresholds_by_type = {
        "high_growth": {"fund_buy": 60, "fund_hold": 45, "fund_sell": 30, "tech_buy": 70, "tech_sell": 50, "peg_upper": 2.2, "peg_lower": 1.4},
        "value": {"fund_buy": 75, "fund_hold": 60, "fund_sell": 40, "tech_buy": 65, "tech_sell": 40, "peg_upper": 1.2, "peg_lower": 0.8},
        "speculative": {"fund_buy": 50, "fund_hold": 30, "fund_sell": 20, "tech_buy": 80, "tech_sell": 60, "peg_upper": 99, "peg_lower": 0},
        "stable": {"fund_buy": 70, "fund_hold": 50, "fund_sell": 35, "tech_buy": 75, "tech_sell": 45, "peg_upper": 1.5, "peg_lower": 1.0},
    }

    market_mode_adjustment = {
        "bull": {"fund_buy": -5, "tech_buy": -5},
        "bear": {"fund_buy": +5, "tech_buy": +5},
        "neutral": {"fund_buy": 0, "tech_buy": 0}
    }

    th = thresholds_by_type.get(stock_type, thresholds_by_type["stable"])
    adj = market_mode_adjustment.get(market_mode, market_mode_adjustment["neutral"])
    adjusted_fund_buy = th['fund_buy'] + adj['fund_buy']
    adjusted_tech_buy = th['tech_buy'] + adj['tech_buy']

    peg = val_details.get('peg', float('nan'))
    pe = val_details.get('pe', float('nan'))
    val_score = val_details.get('valuation', float('nan'))

    short_term = []
    pullback_target = current_price * 0.85 if current_price else None

    strong_continuer = (
        (not np.isnan(peg) and peg < th['peg_lower']) +
        (fund_score > adjusted_fund_buy) +
        (current_price is not None and high_52_week is not None and current_price >= 0.96 * high_52_week)
    ) >= 2

    if overbought and not strong_continuer:
        short_term.append(f"Sell to secure profit / Buy on pullback" if pullback_target else "Sell / Reload")
    elif overbought and strong_continuer:
        short_term.append("Hold or Partial Profit-Taking")
    elif tech_score < th['tech_sell'] and fund_score >= adjusted_fund_buy:
        short_term.append("Pullback Opportunity (Strong Fundamentals)")
    elif tech_score < th['tech_sell']:
        short_term.append("Sell or Avoid")
    elif tech_score > adjusted_tech_buy:
        short_term.append("Buy")
    elif underbought:
        short_term.append("Watch for Rebound")
    else:
        short_term.append("Hold")

    if is_high_risk:
        short_term[0] += " (High Risk)"

    # Long-term logic with backup for PEG
    is_peg_missing = np.isnan(peg)
    peg_condition = (not is_peg_missing and peg < th['peg_lower'])
    backup_peg_logic = (is_peg_missing and not np.isnan(pe) and pe < 15 and val_score < 70)

    if fund_score > adjusted_fund_buy and (undervalued or peg_condition or backup_peg_logic):
        long_term = "Buy"
    elif fund_score > th['fund_hold'] and not overvalued:
        long_term = "Moderate Buy"
    elif fund_score < th['fund_sell'] or (not is_peg_missing and peg > th['peg_upper']):
        long_term = "Exit / Avoid"
    else:
        long_term = "Hold"

    return short_term, long_term


def get_market_mode_from_trend_signal(index_ticker: str = "SPY", period: str = "6mo") -> Literal["bull", "neutral", "bear"]:
    stock = yf.Ticker(index_ticker)
    history = stock.history(period=period)

    if history.empty or len(history) < 50:
        return "neutral"

    tech_score, hist = technical_score(index_ticker, history, period)
    short_term_signal, long_term_signal = get_trend_signals(index_ticker, hist)

    bullish_signals = ["Strong Bullish", "Bullish", "Moderate Bullish", "Slight Bullish"]
    bearish_signals = ["Strong Bearish", "Bearish", "Moderate Bearish", "Slight Bearish"]

    if long_term_signal == "N/A":
        if short_term_signal in bullish_signals:
            return "bull"
        elif short_term_signal in bearish_signals:
            return "bear"
        else:
            return "neutral"

    if short_term_signal in bullish_signals and long_term_signal in bullish_signals:
        return "bull"
    elif short_term_signal in bearish_signals or long_term_signal in bearish_signals:
        return "bear"
    else:
        return "neutral"


def get_52_week_high(ticker: str) -> float:
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="1y")
        if history.empty:
            return float("nan")
        return history['Close'].max()
    except Exception as e:
        print(f"⚠️ Error fetching 52-week high for {ticker}: {e}")
        return float("nan")


import yfinance as yf
import pandas as pd
import numpy as np
from tabulate import tabulate

# Assume these functions are defined elsewhere in your codebase
# technical_score, news_sentiment_score, fundamental_score, valuation_score,
# is_overvalued, is_undervalued, is_overbought, is_underbought,
# classify_stock_type, get_two_action_recommendation

def analyze_stock(ticker, period="3mo"):
    try:
        stock = yf.Ticker(ticker)
        stock_info = stock.info
        history = stock.history(period=period)

        if history.empty or len(history) < 50:
            print(f"⚠️ Data not sufficient for {ticker}")
            return None

        # Step 1: Detect market mode using SPY
        market_mode = get_market_mode_from_trend_signal("SPY", period)

        # Step 2: Run technicals and signals
        tech, history_with_indicators = technical_score(ticker, history, period)
        short_term_signal, long_term_signal = get_trend_signals(ticker, history_with_indicators)
        news = news_sentiment_score(ticker)
        fund, fund_indicators = fundamental_score(ticker, stock_info, history)

        valuation_class, valuation_score_val, val_details = valuation_score(ticker, stock_info)
        overvalued, _ = is_overvalued(ticker, stock_info)
        undervalued, _ = is_undervalued(ticker, stock_info)
        current_price = stock_info.get('currentPrice', history['Close'].iloc[-1].item() if not history.empty else None)
        overbought, momentum_details = is_overbought(ticker, history, current_price)
        underbought, _ = is_underbought(ticker, history, current_price)
        stock_type = classify_stock_type(stock_info)

        high_52_week = get_52_week_high(ticker)
        div_yield_forward, div_yield_trailing, ex_div_date = get_dividend_info(ticker)
        is_high_risk = stock_type in ['speculative'] or (stock_type == 'high_growth' and fund < 50)

        short_term_rec, long_term_rec = get_two_action_recommendation(
            tech, fund, news, overvalued, overbought, undervalued, underbought,
            val_details, stock_type, current_price, high_52_week,
            is_high_risk=is_high_risk, market_mode=market_mode
        )

        total = round(0.5 * tech + 0.5 * fund, 2)
        open_price = history_with_indicators['Open'].iloc[-1]
        close_price = history_with_indicators['Close'].iloc[-1]
        price_change = ((close_price - open_price) / open_price) * 100

        data = {
            'Ticker': ticker,
            'Period': period,
            'Stock Type': stock_type,
            'Tech Score': tech,
            'Fund Score': fund,
            'Total Score': total,
            'Valuation': valuation_class,
            'PEG': f"{val_details.get('peg', float('nan')):.2f}",
            'P/E': f"{val_details.get('pe', float('nan')):.2f}",
            'EPS': f"{fund_indicators.get('eps', float('nan')):.2f}",
            'EPS Growth (YoY)': f"{fund_indicators.get('eps_growth', float('nan')):.2f}",
            'P/S': f"{val_details.get('ps', float('nan')):.2f}",
            'ROE': f"{fund_indicators.get('roe', float('nan')):.2f}",
            'Revenue Growth': f"{fund_indicators.get('revenue_growth', float('nan')):.2f}",
            'Debt-to-Equity': f"{fund_indicators.get('debt_to_equity', float('nan')):.2f}",
            'FCF Yield': f"{fund_indicators.get('fcf_yield', float('nan')):.2f}",
            'Forward Dividend Yield': div_yield_forward,
            'Trailing Dividend Yield': div_yield_trailing,
            'Ex-Div Date': ex_div_date,
            'Momentum': 'Overbought' if overbought else 'Underbought' if underbought else 'Neutral',
            'RSI': f"{momentum_details.get('rsi', float('nan')):.2f}",
            'Price > MA20 (%)': f"{momentum_details.get('price_above_ma20', float('nan')):.2f}",
            '52W High': f"{high_52_week:.2f}",
            'Current vs 52W High (%)': f"{((current_price / high_52_week - 1) * 100):.2f}%" if current_price and high_52_week else "N/A",
            'Strong Momentum Exception Rule': 'Yes' if stock_type == 'high_growth' and fund > 55 and tech > 70 and val_details.get('peg', 99) < 2.2 and overbought and current_price >= 0.98 * high_52_week else 'No',
            'Short-Term Rec': ', '.join(short_term_rec),
            'Long-Term Rec': long_term_rec,
            'Open Price': f"${open_price:.2f}",
            'Close Price': f"${close_price:.2f}",
            'Price Change (%)': f"{price_change:.2f}%",
            'Trend Signal': short_term_signal,
            'Market Mode': market_mode
        }

        print(f"\n📊 {ticker} Stock Analysis (Period: {period})")
        print(tabulate([data], headers="keys", tablefmt="grid", floatfmt=".2f"))

        df = pd.DataFrame([data])
        output_filename = f"{ticker}_analysis.xlsx"
        df.to_excel(output_filename, index=False)
        print(f"\U0001f4be Exported to {output_filename}")

        return data

    except Exception as e:
        print(f"❌ Error analyzing {ticker}: {e}")
        return None





def analyze_bulk_stocks(ticker_list, period="3mo"):
    try:
        results = []

        print(f"\nAnalyzing {len(ticker_list)} stocks...")
        for ticker in tqdm(ticker_list, desc="Processing"):
            result = analyze_stock(ticker, period)
            if result:
                results.append(result)
            time.sleep(0.5)  # Avoid yfinance rate limits

        if not results:
            print("No valid results to save.")
            return

        # Create DataFrame and save to Excel
        df = pd.DataFrame(results)
        today = datetime.today().strftime('%Y%m%d')
        filename = f"bulk_analysis_{today}.xlsx"
        df.to_excel(filename, index=False, engine='openpyxl')
        print(f"\nResults saved to {filename}")

    except Exception as e:
        print(f"Error in bulk analysis: {e}")

# Call command for Colab
  # Change to 500 for full analysis
#for dividends stocks

import finnhub

finnhub_client = finnhub.Client(api_key="YOUR_FINNHUB_API_KEY")  # Replace with your real key

def get_dividend_payout_and_growth_years(ticker):
    try:
        payout_ratio = "N/A"
        consecutive_years = "N/A"

        # Using yfinance to estimate payout ratio
        stock = yf.Ticker(ticker)
        info = stock.info
        dividends = stock.dividends
        earnings = stock.financials.loc['Net Income'] if 'Net Income' in stock.financials.index else None

        if dividends is not None and not dividends.empty and earnings is not None:
            last_year_div = dividends[-4:].sum()
            last_year_earnings = earnings[-1]
            if last_year_earnings > 0:
                payout_ratio = round((last_year_div / last_year_earnings) * 100, 2)

        # Use finnhub to estimate consecutive growth years (fallback logic)
        # For demonstration, set dummy value
        consecutive_years = "12+"  # Placeholder logic

        return payout_ratio, consecutive_years

    except Exception as e:
        print(f"⚠️ Error fetching dividend payout info for {ticker}: {e}")
        return "N/A", "N/A"

# Integrate into analyze_stock output
def analyze_stock_with_dividend_upgrade(ticker, period="3mo"):
    data = analyze_stock(ticker, period)
    if not data:
        return None

    payout, years_growth = get_dividend_payout_and_growth_years(ticker)
    data['Dividend Payout Ratio (%)'] = payout
    data['Consecutive Dividend Growth Years'] = years_growth

    # Overwrite export with new fields
    df = pd.DataFrame([data])
    output_filename = f"{ticker}_analysis.xlsx"
    df.to_excel(output_filename, index=False)
    print(f"\n🔄 Updated with dividend payout info → {output_filename}")
    return data

def bulk_analysis_with_dividend(tickers, period="3mo"):
    results = []
    for ticker in tickers:
        print(f"\n📊 Analyzing {ticker}...")
        data = analyze_stock_with_dividend_upgrade(ticker, period)
        if data:
            results.append(data)

    if not results:
        print("No valid results to save.")
        return None

    df = pd.DataFrame(results)
    today_str = datetime.today().strftime("%Y%m%d")
    filename = f"dividend_bulk_{today_str}.xlsx"
    df.to_excel(filename, index=False, engine='openpyxl')
    print(f"\n✅ Results saved to {filename}")
    return df

#ticker list
tech_ticker_list = [
    "NVDA", "AMD", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ASML",
    "INTC", "ORCL", "CRM", "ADBE", "SNOW", "PLTR", "SHOP", "UBER", "BIDU", "TSM",
    "ARM", "ANET", "SMCI", "NET", "DDOG", "ZS", "AI", "PATH", "U", "RBLX"]

nuclear_ticker_list = [
    "CCJ", "CEG", "VST", "GEV", "BWXT", "OKLO", "SMR", "NNE", "LEU", "MIR",
    "UEC", "HII", "FLR", "ETR", "D"
]
daily_watch_list = ["NBIS", "CRWV", "NET", "CRWD", "ZS"]
top_5_tech = ["NVDA", "AMD", "AAPL", "MSFT", "GOOGL"]
div_stock = ["PCAR", "ROST", "NVT", "ARE","MO", "AES"]
analyze_stock("")
#analyze_bulk_stocks(nuclear_ticker_list)
#analyze_bulk_stocks(top_5_tech)
#bulk_analysis_with_dividend(div_stock)