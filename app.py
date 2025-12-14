import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from typing import Optional

# Optional: FRED macro data
try:
    from fredapi import Fred
except ImportError:
    Fred = None

# ============================================================================
# CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="Investment Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Modern Investment Terminal - Your gateway to market intelligence"
    }
)

# Custom CSS for modern, clean design
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f8fafc;
    }
    
    /* Card-like containers */
    .stContainer {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
    }
    
    /* Headers */
    h1 {
        color: #1e293b;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #334155;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    h3 {
        color: #475569;
        font-weight: 500;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
    }
    
    [data-testid="stMetricLabel"] {
        color: #64748b;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid #334155;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #f1f5f9;
        border-bottom: 2px solid #475569;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 8px;
        padding: 10px 20px;
        color: #64748b;
        font-weight: 600;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: 1px solid #2563eb;
    }
    
    /* Dataframes */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        background-color: #eff6ff;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f1f5f9;
        border-radius: 8px;
        font-weight: 600;
        color: #334155;
        border: 1px solid #e2e8f0;
    }
    
    /* Divider */
    hr {
        border-color: #e2e8f0;
        margin: 2rem 0;
    }
    
    /* Text colors */
    p, li {
        color: #475569;
    }
    
    .stMarkdown {
        color: #334155;
    }
    </style>
""", unsafe_allow_html=True)

# Try to read API keys
try:
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
except Exception:
    NEWS_API_KEY = "YOUR_NEWSAPI_KEY_HERE"

COUNTRY_MAP = {
    "🌍 All Countries": None,
    "🇺🇸 United States": "us",
    "🇬🇧 United Kingdom": "gb",
    "🇿🇦 South Africa": "za",
    "🇩🇪 Germany": "de",
    "🇫🇷 France": "fr",
    "🇨🇦 Canada": "ca",
    "🇯🇵 Japan": "jp",
    "🇨🇳 China": "cn",
    "🇮🇳 India": "in",
    "🇦🇺 Australia": "au",
}

SECTOR_KEYWORDS = {
    "All Sectors": None,
    "💻 Technology": "technology OR semiconductor OR software OR AI OR cloud",
    "💰 Finance": "banking OR finance OR credit OR insurance",
    "⚡ Energy": "oil OR gas OR renewable OR solar OR energy",
    "🏥 Healthcare": "biotech OR pharmaceuticals OR healthcare",
    "🛍️ Consumer": "retail OR consumer goods OR e-commerce",
    "🚗 Automotive": "EV OR automotive OR cars OR transportation",
    "₿ Crypto": "crypto OR bitcoin OR ethereum OR blockchain",
    "✈️ Aerospace & Defence": "aerospace OR defence OR military",
    "🏢 Real Estate": "real estate OR REIT OR housing",
    "🏭 Manufacturing": "industrial OR manufacturing OR production",
}

MACRO_SERIES = {
    "Consumer Price Index (CPI)": {
        "id": "CPIAUCSL",
        "units": "Index",
        "description": "Headline inflation measure for urban consumers",
        "icon": "💵"
    },
    "Unemployment Rate": {
        "id": "UNRATE",
        "units": "%",
        "description": "Percentage of labor force without jobs",
        "icon": "👥"
    },
    "Federal Funds Rate": {
        "id": "FEDFUNDS",
        "units": "%",
        "description": "Target interest rate set by the Fed",
        "icon": "📊"
    },
    "Nonfarm Payrolls": {
        "id": "PAYEMS",
        "units": "Thousands",
        "description": "Total employed in non-agricultural sectors",
        "icon": "💼"
    },
}
# ============================================================================
# YIELD CURVE (FRED TREASURY CONSTANT MATURITY)
# ============================================================================

YIELD_CURVE_SERIES = {
    "1M": "DGS1MO",
    "3M": "DGS3MO",
    "6M": "DGS6MO",
    "1Y": "DGS1",
    "2Y": "DGS2",
    "5Y": "DGS5",
    "10Y": "DGS10",
    "30Y": "DGS30",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
# ============================================================================
# SYMBOL RESOLUTION (NAME -> TICKER)
# ============================================================================
def style_plotly(fig, height: Optional[int] = None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        font=dict(color="#e5e7eb", size=13),
        margin=dict(l=12, r=12, t=40, b=12),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#111827", font_color="#e5e7eb"),
        legend=dict(bgcolor="rgba(15,23,42,0.6)", bordercolor="rgba(148,163,184,0.25)", borderwidth=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.12)")
    if height:
        fig.update_layout(height=height)
    return fig



def section_header(title: str, help_text: str, level: int = 2):
    safe = help_text.replace('"', "'")
    st.markdown(
        f"""
        <div class="hdr-wrap">
          <h{level} style="margin:0;">{title}</h{level}>
          <span class="help-dot" title="{safe}">?</span>
        </div>
        """,
        unsafe_allow_html=True
    )

SYMBOL_ALIASES = {
    # ---------------------------
    # US mega caps / common names
    # ---------------------------
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "nvidia": "NVDA",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "amd": "AMD",
    "intel": "INTC",
    "palantir": "PLTR",
    "berkshire": "BRK-B",
    "berkshire hathaway": "BRK-B",

    # ---------------------------
    # Indices / ETFs
    # ---------------------------
    "s&p 500": "^GSPC",
    "sp500": "^GSPC",
    "s&p500": "^GSPC",
    "nasdaq": "^IXIC",
    "dow": "^DJI",
    "dow jones": "^DJI",
    "vix": "^VIX",
    "spy": "SPY",
    "qqq": "QQQ",
    "iwm": "IWM",
    "dia": "DIA",

    # ---------------------------
    # Crypto (Yahoo Finance uses -USD pairs)
    # ---------------------------
    "bitcoin": "BTC-USD",
    "btc": "BTC-USD",
    "ethereum": "ETH-USD",
    "eth": "ETH-USD",
    "solana": "SOL-USD",
    "sol": "SOL-USD",
    "ripple": "XRP-USD",
    "xrp": "XRP-USD",
    "cardano": "ADA-USD",
    "ada": "ADA-USD",
    "dogecoin": "DOGE-USD",
    "doge": "DOGE-USD",
    "litecoin": "LTC-USD",
    "ltc": "LTC-USD",
    "chainlink": "LINK-USD",
    "link": "LINK-USD",
    "polkadot": "DOT-USD",
    "dot": "DOT-USD",

    # ---------------------------
    # Commodities / Futures (Yahoo Finance tickers)
    # ---------------------------
    "gold": "GC=F",
    "gold futures": "GC=F",
    "silver": "SI=F",
    "silver futures": "SI=F",
    "oil": "CL=F",
    "crude oil": "CL=F",
    "wti": "CL=F",
    "brent": "BZ=F",
    "natural gas": "NG=F",
    "gas": "NG=F",
    "copper": "HG=F",
    "corn": "ZC=F",
    "wheat": "ZW=F",
    "soybeans": "ZS=F",
    "cotton": "CT=F",
    "coffee": "KC=F",
    "sugar": "SB=F",

    # ---------------------------
    # FX (major pairs)
    # ---------------------------
    "usd zar": "USDZAR=X",
    "usdzar": "USDZAR=X",
    "eur usd": "EURUSD=X",
    "eurusd": "EURUSD=X",
    "gbp usd": "GBPUSD=X",
    "gbpusd": "GBPUSD=X",
    "usd jpy": "USDJPY=X",
    "usdjpy": "USDJPY=X",
}

def resolve_symbol(text: str) -> str:
    """
    Convert a company/asset name or ticker into a Yahoo Finance symbol.
    - Case-insensitive
    - Falls back to uppercase input if not found in aliases
    """
    if not text:
        return ""

    raw = text.strip()
    key = raw.lower()

    if key in SYMBOL_ALIASES:
        return SYMBOL_ALIASES[key]

    # light normalisation
    key2 = (
        key.replace("&", "and")
           .replace(".", "")
           .replace(",", "")
           .replace("  ", " ")
           .strip()
    )
    if key2 in SYMBOL_ALIASES:
        return SYMBOL_ALIASES[key2]

    # otherwise assume it's already a valid Yahoo symbol
    return raw.upper()

# ========= TECHNICAL ANALYSIS HELPERS =========

def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Standard RSI (Wilder-style approximation)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use exponential moving average for smoother RSI
    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_mas(close: pd.Series, short_window: int = 20, long_window: int = 50):
    """Return short and long simple moving averages."""
    short_ma = close.rolling(short_window).mean()
    long_ma = close.rolling(long_window).mean()
    return short_ma, long_ma


def detect_crossovers(short_ma: pd.Series, long_ma: pd.Series):
    """
    Detect golden/death crosses:
      - golden: short crosses ABOVE long
      - death:  short crosses BELOW long
    Returns two boolean Series.
    """
    prev_short = short_ma.shift(1)
    prev_long = long_ma.shift(1)

    golden = (short_ma > long_ma) & (prev_short <= prev_long)
    death = (short_ma < long_ma) & (prev_short >= prev_long)
    return golden, death

def fetch_news_v2(query: str, country: str, sector: str, max_articles: int = 20):
    if NEWS_API_KEY == "YOUR_NEWSAPI_KEY_HERE":
        return [], "⚠️ Add your NEWS_API_KEY to Streamlit secrets."

    url = "https://newsapi.org/v2/everything"

    search_terms = []
    if query:
        search_terms.append(query)
    if sector and sector in SECTOR_KEYWORDS and SECTOR_KEYWORDS[sector]:
        search_terms.append(f"({SECTOR_KEYWORDS[sector]})")

    full_query = " AND ".join(search_terms) if search_terms else "markets"

    params = {
        "q": full_query,
        "language": "en",
        "pageSize": max_articles,
        "apiKey": NEWS_API_KEY,
        "sortBy": "publishedAt",
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()

        if data.get("status") != "ok":
            return [], f"API error: {data.get('message')}"

        return data.get("articles", []), None

    except Exception as e:
        return [], f"Error fetching news: {e}"


def _get_fred():
    if Fred is None:
        return None, "fredapi is not installed."

    api_key = st.secrets.get("FRED_API_KEY", "")
    if not api_key or api_key == "YOUR_FRED_API_KEY_HERE":
        return None, "Add your FRED_API_KEY to Streamlit secrets."

    try:
        return Fred(api_key=api_key), None
    except Exception as e:
        return None, f"Error initialising FRED: {e}"


def load_macro_series(macro_name, lookback_years=5):
    fred, err = _get_fred()
    if err:
        return None, err

    meta = MACRO_SERIES.get(macro_name)
    if not meta:
        return None, "Unknown macro series."

    series_id = meta["id"]
    end = date.today()
    start = end - relativedelta(years=lookback_years)

    try:
        s = fred.get_series(series_id, observation_start=start, observation_end=end)
    except Exception as e:
        return None, f"FRED error: {e}"

    if s is None or s.empty:
        return None, "No data returned."

    s = s.dropna()
    s.name = macro_name
    return s, None

def load_yield_curve(lookback_years: int = 5):
    """
    Load multiple Treasury yield series from FRED and return a DataFrame:
    index = date, columns = tenors (e.g., 1M, 3M, 2Y, 10Y), values = yields (%).
    """
    fred, err = _get_fred()
    if err:
        return None, err

    end = date.today()
    start = end - relativedelta(years=lookback_years)

    data = {}
    for tenor, series_id in YIELD_CURVE_SERIES.items():
        try:
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
        except Exception as e:
            return None, f"FRED error loading {series_id}: {e}"

        if s is None or s.empty:
            data[tenor] = pd.Series(dtype=float)
        else:
            data[tenor] = s.dropna()

    df = pd.DataFrame(data).sort_index()
    if df.empty:
        return None, "No yield curve data returned."
    return df, None


def latest_curve_snapshot(df: pd.DataFrame):
    """
    Get the latest row with any available yields.
    Returns (as_of_date, snapshot_series_sorted_by_tenor).
    """
    if df is None or df.empty:
        return None, None

    tmp = df.dropna(how="all")
    if tmp.empty:
        return None, None

    as_of = tmp.index[-1]
    snap = tmp.loc[as_of].dropna()

    # tenor ordering
    order = list(YIELD_CURVE_SERIES.keys())
    snap = snap.reindex([t for t in order if t in snap.index])
    return as_of, snap

def build_release_table(series, last_n=12):
    s = series.sort_index().dropna()
    tail = s.tail(last_n)

    df = tail.to_frame(name="Value")
    df["Previous"] = df["Value"].shift(1)
    df["Change"] = df["Value"] - df["Previous"]
    df["% Change"] = df["Value"].pct_change() * 100
    df = df.reset_index().rename(columns={"index": "Date"})
    
    return df


def compute_market_impact(releases: pd.DataFrame, ticker: str, window_days: int = 3):
    """
    For each macro release date, compute the % move in `ticker` from the close on
    (or just before) the release date to `window_days` trading days after.
    Returns a DataFrame or None if nothing can be computed.
    """
    if releases is None or releases.empty:
        return None

    # Ensure Date column is datetime
    rel = releases.copy()
    rel["Date"] = pd.to_datetime(rel["Date"])

    start_date = rel["Date"].min() - relativedelta(days=10)
    end_date = rel["Date"].max() + relativedelta(days=window_days + 5)

    try:
        hist = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )
    except Exception:
        return None

    if hist.empty or "Close" not in hist.columns:
        return None

    hist = hist.sort_index()
    impacts = []

    for _, row in rel.iterrows():
        d = row["Date"]

        before = hist.loc[hist.index <= d]
        if before.empty:
            continue
        before_px = float(before["Close"].iloc[-1])
        before_date = before.index[-1]

        after = hist.loc[hist.index >= d]
        if after.empty:
            continue

        idx = min(window_days - 1, len(after) - 1)
        after_px = float(after["Close"].iloc[idx])
        after_date = after.index[idx]

        move_pct = (after_px / before_px - 1.0) * 100.0

        impacts.append(
            {
                "Release Date": d,
                "Before Date": before_date,
                "After Date": after_date,
                "Price Before": before_px,
                "Price After": after_px,
                "Move (%)": move_pct,
                "Macro Value": float(row["Value"]),
                "Macro Change": float(row["Change"]) if pd.notna(row["Change"]) else np.nan,
                "Macro % Change": float(row["% Change"]) if pd.notna(row["% Change"]) else np.nan,
            }
        )

    if not impacts:
        return None

    return pd.DataFrame(impacts)



# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.markdown("## 🎯 Navigation")
page = st.sidebar.radio(
    "Select Dashboard",
    ["📈 Stocks", "🌍 Macro"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Quick Tips")
st.sidebar.info("""
**Stocks**: Compare multiple tickers with advanced charts and fundamentals

**News**: Filter global financial news by country, sector, and keywords

**Macro**: Track economic indicators and market reactions
""")

# ============================================================================
# STOCKS PAGE
# ============================================================================

def stocks_page():
    st.markdown("<h1>📈 Stock Market Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("Compare tickers, analyze performance, and explore market correlations")
    
    st.markdown("---")
    
    # Controls in clean columns
    col1, col2 = st.columns([3, 1])
    
    with col1:
        tickers_input = st.text_input(
            "🎯 Enter Stock Tickers",
            value="AAPL, MSFT, NVDA",
            help="Separate multiple tickers with commas (e.g., AAPL, MSFT, TSLA, SPY)",
            placeholder="AAPL, MSFT, NVDA"
        )
    
    with col2:
        st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
        chart_mode = st.selectbox(
            "📊 Chart Type",
            ["📉 Line Chart", "🕯️ Candlestick"],
            help="Candlesticks work best with a single ticker"
        )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        period = st.selectbox(
            "📅 Time Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
            index=3
        )
    with col2:
        interval = st.selectbox(
            "⏱️ Interval",
            ["1d", "1wk", "1mo"],
            index=0
        )
    with col3:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)

    raw_inputs = [t for t in tickers_input.split(",") if t.strip()]
    tickers = [resolve_symbol(t) for t in raw_inputs]


    if not tickers:
        st.info("👆 Enter at least one ticker to begin your analysis")
        return

    # Download data
    with st.spinner("🔄 Fetching market data..."):
        try:
            data = yf.download(tickers, period=period, interval=interval, auto_adjust=True, progress=False)
        except Exception as e:
            st.error(f"❌ Could not download data: {e}")
            return

    if data.empty:
        st.warning("⚠️ No data available for the selected tickers and period")
        return

    closes = data["Close"] if "Close" in data else data
    vols = data["Volume"] if "Volume" in data else None

    if isinstance(closes, pd.Series):
        closes = closes.to_frame(name=tickers[0])

    closes.columns = [str(c).upper() for c in closes.columns]

    # Calculate metrics
    returns = closes.pct_change().dropna()
    if returns.empty:
        st.info("⚠️ Insufficient data to compute returns")
        return

    total_return = (closes.iloc[-1] / closes.iloc[0] - 1.0) * 100
    daily_vol = returns.std() * 100

    # ========== SNAPSHOT SECTION ==========
    st.markdown("<h2>📊 Market Snapshot</h2>", unsafe_allow_html=True)
    
    snap_left, snap_right = st.columns([2, 1])

    with snap_left:
        st.markdown("### Live Prices")
        cols_per_row = 3
        ticker_list = list(closes.columns)

        for row_start in range(0, len(ticker_list), cols_per_row):
            row_tickers = ticker_list[row_start : row_start + cols_per_row]
            row_cols = st.columns(len(row_tickers))

            for col_obj, t in zip(row_cols, row_tickers):
                latest = closes[t].iloc[-1]
                ret = total_return[t]
                vol = daily_vol[t]

                with col_obj:
                    st.metric(
                        label=t,
                        value=f"${latest:,.2f}",
                        delta=f"{ret:.2f}%",
                        help=f"Daily volatility: {vol:.2f}%"
                    )

    with snap_right:
        with st.expander("📋 Fundamentals & Ratios", expanded=False):
            st.caption("Financial metrics from Yahoo Finance")

            ratio_def = {
                "Market Cap": ("marketCap", lambda v: f"${v/1e9:,.1f}B"),
                "P/E (TTM)": ("trailingPE", lambda v: f"{v:.2f}"),
                "P/E (Fwd)": ("forwardPE", lambda v: f"{v:.2f}"),
                "Price/Book": ("priceToBook", lambda v: f"{v:.2f}"),
                "Price/Sales": ("priceToSalesTrailing12Months", lambda v: f"{v:.2f}"),
                "Div Yield": ("dividendYield", lambda v: f"{v*100:.2f}%"),
                "ROE": ("returnOnEquity", lambda v: f"{v*100:.1f}%"),
                "ROA": ("returnOnAssets", lambda v: f"{v*100:.1f}%"),
                "Profit Margin": ("profitMargins", lambda v: f"{v*100:.1f}%"),
                "Debt/Equity": ("debtToEquity", lambda v: f"{v:.2f}"),
                "Beta": ("beta", lambda v: f"{v:.2f}"),
            }

            labels = list(ratio_def.keys())
            default_labels = ["Market Cap", "P/E (TTM)", "P/E (Fwd)", "Div Yield", "ROE", "Beta"]

            selected = st.multiselect(
                "Select metrics to display",
                labels,
                default=default_labels
            )

            if not selected:
                st.info("Select at least one metric")
            else:
                rows = []
                for t in closes.columns:
                    try:
                        tk = yf.Ticker(t)
                        try:
                            info = tk.info
                        except Exception:
                            info = tk.get_info()
                    except Exception:
                        info = {}

                    row = {"Ticker": t}
                    for label in selected:
                        field, fmt = ratio_def[label]
                        raw = info.get(field, None)
                        if raw is None or raw == "None":
                            display = "–"
                        else:
                            try:
                                display = fmt(raw)
                            except Exception:
                                display = str(raw)
                        row[label] = display
                    rows.append(row)

                if rows:
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ========== PRICE CHARTS ==========
    st.markdown("<h2>📈 Price Action & Volume</h2>", unsafe_allow_html=True)

    single_ticker = len(closes.columns) == 1
    use_candlestick = "Candlestick" in chart_mode and single_ticker

    if use_candlestick:
        t = closes.columns[0]
        with st.spinner(f"📊 Loading candlestick data for {t}..."):
            try:
                hist = yf.Ticker(t).history(period=period, interval=interval)
            except Exception as e:
                st.error(f"Could not get OHLC data: {e}")
                hist = None

        needed = {"Open", "High", "Low", "Close"}
        if hist is None or hist.empty or not needed.issubset(hist.columns):
            st.warning("OHLC data unavailable – showing line chart instead")
            hist = None

        if hist is not None:
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.7, 0.3], vertical_spacing=0.03
            )

            fig.add_trace(
                go.Candlestick(
                    x=hist.index,
                    open=hist["Open"],
                    high=hist["High"],
                    low=hist["Low"],
                    close=hist["Close"],
                    name=t,
                    increasing_line_color='#10b981',
                    decreasing_line_color='#ef4444'
                ),
                row=1, col=1
            )

            if "Volume" in hist.columns:
                fig.add_trace(
                    go.Bar(x=hist.index, y=hist["Volume"], name="Volume", 
                           marker_color='#3b82f6', opacity=0.6),
                    row=2, col=1
                )

            fig.update_layout(
                height=600,
                template='plotly_white',
                paper_bgcolor='#ffffff',
                plot_bgcolor='#f8fafc',
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis2_title="Date",
                yaxis2_title="Volume",
                showlegend=False,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            use_candlestick = False

    if not use_candlestick:
    
        fig_price = go.Figure()
        # Distinct, vibrant colors that are easy to tell apart
        colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16']
        
        for idx, col in enumerate(closes.columns):
            fig_price.add_trace(
                go.Scatter(
                    x=closes.index, y=closes[col],
                    mode="lines", name=col,
                    line=dict(color=colors[idx % len(colors)], width=3)
                )
            )

        fig_price.update_layout(
            height=450,
            template='plotly_white',
            paper_bgcolor='#ffffff',
            plot_bgcolor='#f8fafc',
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode='x unified',
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="right", 
                x=1,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#e2e8f0',
                borderwidth=1
            ),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_price, use_container_width=True)

        if vols is not None and single_ticker:
            vol_series = vols[closes.columns[0]] if isinstance(vols, pd.DataFrame) else vols
            
            fig_vol = go.Figure(
                data=[go.Bar(x=vol_series.index, y=vol_series.values, 
                            name="Volume", marker_color='#3b82f6', opacity=0.7)]
            )
            fig_vol.update_layout(
                height=200,
                template='plotly_white',
                paper_bgcolor='#ffffff',
                plot_bgcolor='#f8fafc',
                xaxis_title="Date",
                yaxis_title="Volume",
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=40)
            )
            st.plotly_chart(fig_vol, use_container_width=True)

    # ========== NORMALIZED PERFORMANCE ==========
    st.markdown("<h2>📊 Relative Performance</h2>", unsafe_allow_html=True)
    st.caption("Normalized to 100 at start date for easy comparison")
    
    rebased = closes / closes.iloc[0] * 100.0

    fig_norm = go.Figure()
    # Use same distinct colors as price chart
    colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16']
    
    for idx, col in enumerate(rebased.columns):
        fig_norm.add_trace(
            go.Scatter(
                x=rebased.index, y=rebased[col],
                mode="lines", name=col,
                line=dict(color=colors[idx % len(colors)], width=3)
            )
        )

    fig_norm.update_layout(
        height=400,
        template='plotly_white',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#f8fafc',
        xaxis_title="Date",
        yaxis_title="Indexed Performance (Start = 100)",
        hovermode='x unified',
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#e2e8f0',
            borderwidth=1
        ),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_norm, use_container_width=True)

    # ========== CORRELATION ==========
    st.markdown("<h2>🔗 Correlation Matrix</h2>", unsafe_allow_html=True)
    st.caption("Daily return correlations between selected tickers")

    corr = returns.corr().round(2)
    fig_corr = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            colorbar=dict(title="Correlation"),
            text=corr.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 12, "color": "white"}
        )
    )
    fig_corr.update_layout(
        height=450,
        template='plotly_white',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#f8fafc',
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
        # =========================================================================
    # TECHNICAL ANALYSIS
    # =========================================================================
    st.markdown("<h2>📐 Technical Analysis</h2>", unsafe_allow_html=True)
    st.caption("RSI and moving averages for a selected ticker")

    # Only makes sense for a single ticker at a time
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    with tech_col1:
        ta_ticker = st.selectbox(
            "Ticker for TA",
            closes.columns.tolist(),
            index=0,
            help="Choose which ticker to analyse technically"
        )
    with tech_col2:
        rsi_window = st.number_input(
            "RSI window",
            min_value=5,
            max_value=50,
            value=14,
            step=1
        )
    with tech_col3:
        short_window = st.number_input(
            "Short MA",
            min_value=5,
            max_value=200,
            value=20,
            step=1
        )

    long_window = st.number_input(
        "Long MA",
        min_value=10,
        max_value=400,
        value=50,
        step=5,
        help="Typical combos: 20/50, 50/200, etc."
    )

    close_series = closes[ta_ticker].dropna()

    if close_series.empty:
        st.info("No price data available for technicals.")
    else:
        # ---- Compute indicators ----
        rsi = compute_rsi(close_series, window=rsi_window)
        short_ma, long_ma = compute_mas(close_series, short_window=short_window, long_window=long_window)
        golden, death = detect_crossovers(short_ma, long_ma)

        # ---- Price + MAs chart ----
        st.markdown(f"### Price & Moving Averages – {ta_ticker}")

        fig_ta_price = go.Figure()

        fig_ta_price.add_trace(
            go.Scatter(
                x=close_series.index,
                y=close_series.values,
                mode="lines",
                name="Close",
                line=dict(color="#111827", width=2)
            )
        )
        fig_ta_price.add_trace(
            go.Scatter(
                x=short_ma.index,
                y=short_ma.values,
                mode="lines",
                name=f"MA {short_window}",
                line=dict(color="#3b82f6", width=2)
            )
        )
        fig_ta_price.add_trace(
            go.Scatter(
                x=long_ma.index,
                y=long_ma.values,
                mode="lines",
                name=f"MA {long_window}",
                line=dict(color="#f97316", width=2)
            )
        )

        # Mark most recent golden/death crosses
        golden_dates = close_series.index[golden.fillna(False)]
        death_dates = close_series.index[death.fillna(False)]

        if len(golden_dates) > 0:
            last_golden = golden_dates[-1]
            fig_ta_price.add_trace(
                go.Scatter(
                    x=[last_golden],
                    y=[close_series.loc[last_golden]],
                    mode="markers",
                    marker=dict(color="#22c55e", size=12, symbol="triangle-up"),
                    name="Last golden cross"
                )
            )

        if len(death_dates) > 0:
            last_death = death_dates[-1]
            fig_ta_price.add_trace(
                go.Scatter(
                    x=[last_death],
                    y=[close_series.loc[last_death]],
                    mode="markers",
                    marker=dict(color="#ef4444", size=12, symbol="triangle-down"),
                    name="Last death cross"
                )
            )

        fig_ta_price.update_layout(
            height=450,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
        )

        st.plotly_chart(fig_ta_price, use_container_width=True)

        # ---- RSI chart ----
        st.markdown(f"### RSI – {ta_ticker}")

        fig_rsi = go.Figure()
        fig_rsi.add_trace(
            go.Scatter(
                x=rsi.index,
                y=rsi.values,
                mode="lines",
                name=f"RSI ({rsi_window})",
                line=dict(color="#6366f1", width=2)
            )
        )
        # 30 / 70 lines
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef4444")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#22c55e")

        fig_rsi.update_layout(
            height=250,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            xaxis_title="Date",
            yaxis_title="RSI",
            yaxis=dict(range=[0, 100]),
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
        )

        st.plotly_chart(fig_rsi, use_container_width=True)



# ============================================================================
# NEWS PAGE
# ============================================================================

def news_page():
    st.markdown("<h1>📰 Global Financial News</h1>", unsafe_allow_html=True)
    st.markdown("Stay informed with real-time financial news from around the world")
    
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        country = st.selectbox(
            "🌍 Select Country",
            list(COUNTRY_MAP.keys()),
            index=0
        )
    
    with col2:
        sector = st.selectbox(
            "🏢 Select Sector",
            list(SECTOR_KEYWORDS.keys()),
            index=0
        )
    
    with col3:
        max_articles = st.slider("📊 Max Articles", 5, 40, 20, step=5)

    query = st.text_input(
        "🔍 Search Keywords or Ticker",
        placeholder="e.g., NVDA, interest rates, inflation, Tesla..."
    )

    search_btn = st.button("🔎 Search News", use_container_width=True)

    if search_btn:
        with st.spinner("🔄 Fetching latest news..."):
            chosen_country_code = COUNTRY_MAP[country]
            chosen_sector = sector if sector != "All Sectors" else None

            articles, error = fetch_news_v2(
                query=query,
                country=chosen_country_code,
                sector=chosen_sector,
                max_articles=max_articles
            )

        if error:
            st.warning(f"⚠️ {error}")
            return

        if not articles:
            st.info("📭 No news found matching your filters. Try adjusting your search criteria.")
            return

        st.markdown(f"<h2>📑 {len(articles)} Articles Found</h2>", unsafe_allow_html=True)
        st.markdown("---")

        for art in articles:
            title = art.get("title", "No title")
            source = art.get("source", {}).get("name", "Unknown")
            url = art.get("url", "")
            desc = art.get("description", "")
            img_url = art.get("urlToImage", "")
            ts = art.get("publishedAt", "")

            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%b %d, %Y • %H:%M")
            except:
                pass

            col_img, col_content = st.columns([1, 3])
            
            with col_img:
                if img_url:
                    st.image(img_url, use_column_width=True)
            
            with col_content:
                st.markdown(f"### [{title}]({url})")
                st.caption(f"**{source}** • {ts}")
                if desc:
                    st.write(desc)
            
            st.markdown("---")


# ============================================================================
# MACRO PAGE
# ============================================================================

def macro_page():
    st.markdown("<h1>🌍 Macroeconomic Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("Track key economic indicators and analyze market reactions to data releases")
    
    st.markdown("---")

    fred, fred_err = _get_fred()
    if fred_err:
        st.warning("""
        ⚠️ **FRED API Setup Required**
        
        To use the Macro Dashboard, you need to:
        1. Add your `FRED_API_KEY` to Streamlit secrets
        2. Ensure the `fredapi` package is installed
        
        Get your free API key at: https://fred.stlouisfed.org/docs/api/api_key.html
        """)
        return

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        macro_name = st.selectbox(
            "📊 Select Economic Indicator",
            list(MACRO_SERIES.keys()),
            format_func=lambda x: f"{MACRO_SERIES[x]['icon']} {x}",
            index=0
        )

    with col2:
        lookback_years = st.slider("📅 History (Years)", 1, 20, 5)

    with col3:
        last_n = st.slider("📋 Recent Releases", 5, 30, 10)

    meta = MACRO_SERIES[macro_name]
    
    st.info(f"**{meta['icon']} {meta['description']}** • Units: `{meta['units']}`")

    st.markdown("---")

    series, err = load_macro_series(macro_name, lookback_years=lookback_years)

    if err:
        st.warning(f"⚠️ {err}")
        return

    if series is None or series.empty:
        st.info("📭 Could not load macro data. Check your FRED API key and internet connection.")
        return

    releases = build_release_table(series, last_n=last_n)

    tab_calendar, tab_history, tab_impact, tab_curve = st.tabs([
    "Release calendar",
    "Historical data",
    "Market impact",
    "Yield curve"
])


    # ========== TAB 1: CALENDAR ==========
    with tab_calendar:
        st.markdown(f"<h2>📅 Recent {macro_name} Releases</h2>", unsafe_allow_html=True)

        if releases is None or releases.empty:
            st.info("📭 No release data available")
        else:
            nice = releases.copy()
            nice["Date"] = pd.to_datetime(nice["Date"]).dt.strftime("%Y-%m-%d")
            
            # Round numeric columns
            for col in ["Value", "Previous", "Change", "% Change"]:
                if col in nice.columns:
                    nice[col] = pd.to_numeric(nice[col], errors='coerce').round(2)

            st.dataframe(
                nice,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Date": st.column_config.TextColumn("📅 Date", width="medium"),
                    "Value": st.column_config.NumberColumn(f"📊 Value ({meta['units']})", format="%.2f"),
                    "Previous": st.column_config.NumberColumn("Previous", format="%.2f"),
                    "Change": st.column_config.NumberColumn("Δ Change", format="%.2f"),
                    "% Change": st.column_config.NumberColumn("% Change", format="%.2f%%"),
                }
            )

    # ========== TAB 2: HISTORY ==========
    with tab_history:
        st.markdown(f"<h2>📈 {macro_name} - Historical Trend</h2>", unsafe_allow_html=True)
        st.caption(f"Data spanning the last {lookback_years} years")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines+markers",
                name=macro_name,
                line=dict(color='#3b82f6', width=3),
                marker=dict(size=5, color='#2563eb')
            )
        )

        fig.update_layout(
            height=500,
            template='plotly_white',
            paper_bgcolor='#ffffff',
            plot_bgcolor='#f8fafc',
            xaxis_title="Date",
            yaxis_title=meta['units'],
            hovermode='x unified',
            showlegend=False,
            margin=dict(l=10, r=10, t=40, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Latest Value", f"{series.iloc[-1]:.2f}")
        with col2:
            st.metric("Period High", f"{series.max():.2f}")
        with col3:
            st.metric("Period Low", f"{series.min():.2f}")
        with col4:
            st.metric("Average", f"{series.mean():.2f}")

    # ========== TAB 3: MARKET REACTION ==========
    with tab_impact:
        st.markdown("<h2>💥 Market Reaction Analysis</h2>", unsafe_allow_html=True)
        st.caption("How markets moved following each data release")

        col1, col2 = st.columns([2, 1])
        
        with col1:
            symbol = st.text_input(
                "Symbol to analyse",
                value="S&P 500",
                help="You can type a name or a ticker (case-insensitive). Examples: S&P 500, SPY, Nvidia, Gold, Brent, Bitcoin, USDZAR",
                placeholder="e.g. Nvidia, Gold, Bitcoin, USDZAR"
            )

        
        with col2:
            window_days = st.slider(
                "📅 Days After Release",
                1, 10, 3,
                help="Measure price movement this many trading days after release"
            )

        if not symbol:
            st.info("Enter a symbol to analyse market reactions")
            return

        resolved_symbol = resolve_symbol(symbol)
        st.caption(f"Resolved symbol: {resolved_symbol}")

        with st.spinner(f"Computing impact on {resolved_symbol}..."):
            impacts = compute_market_impact(releases, resolved_symbol, window_days=window_days)


        if impacts is None or impacts.empty:
            st.info(f"Could not compute market impact for {resolved_symbol}. The symbol may be invalid or lack historical data.")
            return

        st.success(f"Showing price movement in **{resolved_symbol}** from release date to {window_days} trading day(s) after")


        # Display table
        show = impacts.copy()
        show["Release Date"] = pd.to_datetime(show["Release Date"]).dt.strftime("%Y-%m-%d")
        show["Before Date"] = pd.to_datetime(show["Before Date"]).dt.strftime("%Y-%m-%d")
        show["After Date"] = pd.to_datetime(show["After Date"]).dt.strftime("%Y-%m-%d")
        
        for col in ["Move (%)", "Price Before", "Price After", "Macro Value", "Macro Change", "Macro % Change"]:
            if col in show.columns:
                show[col] = pd.to_numeric(show[col], errors='coerce').round(2)

        st.dataframe(
            show,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Release Date": st.column_config.TextColumn("📅 Release", width="medium"),
                "Move (%)": st.column_config.NumberColumn("📊 Move (%)", format="%.2f%%"),
                "Price Before": st.column_config.NumberColumn("Before ($)", format="$%.2f"),
                "Price After": st.column_config.NumberColumn("After ($)", format="$%.2f"),
            }
        )

        # ===== Bar chart of moves =====
        fig_imp = go.Figure()

        # Clean numeric series for plotting
        move_series = pd.to_numeric(impacts["Move (%)"], errors="coerce").fillna(0.0)

        # Vectorised colour mapping: blue for up, red for down/flat
        colors = np.where(move_series.to_numpy() > 0, "#00d4ff", "#ff4444")

        fig_imp.add_trace(
            go.Bar(
                x=pd.to_datetime(impacts["Release Date"]).dt.strftime("%Y-%m-%d"),
                y=move_series.to_numpy(),
                name="Price Movement",
                marker_color=colors,
                text=move_series.round(2).to_numpy(),
                texttemplate="%{text:.2f}%",
                textposition="outside",
            )
        )

        fig_imp.update_layout(
            height=450,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            xaxis_title="Release Date",
            yaxis_title="Price Movement (%)",
            showlegend=False,
            margin=dict(l=10, r=10, t=40, b=10),
        )

        st.plotly_chart(fig_imp, use_container_width=True)

        # ===== Summary stats =====
        move_series_num = pd.to_numeric(impacts["Move (%)"], errors="coerce")
        avg_move = float(move_series_num.mean())
        max_gain = float(move_series_num.max())
        max_loss = float(move_series_num.min())

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Move", f"{avg_move:.2f}%")
        with col2:
            st.metric("Largest Gain", f"{max_gain:.2f}%")
        with col3:
            st.metric("Largest Loss", f"{max_loss:.2f}%")
    
    with tab_curve:
        section_header(
            "Yield curve",
            "The yield curve shows Treasury yields across maturities. A normal curve slopes upward. Inversions (short rates above long rates) can signal tighter financial conditions.",
            level=2
        )

        yc_col1, yc_col2, yc_col3 = st.columns([1.2, 1, 1])

        with yc_col1:
            yc_lookback = st.slider("History (years)", 1, 20, 5, key="yc_years")
        with yc_col2:
            show_spreads = st.checkbox("Show common spreads", value=True)
        with yc_col3:
            show_history_lines = st.checkbox("Show history lines", value=False)

        yc_df, yc_err = load_yield_curve(lookback_years=yc_lookback)
        if yc_err:
            st.warning(yc_err)
        elif yc_df is None or yc_df.empty:
            st.info("No yield curve data available.")
        else:
            as_of, snap = latest_curve_snapshot(yc_df)

            if as_of is None or snap is None or snap.empty:
                st.info("No recent yield curve snapshot available.")
            else:
                st.caption(f"As of: {pd.to_datetime(as_of).strftime('%Y-%m-%d')} (FRED)")

                # Snapshot metrics
                mcols = st.columns(len(snap.index))
                for i, tenor in enumerate(snap.index):
                    with mcols[i]:
                        with st.container(border=True):
                            st.metric(tenor, f"{float(snap[tenor]):.2f}%")

                # Spreads (simple and useful)
                if show_spreads:
                    # Only compute spreads if both legs exist
                    spreads = []
                    def _spread(a, b, label):
                        if a in snap.index and b in snap.index:
                            spreads.append((label, float(snap[a] - snap[b])))

                    _spread("10Y", "2Y", "10Y–2Y")
                    _spread("10Y", "3M", "10Y–3M")
                    _spread("30Y", "10Y", "30Y–10Y")

                    if spreads:
                        scol1, scol2, scol3 = st.columns(3)
                        for col, (name, val) in zip([scol1, scol2, scol3], spreads[:3]):
                            with col:
                                with st.container(border=True):
                                    st.metric(name, f"{val:.2f}%")

                st.markdown("---")

                # Yield curve chart (cross-section)
                fig_curve = go.Figure()
                fig_curve.add_trace(
                    go.Scatter(
                        x=list(snap.index),
                        y=[float(v) for v in snap.values],
                        mode="lines+markers",
                        name="Latest curve"
                    )
                )

                # Optional: add a few historical curves for context
                if show_history_lines:
                    tmp = yc_df.dropna(how="all")
                    sample_dates = []
                    if len(tmp) > 0:
                        sample_dates = [
                            tmp.index[max(0, len(tmp) - 63)],   # ~3 months ago (trading days)
                            tmp.index[max(0, len(tmp) - 252)],  # ~1 year ago
                        ]
                        sample_dates = [d for d in sample_dates if d in tmp.index and d != as_of]

                    for d in sample_dates:
                        row = tmp.loc[d].dropna()
                        if row.empty:
                            continue
                        row = row.reindex(list(YIELD_CURVE_SERIES.keys()))
                        fig_curve.add_trace(
                            go.Scatter(
                                x=list(row.index),
                                y=[float(v) for v in row.values],
                                mode="lines",
                                name=pd.to_datetime(d).strftime("%Y-%m-%d"),
                                opacity=0.6
                            )
                        )

                fig_curve.update_layout(
                    title="Treasury yield curve",
                    xaxis_title="Maturity",
                    yaxis_title="Yield (%)",
                )
                style_plotly(fig_curve, height=460)
                st.plotly_chart(fig_curve, use_container_width=True)

                # History chart (tenor time series)
                st.markdown("---")
                section_header(
                    "Yield history",
                    "Time series of yields by maturity. Use this to see how the curve shifts (parallel moves) or twists (short vs long moving differently).",
                    level=3
                )

                tenors = list(YIELD_CURVE_SERIES.keys())
                selected_tenors = st.multiselect(
                    "Maturities to plot",
                    options=tenors,
                    default=["3M", "2Y", "10Y"],
                    key="yc_tenors"
                )

                if selected_tenors:
                    fig_hist = go.Figure()
                    for t in selected_tenors:
                        if t in yc_df.columns:
                            fig_hist.add_trace(
                                go.Scatter(
                                    x=yc_df.index,
                                    y=yc_df[t],
                                    mode="lines",
                                    name=t
                                )
                            )

                    fig_hist.update_layout(
                        title="Yields over time",
                        xaxis_title="Date",
                        yaxis_title="Yield (%)",
                    )
                    style_plotly(fig_hist, height=460)
                    st.plotly_chart(fig_hist, use_container_width=True)

                # Table (optional quick inspection)
                with st.expander("View recent yield data"):
                    recent = yc_df.tail(20).copy()
                    recent.index = pd.to_datetime(recent.index).strftime("%Y-%m-%d")
                    st.dataframe(recent, use_container_width=True)

        

# ============================================================================
# MAIN ROUTER
# ============================================================================

if page == "📈 Stocks":
    stocks_page()
elif page == "📰 News":
    news_page()
elif page == "🌍 Macro":
    macro_page()