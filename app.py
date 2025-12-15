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
    menu_items={"About": "Modern Investment Terminal - Your gateway to market intelligence"},
)

# ✅ IMPORTANT:
# - Do NOT force global text to light (#e5e7eb) if your app/charts are light.
# - Keep main content light so it stays readable on mobile.
st.markdown(
    """
<style>
/* ----- App background (LIGHT main) ----- */
[data-testid="stAppViewContainer"] {
  background: #f8fafc;
}
.main .block-container {
  padding-top: 1.25rem;
}

/* ----- Sidebar (DARK) ----- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
  border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * {
  color: #e2e8f0 !important;
}
[data-testid="stSidebar"] a { color: #93c5fd !important; }

/* ----- Headings (MAIN area) ----- */
h1, h2, h3, h4, h5, h6 {
  color: #0f172a !important;
  letter-spacing: -0.2px;
}
p, li, .stMarkdown {
  color: #334155;
}

/* ----- Cards / containers ----- */
div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMetric"]) {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 12px 14px;
}

/* ----- Metrics ----- */
[data-testid="stMetricValue"] {
  font-size: 1.8rem;
  font-weight: 800;
  color: #0f172a !important;
}
[data-testid="stMetricLabel"] {
  color: #475569 !important;
  font-weight: 700;
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.6px;
}
[data-testid="stMetricDelta"] {
  font-weight: 700;
}

/* ----- Buttons ----- */
.stButton>button {
  background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
  color: white;
  border: none;
  border-radius: 10px;
  padding: 0.6rem 1.2rem;
  font-weight: 700;
  transition: all 0.2s ease;
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
}
.stButton>button:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 18px rgba(37, 99, 235, 0.35);
}

/* ----- Tabs ----- */
.stTabs [data-baseweb="tab"] {
  background: #ffffff;
  border-radius: 10px;
  padding: 10px 16px;
  color: #475569;
  font-weight: 700;
  border: 1px solid #e2e8f0;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
  color: white !important;
  border: 1px solid #2563eb;
}

/* ----- Expanders ----- */
.streamlit-expanderHeader {
  background: #ffffff;
  border-radius: 10px;
  font-weight: 700;
  color: #0f172a;
  border: 1px solid #e2e8f0;
}

/* ----- Alerts ----- */
.stAlert {
  border-radius: 10px;
  border-left: 4px solid #3b82f6;
}

/* ----- Reduce weird dark-mode inversion issues on mobile browsers ----- */
html { color-scheme: light; }

</style>
""",
    unsafe_allow_html=True,
)

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
        "icon": "💵",
    },
    "Unemployment Rate": {
        "id": "UNRATE",
        "units": "%",
        "description": "Percentage of labour force without jobs",
        "icon": "👥",
    },
    "Federal Funds Rate": {
        "id": "FEDFUNDS",
        "units": "%",
        "description": "Target interest rate set by the Fed",
        "icon": "📊",
    },
    "Nonfarm Payrolls": {
        "id": "PAYEMS",
        "units": "Thousands",
        "description": "Total employed in non-agricultural sectors",
        "icon": "💼",
    },
}

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
# HELPERS
# ============================================================================
def _nearest_on_or_before(df: pd.DataFrame, target_date: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Nearest available index date on or before target_date."""
    if df is None or df.empty:
        return None
    idx = df.index[df.index <= target_date]
    if len(idx) == 0:
        return None
    return idx[-1]


def compute_spread_series(yc_df: pd.DataFrame, long_tenor: str, short_tenor: str) -> pd.Series:
    """Return time series of (long - short) where possible."""
    if yc_df is None or yc_df.empty:
        return pd.Series(dtype=float)
    if long_tenor not in yc_df.columns or short_tenor not in yc_df.columns:
        return pd.Series(dtype=float)
    s = yc_df[long_tenor] - yc_df[short_tenor]
    return s.dropna()


def classify_regime(spread_value: float, flat_band: float = 0.25) -> str:
    """
    Classify curve regime by spread.
    flat_band is in percentage points (e.g. 0.25 = 25 bps).
    """
    if spread_value > flat_band:
        return "Normal"
    if spread_value < -flat_band:
        return "Inverted"
    return "Flat"


def regime_duration_days(spread_series: pd.Series, flat_band: float = 0.25) -> int:
    """How many consecutive observations (days) we’ve been in the current regime."""
    if spread_series is None or spread_series.empty:
        return 0
    s = spread_series.dropna()
    if s.empty:
        return 0

    current = classify_regime(float(s.iloc[-1]), flat_band=flat_band)

    # Walk backwards until regime changes
    count = 0
    for v in reversed(s.values):
        if classify_regime(float(v), flat_band=flat_band) == current:
            count += 1
        else:
            break
    return count


def curve_curvature(snap: pd.Series) -> Optional[float]:
    """
    A simple curvature measure:
      2*5Y - (2Y + 10Y)
    Positive = belly high relative to wings.
    """
    needed = {"2Y", "5Y", "10Y"}
    if snap is None or snap.empty or not needed.issubset(set(snap.index)):
        return None
    return float(2.0 * snap["5Y"] - (snap["2Y"] + snap["10Y"]))


def compute_yield_changes(yc_df: pd.DataFrame, as_of: pd.Timestamp, months_back: int) -> Optional[pd.Series]:
    """
    Cross-section changes: yield(as_of) - yield(nearest date months_back ago).
    Returns Series indexed by tenor.
    """
    if yc_df is None or yc_df.empty:
        return None
    as_of = pd.to_datetime(as_of)
    past_target = as_of - relativedelta(months=months_back)
    past_date = _nearest_on_or_before(yc_df.dropna(how="all"), past_target)
    if past_date is None:
        return None

    snap_now = yc_df.loc[as_of].dropna()
    snap_past = yc_df.loc[past_date].dropna()

    tenors = [t for t in YIELD_CURVE_SERIES.keys() if t in snap_now.index and t in snap_past.index]
    if not tenors:
        return None

    return (snap_now[tenors] - snap_past[tenors]).astype(float)


def forward_rate_proxy(y1: float, y2: float, t1: float, t2: float) -> Optional[float]:
    """
    Simple annualised forward rate proxy from spot yields.
    y1,y2 in decimals (e.g. 0.045), t in years.
    Uses: (1+y2)^t2 / (1+y1)^t1 = (1+f)^(t2-t1)
    """
    if any(pd.isna(x) for x in [y1, y2, t1, t2]) or t2 <= t1:
        return None
    try:
        lhs = (1.0 + y2) ** t2 / (1.0 + y1) ** t1
        f = lhs ** (1.0 / (t2 - t1)) - 1.0
        return float(f)
    except Exception:
        return None

def style_plotly(fig: go.Figure, height: Optional[int] = None) -> go.Figure:
    """
    Force consistent light styling (white background + dark text),
    independent of device/browser dark mode.
    """
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a", size=13),
        margin=dict(l=14, r=14, t=55, b=14),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", font_color="#0f172a"),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(15,23,42,0.10)",
            borderwidth=1,
            font=dict(color="#0f172a"),
        ),
        title=dict(font=dict(color="#0f172a")),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(15,23,42,0.08)",
        zeroline=False,
        tickfont=dict(color="#0f172a"),
        title_font=dict(color="#0f172a"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(15,23,42,0.08)",
        zeroline=False,
        tickfont=dict(color="#0f172a"),
        title_font=dict(color="#0f172a"),
    )

    if height is not None:
        fig.update_layout(height=height)

    return fig


SYMBOL_ALIASES = {
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
    if not text:
        return ""
    raw = text.strip()
    key = raw.lower()
    if key in SYMBOL_ALIASES:
        return SYMBOL_ALIASES[key]
    key2 = key.replace("&", "and").replace(".", "").replace(",", "").replace("  ", " ").strip()
    if key2 in SYMBOL_ALIASES:
        return SYMBOL_ALIASES[key2]
    return raw.upper()


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_mas(close: pd.Series, short_window: int = 20, long_window: int = 50):
    return close.rolling(short_window).mean(), close.rolling(long_window).mean()


def detect_crossovers(short_ma: pd.Series, long_ma: pd.Series):
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
        data[tenor] = (s.dropna() if s is not None else pd.Series(dtype=float))

    df = pd.DataFrame(data).sort_index()
    if df.empty:
        return None, "No yield curve data returned."
    return df, None


def latest_curve_snapshot(df: pd.DataFrame):
    if df is None or df.empty:
        return None, None
    tmp = df.dropna(how="all")
    if tmp.empty:
        return None, None
    as_of = tmp.index[-1]
    snap = tmp.loc[as_of].dropna()
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
    if releases is None or releases.empty:
        return None

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


def section_header(title: str, help_text: str, level: int = 2):
    safe = help_text.replace('"', "'")
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:10px;">
          <h{level} style="margin:0;">{title}</h{level}>
          <span title="{safe}" style="display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;border-radius:999px;background:#e2e8f0;color:#0f172a;font-weight:800;">?</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.markdown("## 🎯 Navigation")
page = st.sidebar.radio(
    "Select Dashboard",
    ["📈 Stocks", "🌍 Macro"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Quick Tips")
st.sidebar.info(
    """
**Stocks**: Compare multiple tickers with advanced charts and fundamentals

**Macro**: Track economic indicators and market reactions
"""
)


# ============================================================================
# STOCKS PAGE
# ============================================================================

def stocks_page():
    st.markdown("<h1>📈 Stock Market Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("Compare tickers, analyse performance, and explore market correlations")
    st.markdown("---")

    col1, col2 = st.columns([3, 1])

    with col1:
        tickers_input = st.text_input(
            "🎯 Enter Stock Tickers",
            value="AAPL, MSFT, NVDA",
            help="Separate multiple tickers with commas (e.g., AAPL, MSFT, TSLA, SPY). You can also type names like Nvidia.",
            placeholder="AAPL, MSFT, NVDA",
        )

    with col2:
        st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
        chart_mode = st.selectbox(
            "📊 Chart Type",
            ["📉 Line Chart", "🕯️ Candlestick"],
            help="Candlesticks work best with a single ticker",
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        period = st.selectbox("📅 Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=3)
    with c2:
        interval = st.selectbox("⏱️ Interval", ["1d", "1wk", "1mo"], index=0)
    with c3:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)

    raw_inputs = [t for t in tickers_input.split(",") if t.strip()]
    tickers = [resolve_symbol(t) for t in raw_inputs]

    if not tickers:
        st.info("👆 Enter at least one ticker to begin your analysis")
        return

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

    returns = closes.pct_change().dropna()
    if returns.empty:
        st.info("⚠️ Insufficient data to compute returns")
        return

    total_return = (closes.iloc[-1] / closes.iloc[0] - 1.0) * 100
    daily_vol = returns.std() * 100

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
                    st.metric(label=t, value=f"${latest:,.2f}", delta=f"{ret:.2f}%", help=f"Daily volatility: {vol:.2f}%")

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

            selected = st.multiselect("Select metrics to display", labels, default=default_labels)

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

    st.markdown("<h2>📈 Price Action & Volume</h2>", unsafe_allow_html=True)

    single_ticker = len(closes.columns) == 1
    use_candlestick = ("Candlestick" in chart_mode) and single_ticker

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
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            fig.add_trace(
                go.Candlestick(
                    x=hist.index,
                    open=hist["Open"],
                    high=hist["High"],
                    low=hist["Low"],
                    close=hist["Close"],
                    name=t,
                ),
                row=1,
                col=1,
            )
            if "Volume" in hist.columns:
                fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], name="Volume", opacity=0.6), row=2, col=1)

            fig.update_layout(showlegend=False, xaxis_title="Date", yaxis_title="Price")
            style_plotly(fig, height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            use_candlestick = False

    if not use_candlestick:
        fig_price = go.Figure()
        colours = ["#2563eb", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16"]

        for idx, col in enumerate(closes.columns):
            fig_price.add_trace(
                go.Scatter(
                    x=closes.index,
                    y=closes[col],
                    mode="lines",
                    name=col,
                    line=dict(color=colours[idx % len(colours)], width=3),
                )
            )

        fig_price.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")
        style_plotly(fig_price, height=460)
        st.plotly_chart(fig_price, use_container_width=True)

        if vols is not None and single_ticker:
            vol_series = vols[closes.columns[0]] if isinstance(vols, pd.DataFrame) else vols
            fig_vol = go.Figure(data=[go.Bar(x=vol_series.index, y=vol_series.values, name="Volume", opacity=0.7)])
            fig_vol.update_layout(xaxis_title="Date", yaxis_title="Volume", showlegend=False)
            style_plotly(fig_vol, height=240)
            st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("<h2>📊 Relative Performance</h2>", unsafe_allow_html=True)
    st.caption("Normalised to 100 at start date for easy comparison")

    rebased = closes / closes.iloc[0] * 100.0
    fig_norm = go.Figure()
    colours = ["#2563eb", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16"]

    for idx, col in enumerate(rebased.columns):
        fig_norm.add_trace(
            go.Scatter(x=rebased.index, y=rebased[col], mode="lines", name=col, line=dict(color=colours[idx % len(colours)], width=3))
        )

    fig_norm.update_layout(xaxis_title="Date", yaxis_title="Indexed Performance (Start = 100)")
    style_plotly(fig_norm, height=420)
    st.plotly_chart(fig_norm, use_container_width=True)

    st.markdown("<h2>🔗 Correlation Matrix</h2>", unsafe_allow_html=True)
    st.caption("Daily return correlations between selected tickers")

    corr = returns.corr().round(2)

    fig_corr = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
            text=corr.values,
            texttemplate="%{text:.2f}",
            textfont={"size": 12, "color": "#0f172a"},  # ✅ black text (readable)
        )
    )
    style_plotly(fig_corr, height=480)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("<h2>📐 Technical Analysis</h2>", unsafe_allow_html=True)
    st.caption("RSI and moving averages for a selected ticker")

    tech_col1, tech_col2, tech_col3 = st.columns(3)
    with tech_col1:
        ta_ticker = st.selectbox("Ticker for TA", closes.columns.tolist(), index=0)
    with tech_col2:
        rsi_window = st.number_input("RSI window", min_value=5, max_value=50, value=14, step=1)
    with tech_col3:
        short_window = st.number_input("Short MA", min_value=5, max_value=200, value=20, step=1)

    long_window = st.number_input("Long MA", min_value=10, max_value=400, value=50, step=5)

    close_series = closes[ta_ticker].dropna()
    if close_series.empty:
        st.info("No price data available for technicals.")
        return

    rsi = compute_rsi(close_series, window=rsi_window)
    short_ma, long_ma = compute_mas(close_series, short_window=short_window, long_window=long_window)
    golden, death = detect_crossovers(short_ma, long_ma)

    st.markdown(f"### Price & Moving Averages – {ta_ticker}")

    fig_ta_price = go.Figure()
    fig_ta_price.add_trace(go.Scatter(x=close_series.index, y=close_series.values, mode="lines", name="Close", line=dict(color="#0f172a", width=2)))
    fig_ta_price.add_trace(go.Scatter(x=short_ma.index, y=short_ma.values, mode="lines", name=f"MA {short_window}", line=dict(color="#2563eb", width=2)))
    fig_ta_price.add_trace(go.Scatter(x=long_ma.index, y=long_ma.values, mode="lines", name=f"MA {long_window}", line=dict(color="#f97316", width=2)))

    golden_dates = close_series.index[golden.fillna(False)]
    death_dates = close_series.index[death.fillna(False)]

    if len(golden_dates) > 0:
        last_golden = golden_dates[-1]
        fig_ta_price.add_trace(go.Scatter(x=[last_golden], y=[close_series.loc[last_golden]], mode="markers", marker=dict(color="#22c55e", size=12, symbol="triangle-up"), name="Last golden cross"))

    if len(death_dates) > 0:
        last_death = death_dates[-1]
        fig_ta_price.add_trace(go.Scatter(x=[last_death], y=[close_series.loc[last_death]], mode="markers", marker=dict(color="#ef4444", size=12, symbol="triangle-down"), name="Last death cross"))

    fig_ta_price.update_layout(xaxis_title="Date", yaxis_title="Price")
    style_plotly(fig_ta_price, height=460)
    st.plotly_chart(fig_ta_price, use_container_width=True)

    st.markdown(f"### RSI – {ta_ticker}")

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=rsi.index, y=rsi.values, mode="lines", name=f"RSI ({rsi_window})", line=dict(color="#6366f1", width=2)))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef4444")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="#22c55e")
    fig_rsi.update_layout(xaxis_title="Date", yaxis_title="RSI", yaxis=dict(range=[0, 100]), showlegend=False)
    style_plotly(fig_rsi, height=280)
    st.plotly_chart(fig_rsi, use_container_width=True)


# ============================================================================
# MACRO PAGE
# ============================================================================

def macro_page():
    st.markdown("<h1>🌍 Macroeconomic Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("Track key economic indicators and analyse market reactions to data releases")
    st.markdown("---")

    fred, fred_err = _get_fred()
    if fred_err:
        st.warning(
            """
⚠️ **FRED API Setup Required**

To use the Macro Dashboard, you need to:
1. Add your `FRED_API_KEY` to Streamlit secrets
2. Ensure the `fredapi` package is installed
"""
        )
        return

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        macro_name = st.selectbox(
            "📊 Select Economic Indicator",
            list(MACRO_SERIES.keys()),
            format_func=lambda x: f"{MACRO_SERIES[x]['icon']} {x}",
            index=0,
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
        st.info("📭 Could not load macro data.")
        return

    releases = build_release_table(series, last_n=last_n)

    tab_calendar, tab_history, tab_impact, tab_curve = st.tabs(["Release calendar", "Historical data", "Market impact", "Yield curve"])

    with tab_calendar:
        st.markdown(f"<h2>📅 Recent {macro_name} Releases</h2>", unsafe_allow_html=True)
        if releases is None or releases.empty:
            st.info("📭 No release data available")
        else:
            nice = releases.copy()
            nice["Date"] = pd.to_datetime(nice["Date"]).dt.strftime("%Y-%m-%d")
            for col in ["Value", "Previous", "Change", "% Change"]:
                if col in nice.columns:
                    nice[col] = pd.to_numeric(nice[col], errors="coerce").round(2)
            st.dataframe(nice, hide_index=True, use_container_width=True)

    with tab_history:
        st.markdown(f"<h2>📈 {macro_name} - Historical Trend</h2>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines+markers", name=macro_name, line=dict(width=3)))
        fig.update_layout(xaxis_title="Date", yaxis_title=meta["units"], showlegend=False)
        style_plotly(fig, height=520)
        st.plotly_chart(fig, use_container_width=True)

    with tab_impact:
        st.markdown("<h2>💥 Market Reaction Analysis</h2>", unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            symbol = st.text_input(
                "Symbol to analyse",
                value="S&P 500",
                help="Type a name or ticker. Examples: S&P 500, SPY, Nvidia, Gold, Bitcoin, USDZAR",
            )
        with col2:
            window_days = st.slider("📅 Days After Release", 1, 10, 3)

        if not symbol:
            st.info("Enter a symbol to analyse market reactions")
            return

        resolved_symbol = resolve_symbol(symbol)
        st.caption(f"Resolved symbol: {resolved_symbol}")

        with st.spinner(f"Computing impact on {resolved_symbol}..."):
            impacts = compute_market_impact(releases, resolved_symbol, window_days=window_days)

        if impacts is None or impacts.empty:
            st.info(f"Could not compute market impact for {resolved_symbol}.")
            return

        show = impacts.copy()
        show["Release Date"] = pd.to_datetime(show["Release Date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(show, hide_index=True, use_container_width=True)

        move_series = pd.to_numeric(impacts["Move (%)"], errors="coerce").fillna(0.0)
        bar_colours = np.where(move_series.to_numpy() > 0, "#22c55e", "#ef4444")

        fig_imp = go.Figure()
        fig_imp.add_trace(
            go.Bar(
                x=pd.to_datetime(impacts["Release Date"]).dt.strftime("%Y-%m-%d"),
                y=move_series.to_numpy(),
                marker_color=bar_colours,
                text=move_series.round(2).to_numpy(),
                texttemplate="%{text:.2f}%",
                textposition="outside",
            )
        )
        fig_imp.update_layout(xaxis_title="Release Date", yaxis_title="Price Movement (%)", showlegend=False)
        style_plotly(fig_imp, height=480)
        st.plotly_chart(fig_imp, use_container_width=True)

    with tab_curve:
        section_header(
            "Yield curve",
            "The yield curve shows Treasury yields across maturities. A normal curve slopes upward. Inversions (short rates above long rates) can signal tighter financial conditions.",
            level=2
        )

        yc_col1, yc_col2, yc_col3, yc_col4 = st.columns([1.2, 1, 1, 1])

        with yc_col1:
            yc_lookback = st.slider("History (years)", 1, 20, 5, key="yc_years")
        with yc_col2:
            show_spreads = st.checkbox("Show spreads + regime", value=True)
        with yc_col3:
            show_history_lines = st.checkbox("Show history lines", value=False)
        with yc_col4:
            show_spread_history = st.checkbox("Show spread history", value=True)

        flat_band_bps = st.slider("Flat band (bps) for regime", 5, 50, 25, step=5)
        flat_band = flat_band_bps / 100.0  # bps -> percentage points

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
                as_of = pd.to_datetime(as_of)
                st.caption(f"As of: {as_of.strftime('%Y-%m-%d')} (FRED)")

                # ========= Snapshot metrics =========
                mcols = st.columns(len(snap.index))
                for i, tenor in enumerate(snap.index):
                    with mcols[i]:
                        with st.container(border=True):
                            st.metric(tenor, f"{float(snap[tenor]):.2f}%")

                # ========= Spreads + regime + duration =========
                if show_spreads:
                    spreads = {}
                    def _try_spread(a, b, label):
                        if a in snap.index and b in snap.index:
                            spreads[label] = float(snap[a] - snap[b])

                    _try_spread("10Y", "2Y", "10Y–2Y")
                    _try_spread("10Y", "3M", "10Y–3M")
                    _try_spread("30Y", "10Y", "30Y–10Y")

                    scol1, scol2, scol3, scol4 = st.columns(4)

                    # 10Y–2Y
                    with scol1:
                        val = spreads.get("10Y–2Y", np.nan)
                        regime = classify_regime(val, flat_band=flat_band) if pd.notna(val) else "–"
                        st.metric("10Y–2Y", f"{val:.2f}%" if pd.notna(val) else "–", regime)

                    # 10Y–3M
                    with scol2:
                        val = spreads.get("10Y–3M", np.nan)
                        regime = classify_regime(val, flat_band=flat_band) if pd.notna(val) else "–"
                        st.metric("10Y–3M", f"{val:.2f}%" if pd.notna(val) else "–", regime)

                    # curvature
                    with scol3:
                        curv = curve_curvature(snap)
                        st.metric("Curvature (2*5Y-(2Y+10Y))", f"{curv:.2f}%" if curv is not None else "–")

                    # regime duration (based on 10Y–2Y)
                    with scol4:
                        sp_series = compute_spread_series(yc_df, "10Y", "2Y")
                        days = regime_duration_days(sp_series, flat_band=flat_band)
                        st.metric("Current regime duration", f"{days} obs", "based on 10Y–2Y")

                st.markdown("---")

                # ========= Yield curve chart (cross-section) =========
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
                            tmp.index[max(0, len(tmp) - 63)],   # ~3 months
                            tmp.index[max(0, len(tmp) - 252)],  # ~1 year
                        ]
                        sample_dates = [d for d in sample_dates if d in tmp.index and pd.to_datetime(d) != as_of]

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

                # ========= 1M / 3M changes (cross-section) =========
                ch1 = compute_yield_changes(yc_df, as_of, months_back=1)
                ch3 = compute_yield_changes(yc_df, as_of, months_back=3)

                if (ch1 is not None and not ch1.empty) or (ch3 is not None and not ch3.empty):
                    st.markdown("---")
                    section_header(
                        "Yield changes",
                        "How much each tenor moved versus 1 month ago and 3 months ago (in basis points). This makes tightening/easing obvious.",
                        level=3
                    )

                    # Build a small table
                    change_df = pd.DataFrame(index=[t for t in YIELD_CURVE_SERIES.keys() if t in snap.index])
                    change_df["Level (%)"] = [float(snap[t]) for t in change_df.index]

                    if ch1 is not None and not ch1.empty:
                        change_df["Δ 1M (bps)"] = [float(ch1.get(t, np.nan) * 100.0) for t in change_df.index]
                    if ch3 is not None and not ch3.empty:
                        change_df["Δ 3M (bps)"] = [float(ch3.get(t, np.nan) * 100.0) for t in change_df.index]

                    st.dataframe(
                        change_df.round(2).reset_index().rename(columns={"index": "Tenor"}),
                        hide_index=True,
                        use_container_width=True
                    )

                # ========= Forward-rate proxies =========
                st.markdown("---")
                section_header(
                    "Forward-rate proxies",
                    "Rough implied forward rates using spot yields. Not perfect, but useful for expectations (e.g., what the curve implies about future short rates).",
                    level=3
                )

                # Map tenors to years for the proxy
                tenor_years = {"1Y": 1.0, "2Y": 2.0, "5Y": 5.0, "10Y": 10.0}

                fcols = st.columns(3)

                # 1Y1Y (from 1Y and 2Y)
                with fcols[0]:
                    if "1Y" in snap.index and "2Y" in snap.index:
                        f = forward_rate_proxy(float(snap["1Y"]) / 100.0, float(snap["2Y"]) / 100.0, 1.0, 2.0)
                        st.metric("1Y1Y forward", f"{(f*100):.2f}%" if f is not None else "–")
                    else:
                        st.metric("1Y1Y forward", "–")

                # 2Y3Y (from 2Y and 5Y)
                with fcols[1]:
                    if "2Y" in snap.index and "5Y" in snap.index:
                        f = forward_rate_proxy(float(snap["2Y"]) / 100.0, float(snap["5Y"]) / 100.0, 2.0, 5.0)
                        st.metric("2Y3Y forward", f"{(f*100):.2f}%" if f is not None else "–")
                    else:
                        st.metric("2Y3Y forward", "–")

                # 5Y5Y (from 5Y and 10Y)
                with fcols[2]:
                    if "5Y" in snap.index and "10Y" in snap.index:
                        f = forward_rate_proxy(float(snap["5Y"]) / 100.0, float(snap["10Y"]) / 100.0, 5.0, 10.0)
                        st.metric("5Y5Y forward", f"{(f*100):.2f}%" if f is not None else "–")
                    else:
                        st.metric("5Y5Y forward", "–")

                # ========= Spread history with inversion shading =========
                if show_spread_history:
                    st.markdown("---")
                    section_header(
                        "Spread history",
                        "Tracks key spreads over time. Shaded regions indicate inversion (< 0). This gives context: not just inverted, but for how long and how deep.",
                        level=3
                    )

                    s_10_2 = compute_spread_series(yc_df, "10Y", "2Y")
                    s_10_3m = compute_spread_series(yc_df, "10Y", "3M")

                    fig_sp = go.Figure()

                    if not s_10_2.empty:
                        fig_sp.add_trace(go.Scatter(x=s_10_2.index, y=s_10_2.values, mode="lines", name="10Y–2Y"))

                    if not s_10_3m.empty:
                        fig_sp.add_trace(go.Scatter(x=s_10_3m.index, y=s_10_3m.values, mode="lines", name="10Y–3M"))

                    # zero line
                    fig_sp.add_hline(y=0, line_dash="dash", line_color="rgba(17,24,39,0.35)")

                    # Shade inverted periods using 10Y–2Y where available, otherwise 10Y–3M
                    shade_base = s_10_2 if not s_10_2.empty else s_10_3m
                    if shade_base is not None and not shade_base.empty:
                        inverted = (shade_base < 0).astype(int)
                        # find contiguous inverted segments
                        inv_idx = shade_base.index
                        start = None
                        for i in range(len(inv_idx)):
                            if inverted.iloc[i] == 1 and start is None:
                                start = inv_idx[i]
                            if (inverted.iloc[i] == 0 or i == len(inv_idx)-1) and start is not None:
                                end = inv_idx[i] if inverted.iloc[i] == 0 else inv_idx[i]
                                fig_sp.add_vrect(
                                    x0=start, x1=end,
                                    fillcolor="rgba(239,68,68,0.10)",
                                    line_width=0
                                )
                                start = None

                    fig_sp.update_layout(
                        title="Key yield-curve spreads",
                        xaxis_title="Date",
                        yaxis_title="Spread (%)",
                    )
                    style_plotly(fig_sp, height=420)
                    st.plotly_chart(fig_sp, use_container_width=True)

                # ========= Table (quick inspection) =========
                with st.expander("View recent yield data"):
                    recent = yc_df.tail(20).copy()
                    recent.index = pd.to_datetime(recent.index).strftime("%Y-%m-%d")
                    st.dataframe(recent, use_container_width=True)


# ============================================================================
# MAIN ROUTER
# ============================================================================

if page == "📈 Stocks":
    stocks_page()
elif page == "🌍 Macro":
    macro_page()
