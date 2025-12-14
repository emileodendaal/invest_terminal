import os
from datetime import datetime, timezone, date
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from core.macrocore.storage.db import (
    init_db,
    get_db_path,
    data_health,
    set_meta,
    utc_now_iso,
    connect,
)

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Macro Tracker",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# FRED config
# ============================================================
FRED_BASE = "https://api.stlouisfed.org/fred"

SERIES_YIELDS = {
    "DGS10": "US 10Y Treasury (%)",
    "DGS2": "US 2Y Treasury (%)",
    "DGS3MO": "US 3M Treasury (%)",
}

SERIES_INFLATION = {
    "CPIAUCSL": "US CPI-U (SA, index)",
}

# ============================================================
# Asset aliases
# ============================================================
ASSET_ALIASES = {
    "nvidia": "NVDA",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "meta": "META",
    "tesla": "TSLA",
    "s&p 500": "SPY",
    "sp500": "SPY",
    "nasdaq 100": "QQQ",
    "gold": "GLD",
    "us 20y bonds": "TLT",
    "bitcoin": "BTC-USD",
    "btc": "BTC-USD",
    "usd/zar": "USDZAR=X",
    "usd zar": "USDZAR=X",
}

def resolve_to_ticker(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    return ASSET_ALIASES.get(s.lower(), s.upper())

def fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")

def get_fred_api_key() -> str:
    try:
        k = st.secrets.get("FRED_API_KEY")
        if k:
            return str(k)
    except Exception:
        pass
    k = os.getenv("FRED_API_KEY")
    if not k:
        raise ValueError("Missing FRED API key. Put FRED_API_KEY in .streamlit/secrets.toml or export it in terminal.")
    return k

# ============================================================
# DB init
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = get_db_path(PROJECT_ROOT)
init_db(DB_PATH)

# ============================================================
# DB helpers
# ============================================================
def upsert_macro_observation(db_path: Path, series_id: str, obs_date: str, value: float, source: str = "FRED") -> None:
    fetched_at = utc_now_iso()
    with connect(db_path) as con:
        con.execute(
            """
            INSERT INTO macro_observations(series_id, obs_date, value, source, fetched_at_utc)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(series_id, obs_date)
            DO UPDATE SET value=excluded.value, source=excluded.source, fetched_at_utc=excluded.fetched_at_utc
            """,
            (series_id, obs_date, float(value), source, fetched_at),
        )

def latest_two_values(db_path: Path, series_id: str):
    """
    Returns (latest_date, latest_val, prev_date, prev_val) or (None, None, None, None)
    """
    with connect(db_path) as con:
        rows = con.execute(
            """
            SELECT obs_date, value
            FROM macro_observations
            WHERE series_id=?
            ORDER BY obs_date DESC
            LIMIT 2
            """,
            (series_id,),
        ).fetchall()

    if not rows:
        return None, None, None, None

    latest_date, latest_val = rows[0]["obs_date"], float(rows[0]["value"])
    if len(rows) > 1:
        prev_date, prev_val = rows[1]["obs_date"], float(rows[1]["value"])
    else:
        prev_date, prev_val = None, None

    return latest_date, latest_val, prev_date, prev_val

def load_series_df(db_path: Path, series_id: str) -> pd.DataFrame:
    with connect(db_path) as con:
        rows = con.execute(
            """
            SELECT obs_date, value
            FROM macro_observations
            WHERE series_id=?
            ORDER BY obs_date ASC
            """,
            (series_id,),
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["obs_date", "value"])

    df = pd.DataFrame(rows, columns=["obs_date", "value"])
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna()

def safe_latest_prev(latest, prev):
    """
    If prev is missing (only 1 observation), set prev = latest so deltas become 0.
    If latest missing, return (0,0).
    """
    if latest is None:
        return 0.0, 0.0
    if prev is None:
        return float(latest), float(latest)
    return float(latest), float(prev)

# ============================================================
# CPI maths (YoY / MoM from index level)
# ============================================================
def cpi_changes_from_df(df: pd.DataFrame) -> dict:
    """
    df must contain columns: obs_date (datetime), value (float) for CPI index.
    Returns latest level + MoM% + YoY%.
    """
    if df.empty or len(df) < 2:
        return {"level": None, "mom": None, "yoy": None, "date": None}

    df = df.sort_values("obs_date").reset_index(drop=True)

    latest = df.iloc[-1]
    level = float(latest["value"])
    d = latest["obs_date"].date().isoformat()

    # MoM: compare to previous observation (monthly series -> previous month)
    prev_level = float(df.iloc[-2]["value"])
    mom = (level / prev_level - 1.0) * 100.0 if prev_level != 0 else None

    # YoY: compare to 12 observations back if available
    if len(df) >= 13:
        level_12 = float(df.iloc[-13]["value"])
        yoy = (level / level_12 - 1.0) * 100.0 if level_12 != 0 else None
    else:
        yoy = None

    return {"level": level, "mom": mom, "yoy": yoy, "date": d}

# ============================================================
# FRED fetch helpers
# ============================================================
def fred_fetch_latest(series_id: str) -> tuple[str, float]:
    api_key = get_fred_api_key()
    url = f"{FRED_BASE}/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 10,
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    for obs in data.get("observations", []):
        d = obs.get("date")
        v = obs.get("value")
        if not d or v in (None, "."):
            continue
        try:
            return d, float(v)
        except ValueError:
            continue

    raise RuntimeError(f"No valid observations found for {series_id}")

def fred_fetch_range(series_id: str, start_date: str) -> list[tuple[str, float]]:
    api_key = get_fred_api_key()
    url = f"{FRED_BASE}/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "sort_order": "asc",
    }
    r = requests.get(url, params=params, timeout=45)
    r.raise_for_status()
    data = r.json()

    out: list[tuple[str, float]] = []
    for obs in data.get("observations", []):
        d = obs.get("date")
        v = obs.get("value")
        if not d or v in (None, "."):
            continue
        try:
            out.append((d, float(v)))
        except ValueError:
            continue
    return out

def years_ago_iso(years: int) -> str:
    today = date.today()
    return f"{today.year - years:04d}-{today.month:02d}-{today.day:02d}"

# ============================================================
# Sidebar UI
# ============================================================
with st.sidebar:
    st.title("🌍 Macro Tracker")

    st.subheader("Assets")
    if "assets" not in st.session_state:
        st.session_state.assets = ["SPY", "QQQ", "GLD", "TLT"]

    asset_input = st.text_input("Add asset (name or ticker)", placeholder="e.g. NVIDIA or NVDA")

    c_add, c_clear = st.columns(2)
    with c_add:
        add_clicked = st.button("➕ Add", use_container_width=True)
    with c_clear:
        clear_clicked = st.button("🗑️ Clear", use_container_width=True)

    if add_clicked and asset_input.strip():
        ticker = resolve_to_ticker(asset_input)
        if ticker and ticker not in st.session_state.assets:
            st.session_state.assets.append(ticker)

    if clear_clicked:
        st.session_state.assets = []

    assets = st.multiselect(
        "Selected assets",
        options=sorted(set(st.session_state.assets)),
        default=st.session_state.assets,
    )

    st.subheader("Event window")
    event_window = st.selectbox("Reaction window", options=["30m", "2h", "1d", "5d"], index=2)

    st.subheader("Data actions")

    refresh_latest = st.button("🔄 Refresh latest (yields + CPI)", use_container_width=True)

    backfill_years = st.selectbox("Backfill history (years)", options=[1, 2, 5, 10], index=2)
    backfill = st.button(f"⬇️ Backfill last {backfill_years} years (yields + CPI)", use_container_width=True)

    st.caption("v0.5 — yield curve + CPI (MoM/YoY) operational")

# ============================================================
# Main header
# ============================================================
st.markdown("# 🌍 Macro Tracker")
st.caption("Now includes: real yield curve + real CPI level with MoM/YoY computed from your DB.")

# ============================================================
# Actions
# ============================================================
if refresh_latest:
    try:
        fetched = []

        # Yields
        for sid in SERIES_YIELDS.keys():
            d, v = fred_fetch_latest(sid)
            upsert_macro_observation(DB_PATH, sid, d, v, source="FRED")
            fetched.append(f"{sid} {d}={v}")

        # CPI
        for sid in SERIES_INFLATION.keys():
            d, v = fred_fetch_latest(sid)
            upsert_macro_observation(DB_PATH, sid, d, v, source="FRED")
            fetched.append(f"{sid} {d}={v}")

        set_meta(DB_PATH, "last_update_utc", utc_now_iso())
        st.toast(" | ".join(fetched), icon="✅")
    except Exception as e:
        st.error(f"Refresh failed: {e}")

if backfill:
    try:
        start = years_ago_iso(backfill_years)
        with st.spinner(f"Backfilling from {start} ..."):
            total = 0

            for sid in list(SERIES_YIELDS.keys()) + list(SERIES_INFLATION.keys()):
                obs = fred_fetch_range(sid, start)
                for d, v in obs:
                    upsert_macro_observation(DB_PATH, sid, d, v, source="FRED")
                total += len(obs)

        set_meta(DB_PATH, "last_update_utc", utc_now_iso())
        st.toast(f"Backfill complete. Inserted/updated ~{total} observations.", icon="✅")
    except Exception as e:
        st.error(f"Backfill failed: {e}")

# ============================================================
# Data health
# ============================================================
health = data_health(DB_PATH)
st.markdown("### Data health")
h1, h2, h3, h4 = st.columns(4)
h1.metric("Macro series rows", str(health["macro_series"]))
h2.metric("Events rows", str(health["events"]))
h3.metric("Prices rows", str(health["prices"]))
h4.metric("Last update (UTC)", health["last_update"] or "—")

st.divider()

# ============================================================
# Tabs
# ============================================================
tabs = st.tabs(["🧭 Overview", "📉 Yield Curve", "📈 CPI", "🏦 Rates", "👷 Labour (NFP)", "🏭 GDP"])

# ----------------------------
# Overview
# ----------------------------
with tabs[0]:
    st.subheader("Overview")

    # Yields (from DB)
    d10, y10, d10_prev, y10_prev = latest_two_values(DB_PATH, "DGS10")
    d2, y2, d2_prev, y2_prev = latest_two_values(DB_PATH, "DGS2")
    d3, y3, d3_prev, y3_prev = latest_two_values(DB_PATH, "DGS3MO")

    y10, y10_prev = safe_latest_prev(y10, y10_prev)
    y2, y2_prev = safe_latest_prev(y2, y2_prev)
    y3, y3_prev = safe_latest_prev(y3, y3_prev)

    spread_10_2 = y10 - y2
    spread_10_2_prev = y10_prev - y2_prev

    spread_10_3m = y10 - y3
    spread_10_3m_prev = y10_prev - y3_prev

    # CPI (from DB)
    cpi_df = load_series_df(DB_PATH, "CPIAUCSL")
    cpi = cpi_changes_from_df(cpi_df)

    # Row 1
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric(f"10Y (%) {d10 or ''}".strip(), f"{y10:.2f}", f"{(y10 - y10_prev):+.2f}")
    r1c2.metric(f"2Y (%) {d2 or ''}".strip(), f"{y2:.2f}", f"{(y2 - y2_prev):+.2f}")
    r1c3.metric("10Y–2Y (bps)", f"{spread_10_2*100:.0f}", f"{(spread_10_2 - spread_10_2_prev)*100:+.0f}")
    r1c4.metric("Curve state", "Inverted" if spread_10_2 < 0 else "Normal")

    # Row 2 (CPI)
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)

    if cpi["level"] is None:
        r2c1.metric("CPI index", "—")
        r2c2.metric("CPI MoM (%)", "—")
        r2c3.metric("CPI YoY (%)", "—")
        r2c4.metric("CPI date", "—")
        st.warning("No CPIAUCSL data yet. Click **Refresh latest** or **Backfill** in the sidebar.")
    else:
        r2c1.metric("CPI index", f"{cpi['level']:.2f}")
        r2c2.metric("CPI MoM (%)", f"{cpi['mom']:.2f}" if cpi["mom"] is not None else "—")
        r2c3.metric("CPI YoY (%)", f"{cpi['yoy']:.2f}" if cpi["yoy"] is not None else "—")
        r2c4.metric("CPI date", cpi["date"])

    st.divider()
    st.write("Selected assets (for later event studies):", assets)
    st.write("Reaction window:", event_window)

# ----------------------------
# Yield Curve tab (charts)
# ----------------------------
with tabs[1]:
    st.subheader("Yield Curve")

    df10 = load_series_df(DB_PATH, "DGS10").rename(columns={"value": "10Y"})
    df2 = load_series_df(DB_PATH, "DGS2").rename(columns={"value": "2Y"})
    df3 = load_series_df(DB_PATH, "DGS3MO").rename(columns={"value": "3M"})

    if df10.empty or df2.empty or df3.empty:
        st.warning("Not enough yield history yet. Use **Backfill** (sidebar).")
    else:
        merged = df10.merge(df2, on="obs_date", how="inner").merge(df3, on="obs_date", how="inner")
        merged["10Y-2Y"] = merged["10Y"] - merged["2Y"]
        merged["10Y-3M"] = merged["10Y"] - merged["3M"]

        st.markdown("#### Yields")
        st.line_chart(merged.set_index("obs_date")[["10Y", "2Y", "3M"]])

        st.markdown("#### Spreads")
        st.line_chart(merged.set_index("obs_date")[["10Y-2Y", "10Y-3M"]])

# ----------------------------
# CPI tab (charts + change rates)
# ----------------------------
with tabs[2]:
    st.subheader("CPI")

    cpi_df = load_series_df(DB_PATH, "CPIAUCSL")
    if cpi_df.empty:
        st.warning("No CPIAUCSL data yet. Click **Refresh latest** or **Backfill**.")
    else:
        cpi_df = cpi_df.sort_values("obs_date").reset_index(drop=True)
        cpi_df["MoM_%"] = cpi_df["value"].pct_change() * 100.0
        cpi_df["YoY_%"] = cpi_df["value"].pct_change(12) * 100.0

        latest = cpi_df.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CPI index", f"{latest['value']:.2f}")
        c2.metric("MoM (%)", f"{latest['MoM_%']:.2f}" if pd.notna(latest["MoM_%"]) else "—")
        c3.metric("YoY (%)", f"{latest['YoY_%']:.2f}" if pd.notna(latest["YoY_%"]) else "—")
        c4.metric("Date", latest["obs_date"].date().isoformat())

        st.markdown("#### CPI index level")
        st.line_chart(cpi_df.set_index("obs_date")[["value"]].rename(columns={"value": "CPIAUCSL"}))

        st.markdown("#### CPI change rates")
        st.line_chart(cpi_df.set_index("obs_date")[["MoM_%", "YoY_%"]])

# ----------------------------
# Placeholders for next macro components
# ----------------------------
with tabs[3]:
    st.subheader("Rates")
    st.info("Next: policy rates (EFFR / Fed Funds), FOMC event schedule, and regime tagging.")

with tabs[4]:
    st.subheader("Labour (NFP)")
    st.info("Next: NFP events (actual/forecast/previous) + unemployment context series.")

with tabs[5]:
    st.subheader("GDP")
    st.info("Next: real GDP series + GDP release events and event studies.")

st.divider()
st.markdown("### Status")
st.write(
    {
        "db_path": str(DB_PATH),
        "timestamp_local": fmt_dt(datetime.now()),
    }
)
