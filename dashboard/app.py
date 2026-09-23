"""
Nifty 50 AI Market Analytics — Streamlit Dashboard
====================================================
IBM SkillsBuild Data Analytics with AI Academy Internship Program
Conducted by Bharat Care in association with AICTE

Run:  streamlit run dashboard/app.py
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 0.  PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nifty 50 AI Market Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
  [data-testid="stSidebar"] { background-color: #0B132B; }
  [data-testid="stSidebar"] * { color: #E8EAF0 !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stMultiSelect label { color: #A0AEC0 !important; }
  .sidebar-title { font-size: 22px; font-weight: 700; color: #1C77C3; padding: 12px 0 4px; }
  .sidebar-subtitle { font-size: 11px; color: #7C8DB0; margin-bottom: 16px; }

  .kpi-card {
    background: #F5F7FA; border: 1px solid #E5E7EB;
    border-radius: 10px; padding: 16px 20px; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07);
  }
  .kpi-label { font-size: 11px; color: #57606A; text-transform: uppercase;
               letter-spacing: 0.08em; margin-bottom: 6px; }
  .kpi-value { font-size: 26px; font-weight: 700; color: #1f2328; line-height: 1.1; }
  .kpi-delta-pos { font-size: 13px; color: #16a34a; }
  .kpi-delta-neg { font-size: 13px; color: #dc2626; }

  .section-title { font-size: 22px; font-weight: 700; color: #0B132B;
                   border-left: 4px solid #1C77C3; padding-left: 12px; margin: 20px 0 12px; }
  .disclaimer {
    background: #FFF8E1; border-left: 4px solid #F59E0B;
    padding: 12px 16px; border-radius: 6px; font-size: 13px; color: #78350F;
  }
  div[data-testid="stMetricValue"] { font-size: 22px !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  PATHS
# ──────────────────────────────────────────────────────────────────────────────
DASH_DIR   = Path(__file__).parent
PROJ_DIR   = DASH_DIR.parent
DB_DIR     = PROJ_DIR.parent / "database"
CLEAN_CSV  = PROJ_DIR / "outputs" / "cleaned_data" / "nifty50_cleaned.csv"
RAW_CSV    = DB_DIR / "nifty50_historical_data.csv"
SUMM_CSV   = DB_DIR / "nifty50_summary_statistics.csv"
MODEL_DIR  = PROJ_DIR / "models"
SCALER_DIR = MODEL_DIR / "scalers"
META_JSON  = PROJ_DIR / "outputs" / "metrics" / "pipeline_meta.json"


# ──────────────────────────────────────────────────────────────────────────────
# 3.  DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading market data…")
def load_data():
    """Load cleaned data if available, otherwise fall back to raw CSV."""
    csv_path = CLEAN_CSV if CLEAN_CSV.exists() else RAW_CSV
    if not csv_path.exists():
        return None, f"Dataset not found at {csv_path}. Run the notebook first."

    df = pd.read_csv(csv_path, low_memory=False)

    # Parse date
    try:
        df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_convert(
            "Asia/Kolkata"
        ).dt.normalize()
    except Exception:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    # Ensure numeric OHLCV
    for col in ["Open", "High", "Low", "Close", "Volume", "Daily_Return"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Recompute Daily_Return if missing
    if "Daily_Return" not in df.columns or df["Daily_Return"].isna().mean() > 0.5:
        df["Daily_Return"] = df.groupby("Ticker")["Close"].pct_change()

    # Rolling volatility
    df["Roll_Vol_20"] = df.groupby("Ticker")["Daily_Return"].transform(
        lambda x: x.rolling(20).std() * np.sqrt(252)
    )
    return df, None


@st.cache_data(show_spinner="Loading summary statistics…")
def load_summary():
    if SUMM_CSV.exists():
        s = pd.read_csv(SUMM_CSV)
        for col in ["Starting_Price", "Ending_Price", "Total_Return_%",
                    "Highest_Price", "Lowest_Price", "Avg_Daily_Return_%",
                    "Volatility_%", "Current_Market_Cap", "Current_PE_Ratio"]:
            if col in s.columns:
                s[col] = pd.to_numeric(s[col], errors="coerce")
        return s
    return None


@st.cache_resource(show_spinner="Loading ML models…")
def load_models():
    """Load saved joblib models. Returns dict of {name: model}."""
    try:
        import joblib
    except ImportError:
        return {}, None

    models = {}
    scalers = {}
    model_files = {
        "XGBoost (Close)":         MODEL_DIR / "xgb_close_model.joblib",
        "Random Forest (Close)":   MODEL_DIR / "rf_close_model.joblib",
        "XGBoost (Direction)":     MODEL_DIR / "xgb_direction_model.joblib",
        "Random Forest (Dir)":     MODEL_DIR / "rf_direction_model.joblib",
        "RF (High)":               MODEL_DIR / "rf_high_model.joblib",
        "RF (Low)":                MODEL_DIR / "rf_low_model.joblib",
    }
    for name, path in model_files.items():
        if path.exists():
            try:
                models[name] = joblib.load(path)
            except Exception as e:
                st.warning(f"Could not load {name}: {e}")

    if (SCALER_DIR / "feature_scaler.joblib").exists():
        try:
            import joblib
            scalers["feature"] = joblib.load(SCALER_DIR / "feature_scaler.joblib")
        except Exception:
            pass

    meta = {}
    if META_JSON.exists():
        with open(META_JSON) as f:
            meta = json.load(f)

    return models, scalers, meta


# ──────────────────────────────────────────────────────────────────────────────
# 4.  HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "navy":    "#0B132B",
    "blue":    "#1C77C3",
    "teal":    "#2CA6A4",
    "light":   "#F5F7FA",
    "green":   "#16a34a",
    "red":     "#dc2626",
    "orange":  "#ea580c",
    "purple":  "#7c3aed",
}

PLOTLY_TEMPLATE = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FAFBFC",
    font=dict(family="Segoe UI, sans-serif", size=13, color="#1f2328"),
    xaxis=dict(gridcolor="#E5E7EB", linecolor="#D1D5DB"),
    yaxis=dict(gridcolor="#E5E7EB", linecolor="#D1D5DB"),
)


def kpi(label: str, value: str, delta: str = "", positive: bool = True):
    delta_cls = "kpi-delta-pos" if positive else "kpi-delta-neg"
    delta_html = f'<div class="{delta_cls}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {delta_html}
    </div>"""


def section(title: str):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def fmt_price(v):
    if pd.isna(v):
        return "N/A"
    return f"₹{v:,.2f}"


def fmt_pct(v):
    if pd.isna(v):
        return "N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"


def fmt_vol(v):
    if pd.isna(v):
        return "N/A"
    if v >= 1e9:
        return f"{v/1e9:.2f}B"
    if v >= 1e6:
        return f"{v/1e6:.2f}M"
    if v >= 1e3:
        return f"{v/1e3:.1f}K"
    return str(int(v))


# ──────────────────────────────────────────────────────────────────────────────
# 5.  SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">📈 Nifty 50 Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-subtitle">IBM SkillsBuild × Bharat Care × AICTE</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    PAGE = st.radio(
        "Navigation",
        options=[
            "🏠 Executive Overview",
            "🔍 Stock Explorer",
            "📅 Year Explorer",
            "🏭 Sector Analysis",
            "📊 Technical Analysis",
            "🤖 ML Predictions",
            "📉 Model Performance",
            "🗃️ Data Explorer",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("⚠️ For educational purposes only. Not financial advice.")


# ──────────────────────────────────────────────────────────────────────────────
# 6.  LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
df, load_err = load_data()
df_summary   = load_summary()

if load_err or df is None:
    st.error(
        f"**Data not found.**\n\n{load_err or ''}\n\n"
        "Please run the Jupyter notebook `Nifty50_AI_Market_Analytics.ipynb` first, "
        "or ensure the CSV file exists at:\n\n"
        f"`{RAW_CSV}`"
    )
    st.stop()

ALL_TICKERS  = sorted(df["Ticker"].unique())
ALL_SECTORS  = sorted(df["Sector"].unique())
ALL_YEARS    = sorted(df["Year"].dropna().unique().astype(int))
TICKER_NAMES = (
    df[["Ticker", "Company_Name"]].drop_duplicates()
    .set_index("Ticker")["Company_Name"]
    .to_dict()
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if PAGE == "🏠 Executive Overview":
    st.title("Nifty 50 AI Market Analytics")
    st.caption(
        "IBM SkillsBuild Data Analytics with AI Academy Internship Program "
        "| Bharat Care × AICTE"
    )

    # ── KPI row ──
    section("Market Snapshot")
    latest_date = df["Date"].max()
    latest      = df[df["Date"] == latest_date]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(kpi("Companies", str(df["Ticker"].nunique())), unsafe_allow_html=True)
    with col2:
        st.markdown(kpi("Sectors", str(df["Sector"].nunique())), unsafe_allow_html=True)
    with col3:
        st.markdown(kpi("Total Records", f"{len(df):,}"), unsafe_allow_html=True)
    with col4:
        st.markdown(kpi("Date Range", f"{df['Date'].min().year}–{df['Date'].max().year}"),
                    unsafe_allow_html=True)
    with col5:
        avg_ret_today = latest["Daily_Return"].mean() * 100
        st.markdown(
            kpi("Avg Return Today", fmt_pct(avg_ret_today), "", avg_ret_today >= 0),
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Market Overview Chart ──
    section("Market-Wide Price Trend (Median Close)")
    yearly_median = (
        df.groupby("Year")["Close"].median().reset_index()
    )
    fig_mkt = px.area(
        yearly_median,
        x="Year",
        y="Close",
        labels={"Close": "Median Close (₹)", "Year": "Year"},
        color_discrete_sequence=[PALETTE["blue"]],
    )
    fig_mkt.update_layout(**PLOTLY_TEMPLATE, title="Median Close Price — All Nifty 50 Stocks")
    fig_mkt.update_traces(line_color=PALETTE["blue"], fillcolor="rgba(28,119,195,0.12)")
    st.plotly_chart(fig_mkt, use_container_width=True)

    # ── Gainers / Losers / Volume leaders ──
    col_g, col_l, col_v = st.columns(3)

    with col_g:
        section("🟢 Top Gainers (Today)")
        top_gain = (
            latest.nlargest(5, "Daily_Return")[["Ticker", "Company_Name", "Close", "Daily_Return"]]
            .assign(Return=lambda x: (x["Daily_Return"] * 100).round(2))
        )
        st.dataframe(
            top_gain[["Ticker", "Close", "Return"]].rename(
                columns={"Return": "Return (%)"}
            ).reset_index(drop=True),
            hide_index=True,
            use_container_width=True,
        )

    with col_l:
        section("🔴 Top Losers (Today)")
        top_loss = (
            latest.nsmallest(5, "Daily_Return")[["Ticker", "Close", "Daily_Return"]]
            .assign(Return=lambda x: (x["Daily_Return"] * 100).round(2))
        )
        st.dataframe(
            top_loss[["Ticker", "Close", "Return"]].rename(
                columns={"Return": "Return (%)"}
            ).reset_index(drop=True),
            hide_index=True,
            use_container_width=True,
        )

    with col_v:
        section("📦 Volume Leaders (Today)")
        top_vol = latest.nlargest(5, "Volume")[["Ticker", "Close", "Volume"]]
        st.dataframe(top_vol.reset_index(drop=True), hide_index=True, use_container_width=True)

    # ── Sector Overview ──
    section("Sector Overview")
    sector_snap = (
        latest.groupby("Sector")
        .agg(Avg_Return=("Daily_Return", "mean"), Companies=("Ticker", "nunique"))
        .reset_index()
        .sort_values("Avg_Return", ascending=False)
    )
    sector_snap["Avg_Return_%"] = (sector_snap["Avg_Return"] * 100).round(3)
    fig_sec = px.bar(
        sector_snap,
        x="Sector",
        y="Avg_Return_%",
        color="Avg_Return_%",
        color_continuous_scale=["#dc2626", "#e5e7eb", "#16a34a"],
        color_continuous_midpoint=0,
        labels={"Avg_Return_%": "Avg Return (%)"},
        text="Avg_Return_%",
    )
    fig_sec.update_layout(**PLOTLY_TEMPLATE, title="Sector Average Return — Latest Trading Day",
                          coloraxis_showscale=False)
    fig_sec.update_traces(texttemplate="%{text:.3f}%", textposition="outside")
    st.plotly_chart(fig_sec, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — STOCK EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🔍 Stock Explorer":
    section("Stock Explorer")

    col_s, col_yr_s, col_yr_e = st.columns([2, 1, 1])
    with col_s:
        sel_ticker = st.selectbox("Select Stock", ALL_TICKERS,
                                   format_func=lambda t: f"{t} — {TICKER_NAMES.get(t, '')}")
    with col_yr_s:
        yr_start = st.selectbox("From Year", ALL_YEARS, index=0)
    with col_yr_e:
        yr_end   = st.selectbox("To Year",   ALL_YEARS, index=len(ALL_YEARS) - 1)

    s = df[(df["Ticker"] == sel_ticker) &
           (df["Year"] >= yr_start) & (df["Year"] <= yr_end)].copy().sort_values("Date")

    if s.empty:
        st.warning("No data for this selection.")
        st.stop()

    # KPI row
    latest_row = s.iloc[-1]
    prev_row   = s.iloc[-2] if len(s) > 1 else latest_row
    ret_today  = (latest_row["Close"] / prev_row["Close"] - 1) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(kpi("Latest Close", fmt_price(latest_row["Close"])), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("Daily Return", fmt_pct(ret_today), "", ret_today >= 0),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("Period High", fmt_price(s["High"].max())), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("Period Low", fmt_price(s["Low"].min())), unsafe_allow_html=True)
    with c5:
        vol_ann = s["Daily_Return"].std() * np.sqrt(252) * 100
        st.markdown(kpi("Ann. Volatility", fmt_pct(vol_ann)), unsafe_allow_html=True)

    st.divider()

    # ── Candlestick ──
    section(f"{sel_ticker} — OHLC Candlestick")
    fig_candle = go.Figure(
        go.Candlestick(
            x=s["Date"],
            open=s["Open"],
            high=s["High"],
            low=s["Low"],
            close=s["Close"],
            name="OHLC",
            increasing_line_color=PALETTE["green"],
            decreasing_line_color=PALETTE["red"],
        )
    )
    # Add SMA if available
    for sma_col, color, name in [
        ("SMA_20", PALETTE["orange"], "SMA 20"),
        ("SMA_50", PALETTE["blue"], "SMA 50"),
        ("SMA_200", PALETTE["purple"], "SMA 200"),
    ]:
        if sma_col in s.columns and s[sma_col].notna().sum() > 5:
            fig_candle.add_trace(
                go.Scatter(x=s["Date"], y=s[sma_col], mode="lines",
                           name=name, line=dict(color=color, width=1.2))
            )
    fig_candle.update_layout(**PLOTLY_TEMPLATE,
                              title=f"{sel_ticker} Candlestick with Moving Averages",
                              xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig_candle, use_container_width=True)

    # ── Volume ──
    section("Volume")
    volume_colors = ["green" if r >= 0 else "red"
                     for r in s["Daily_Return"].fillna(0)]
    fig_vol = go.Figure(
        go.Bar(x=s["Date"], y=s["Volume"], marker_color=volume_colors,
               name="Volume", opacity=0.7)
    )
    fig_vol.update_layout(**PLOTLY_TEMPLATE, title="Daily Volume", height=280)
    st.plotly_chart(fig_vol, use_container_width=True)

    # ── Returns and Rolling Volatility ──
    col_ret, col_rvol = st.columns(2)
    with col_ret:
        section("Daily Returns")
        ret_colors = ["green" if r >= 0 else "red" for r in s["Daily_Return"].fillna(0)]
        fig_ret = go.Figure(
            go.Bar(x=s["Date"], y=s["Daily_Return"] * 100,
                   marker_color=ret_colors, opacity=0.8)
        )
        fig_ret.update_layout(**PLOTLY_TEMPLATE, title="Daily Return (%)", height=300,
                               yaxis_title="Return (%)")
        st.plotly_chart(fig_ret, use_container_width=True)

    with col_rvol:
        section("Rolling Volatility (20-Day)")
        roll_v = s["Daily_Return"].rolling(20).std() * np.sqrt(252) * 100
        fig_rv = go.Figure(
            go.Scatter(x=s["Date"], y=roll_v, mode="lines",
                       line=dict(color=PALETTE["purple"], width=1.5))
        )
        fig_rv.update_layout(**PLOTLY_TEMPLATE, title="20-Day Ann. Volatility (%)", height=300,
                              yaxis_title="Volatility (%)")
        st.plotly_chart(fig_rv, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — YEAR EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "📅 Year Explorer":
    section("Historical Year Explorer")

    col_t, col_y1, col_y2 = st.columns([2, 1, 1])
    with col_t:
        sel_ticker_yr = st.selectbox("Stock", ALL_TICKERS,
                                      format_func=lambda t: f"{t} — {TICKER_NAMES.get(t, '')}")
    with col_y1:
        yr_a = st.selectbox("Year A", ALL_YEARS, index=max(0, len(ALL_YEARS) - 3))
    with col_y2:
        yr_b = st.selectbox("Year B", ALL_YEARS, index=len(ALL_YEARS) - 1)

    def year_summary(ticker, year):
        s = df[(df["Ticker"] == ticker) & (df["Year"] == year)].sort_values("Date")
        if s.empty:
            return None
        return {
            "Year":         year,
            "Open (₹)":     round(s["Open"].iloc[0], 2),
            "Close (₹)":    round(s["Close"].iloc[-1], 2),
            "High (₹)":     round(s["High"].max(), 2),
            "Low (₹)":      round(s["Low"].min(), 2),
            "Return (%)":   round((s["Close"].iloc[-1] / s["Open"].iloc[0] - 1) * 100, 2),
            "Volatility (%)": round(s["Daily_Return"].std() * np.sqrt(252) * 100, 2),
            "Avg Volume":   int(s["Volume"].mean()),
            "Trading Days": len(s),
        }

    rows = [year_summary(sel_ticker_yr, yr_a), year_summary(sel_ticker_yr, yr_b)]
    rows = [r for r in rows if r is not None]
    if rows:
        comp_df = pd.DataFrame(rows)
        st.dataframe(comp_df.T, use_container_width=True)

    # Full year-by-year chart
    section(f"{sel_ticker_yr} — Annual Return History")
    stock_yr = df[df["Ticker"] == sel_ticker_yr].sort_values("Date")
    annual = (
        stock_yr.groupby("Year")
        .apply(lambda g: (g["Close"].iloc[-1] / g["Open"].iloc[0] - 1) * 100)
        .reset_index(name="Annual_Return_%")
    )
    bar_colors = [PALETTE["green"] if v >= 0 else PALETTE["red"] for v in annual["Annual_Return_%"]]
    fig_ar = go.Figure(
        go.Bar(x=annual["Year"], y=annual["Annual_Return_%"],
               marker_color=bar_colors, text=annual["Annual_Return_%"].round(1),
               texttemplate="%{text}%", textposition="outside")
    )
    fig_ar.update_layout(**PLOTLY_TEMPLATE,
                          title=f"{sel_ticker_yr} — Annual Return (%)",
                          yaxis_title="Return (%)", height=420)
    fig_ar.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_ar, use_container_width=True)

    # Price level over full history
    section("Full Price History")
    fig_price = go.Figure(
        go.Scatter(x=stock_yr["Date"], y=stock_yr["Close"], mode="lines",
                   line=dict(color=PALETTE["blue"], width=1.5), name="Close")
    )
    for yr_sel, col in [(yr_a, PALETTE["orange"]), (yr_b, PALETTE["teal"])]:
        yr_data = stock_yr[stock_yr["Year"] == yr_sel]
        if not yr_data.empty:
            fig_price.add_vrect(
                x0=yr_data["Date"].min(), x1=yr_data["Date"].max(),
                fillcolor=col, opacity=0.07, line_width=0,
                annotation_text=str(yr_sel), annotation_position="top left",
            )
    fig_price.update_layout(**PLOTLY_TEMPLATE,
                             title=f"{sel_ticker_yr} — Price History with Selected Years",
                             yaxis_title="Close Price (₹)", height=420)
    st.plotly_chart(fig_price, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SECTOR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🏭 Sector Analysis":
    section("Sector Analysis")

    sel_sector = st.selectbox("Select Sector", ["All Sectors"] + ALL_SECTORS)
    df_sec_filter = df if sel_sector == "All Sectors" else df[df["Sector"] == sel_sector]

    # ── Sector performance table ──
    sector_perf = (
        df_sec_filter.groupby("Sector")
        .agg(
            Companies=("Ticker", "nunique"),
            Avg_Daily_Return=("Daily_Return", "mean"),
            Volatility=("Daily_Return", "std"),
            Median_Cap=("Market_Cap", "median"),
        )
        .reset_index()
    )
    sector_perf["Avg_Return_%"] = (sector_perf["Avg_Daily_Return"] * 100).round(4)
    sector_perf["Ann_Vol_%"]    = (sector_perf["Volatility"] * np.sqrt(252) * 100).round(2)
    sector_perf = sector_perf.sort_values("Avg_Return_%", ascending=False)

    col_a, col_b = st.columns(2)
    with col_a:
        section("Sector Average Daily Return")
        fig_sr = px.bar(
            sector_perf, x="Sector", y="Avg_Return_%",
            color="Avg_Return_%",
            color_continuous_scale=["#dc2626", "#e5e7eb", "#16a34a"],
            color_continuous_midpoint=0, text="Avg_Return_%",
        )
        fig_sr.update_layout(**PLOTLY_TEMPLATE, coloraxis_showscale=False, height=400)
        fig_sr.update_traces(texttemplate="%{text:.4f}%")
        st.plotly_chart(fig_sr, use_container_width=True)

    with col_b:
        section("Sector Annualized Volatility")
        fig_sv = px.bar(sector_perf, x="Sector", y="Ann_Vol_%",
                        color="Ann_Vol_%", color_continuous_scale="Reds",
                        text="Ann_Vol_%")
        fig_sv.update_layout(**PLOTLY_TEMPLATE, coloraxis_showscale=False, height=400)
        fig_sv.update_traces(texttemplate="%{text:.1f}%")
        st.plotly_chart(fig_sv, use_container_width=True)

    # ── Treemap: Market Cap ──
    if "Market_Cap" in df.columns:
        section("Market Capitalization Treemap")
        cap_data = (
            df.groupby(["Sector", "Ticker"])["Market_Cap"]
            .median().reset_index()
            .dropna(subset=["Market_Cap"])
        )
        cap_data["Cap_Bn"] = cap_data["Market_Cap"] / 1e9
        fig_tree = px.treemap(
            cap_data, path=["Sector", "Ticker"], values="Cap_Bn",
            title="Median Market Cap by Sector and Stock (₹ Billion)",
            color="Cap_Bn", color_continuous_scale="Blues",
        )
        fig_tree.update_layout(paper_bgcolor="white")
        st.plotly_chart(fig_tree, use_container_width=True)

    # ── Box plot: return distribution ──
    section("Return Distribution by Sector")
    sample = df_sec_filter.dropna(subset=["Daily_Return"])
    if len(sample) > 100000:
        sample = sample.sample(100000, random_state=42)
    fig_box = px.box(
        sample, x="Sector", y="Daily_Return",
        color="Sector", points=False,
    )
    fig_box.update_layout(**PLOTLY_TEMPLATE, showlegend=False, height=450,
                           yaxis_title="Daily Return", title="Daily Return Distribution by Sector")
    st.plotly_chart(fig_box, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — TECHNICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "📊 Technical Analysis":
    section("Technical Analysis")

    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        sel_ta_ticker = st.selectbox("Stock", ALL_TICKERS, key="ta_ticker",
                                      format_func=lambda t: f"{t} — {TICKER_NAMES.get(t, '')}")
    with col_t2:
        ta_period = st.selectbox("Period", ["1Y", "2Y", "5Y", "10Y", "All"], index=2)

    s_ta = df[df["Ticker"] == sel_ta_ticker].copy().sort_values("Date")
    # Filter by period
    from datetime import timedelta
    latest_d = s_ta["Date"].max()
    period_map = {"1Y": 365, "2Y": 730, "5Y": 1825, "10Y": 3650, "All": 99999}
    days = period_map.get(ta_period, 99999)
    s_ta = s_ta[s_ta["Date"] >= (latest_d - pd.Timedelta(days=days))]

    # Indicator toggles
    st.sidebar.markdown("**Indicator Controls**")
    show_sma  = st.sidebar.checkbox("SMA (20/50/200)", value=True)
    show_ema  = st.sidebar.checkbox("EMA (20/50)", value=False)
    show_bb   = st.sidebar.checkbox("Bollinger Bands", value=True)
    show_rsi  = st.sidebar.checkbox("RSI", value=True)
    show_macd = st.sidebar.checkbox("MACD", value=True)
    show_vol_chart = st.sidebar.checkbox("Volume", value=True)

    # Main price chart
    rows_count = 1 + int(show_rsi) + int(show_macd) + int(show_vol_chart)
    row_heights = [0.5] + [0.17] * (rows_count - 1)
    row_heights = [r / sum(row_heights) for r in row_heights]

    specs = [[{"secondary_y": False}]] * rows_count
    row_idx = {"price": 1}
    next_row = 2
    if show_vol_chart:
        row_idx["volume"] = next_row; next_row += 1
    if show_rsi:
        row_idx["rsi"] = next_row; next_row += 1
    if show_macd:
        row_idx["macd"] = next_row; next_row += 1

    subplot_titles = [f"{sel_ta_ticker} Price"] + \
        (["Volume"] if show_vol_chart else []) + \
        (["RSI (14)"] if show_rsi else []) + \
        (["MACD"] if show_macd else [])

    fig_ta = make_subplots(
        rows=rows_count, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=0.04,
        subplot_titles=subplot_titles,
    )

    # Candlestick
    fig_ta.add_trace(
        go.Candlestick(
            x=s_ta["Date"], open=s_ta["Open"], high=s_ta["High"],
            low=s_ta["Low"], close=s_ta["Close"], name="OHLC",
            increasing_line_color=PALETTE["green"],
            decreasing_line_color=PALETTE["red"],
        ),
        row=1, col=1,
    )
    # SMA
    if show_sma:
        for col, color, name in [
            ("SMA_20", "#f97316", "SMA 20"),
            ("SMA_50", PALETTE["blue"], "SMA 50"),
            ("SMA_200", PALETTE["purple"], "SMA 200"),
        ]:
            if col in s_ta.columns:
                fig_ta.add_trace(
                    go.Scatter(x=s_ta["Date"], y=s_ta[col], mode="lines",
                               name=name, line=dict(color=color, width=1.2)),
                    row=1, col=1,
                )
    # EMA
    if show_ema:
        for col, color, name in [
            ("EMA_20", "#14b8a6", "EMA 20"),
            ("EMA_50", "#6366f1", "EMA 50"),
        ]:
            if col in s_ta.columns:
                fig_ta.add_trace(
                    go.Scatter(x=s_ta["Date"], y=s_ta[col], mode="lines",
                               name=name, line=dict(color=color, width=1, dash="dot")),
                    row=1, col=1,
                )
    # Bollinger Bands
    if show_bb and "BB_High" in s_ta.columns:
        fig_ta.add_trace(
            go.Scatter(x=s_ta["Date"], y=s_ta["BB_High"], mode="lines",
                       name="BB High", line=dict(color="gray", width=0.8, dash="dash"),
                       showlegend=True),
            row=1, col=1,
        )
        fig_ta.add_trace(
            go.Scatter(x=s_ta["Date"], y=s_ta["BB_Low"], mode="lines",
                       name="BB Low", fill="tonexty",
                       fillcolor="rgba(150,150,150,0.1)",
                       line=dict(color="gray", width=0.8, dash="dash")),
            row=1, col=1,
        )
    # Volume
    if show_vol_chart:
        vol_colors = ["green" if r >= 0 else "red"
                      for r in s_ta["Daily_Return"].fillna(0)]
        fig_ta.add_trace(
            go.Bar(x=s_ta["Date"], y=s_ta["Volume"], marker_color=vol_colors,
                   name="Volume", opacity=0.6),
            row=row_idx["volume"], col=1,
        )
    # RSI
    if show_rsi and "RSI_14" in s_ta.columns:
        fig_ta.add_trace(
            go.Scatter(x=s_ta["Date"], y=s_ta["RSI_14"], mode="lines",
                       name="RSI", line=dict(color=PALETTE["teal"], width=1.2)),
            row=row_idx["rsi"], col=1,
        )
        fig_ta.add_hline(y=70, row=row_idx["rsi"], col=1,
                          line_color="red", line_dash="dash", line_width=0.8)
        fig_ta.add_hline(y=30, row=row_idx["rsi"], col=1,
                          line_color="green", line_dash="dash", line_width=0.8)
    # MACD
    if show_macd and "MACD" in s_ta.columns:
        fig_ta.add_trace(
            go.Scatter(x=s_ta["Date"], y=s_ta["MACD"], mode="lines",
                       name="MACD", line=dict(color=PALETTE["blue"], width=1.2)),
            row=row_idx["macd"], col=1,
        )
        if "MACD_Sig" in s_ta.columns:
            fig_ta.add_trace(
                go.Scatter(x=s_ta["Date"], y=s_ta["MACD_Sig"], mode="lines",
                           name="Signal", line=dict(color=PALETTE["red"], width=1)),
                row=row_idx["macd"], col=1,
            )
        if "MACD_Diff" in s_ta.columns:
            macd_bar_colors = ["green" if v >= 0 else "red" for v in s_ta["MACD_Diff"].fillna(0)]
            fig_ta.add_trace(
                go.Bar(x=s_ta["Date"], y=s_ta["MACD_Diff"],
                       marker_color=macd_bar_colors, name="MACD Hist", opacity=0.5),
                row=row_idx["macd"], col=1,
            )

    fig_ta.update_layout(
        paper_bgcolor="white", plot_bgcolor="#FAFBFC",
        font=dict(family="Segoe UI, sans-serif", size=12),
        height=720, showlegend=True,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    st.plotly_chart(fig_ta, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ML PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🤖 ML Predictions":
    section("Machine Learning Predictions")

    st.markdown(
        '<div class="disclaimer">⚠️ <strong>Disclaimer:</strong> All predictions shown here '
        "are statistical estimates based on historical data patterns. They are provided for "
        "educational purposes only and do NOT constitute financial advice. "
        "Model predictions carry significant uncertainty and should never be used as the sole "
        "basis for investment decisions. Past model performance does not guarantee future results.</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    models_dict, scalers_dict, meta = load_models()

    col_pt, col_pm = st.columns([2, 1])
    with col_pt:
        sel_pred_ticker = st.selectbox(
            "Stock", ALL_TICKERS, key="pred_ticker",
            format_func=lambda t: f"{t} — {TICKER_NAMES.get(t, '')}",
        )
    with col_pm:
        sel_model_name = st.selectbox(
            "Model",
            list(models_dict.keys()) if models_dict else ["No models loaded"],
        )

    s_pred = df[df["Ticker"] == sel_pred_ticker].sort_values("Date")

    if s_pred.empty:
        st.warning("No data for this stock.")
    elif not models_dict:
        st.info(
            "No trained models found. Please run the Jupyter notebook first to train and save models."
        )
    else:
        # Show last N actual vs predicted
        section("Actual vs Predicted — Close Price")

        feature_cols = meta.get("feature_cols", [])
        feat_cols_available = [c for c in feature_cols if c in df.columns]

        if feat_cols_available and sel_model_name in models_dict:
            model = models_dict[sel_model_name]
            # Prepare test portion (last 20% of stock data)
            s_pred_ml = s_pred.dropna(subset=["Close"] + feat_cols_available).copy()
            if len(s_pred_ml) > 50:
                test_start = int(len(s_pred_ml) * 0.8)
                test_s = s_pred_ml.iloc[test_start:]
                X_pred = test_s[feat_cols_available].fillna(0).values

                try:
                    y_pred_vals = model.predict(X_pred)
                    y_actual    = test_s.groupby("Ticker")["Close"].shift(-1).iloc[:-1].values
                    y_pred_vals = y_pred_vals[:len(y_actual)]
                    dates_pred  = test_s["Date"].values[:len(y_actual)]

                    from sklearn.metrics import mean_absolute_error, r2_score
                    mae_v  = mean_absolute_error(y_actual, y_pred_vals)
                    r2_v   = r2_score(y_actual, y_pred_vals)

                    fig_pred = go.Figure()
                    fig_pred.add_trace(
                        go.Scatter(x=dates_pred, y=y_actual, mode="lines",
                                   name="Actual", line=dict(color=PALETTE["navy"], width=1.5))
                    )
                    fig_pred.add_trace(
                        go.Scatter(x=dates_pred, y=y_pred_vals, mode="lines",
                                   name="Predicted", line=dict(color=PALETTE["orange"], width=1.5, dash="dot"))
                    )
                    fig_pred.update_layout(**PLOTLY_TEMPLATE, height=400,
                                           title=f"{sel_pred_ticker} — Actual vs Predicted Next-Day Close",
                                           yaxis_title="Price (₹)")
                    st.plotly_chart(fig_pred, use_container_width=True)

                    mc1, mc2 = st.columns(2)
                    with mc1:
                        st.markdown(kpi("MAE", fmt_price(mae_v)), unsafe_allow_html=True)
                    with mc2:
                        st.markdown(kpi("R²", f"{r2_v:.4f}"), unsafe_allow_html=True)

                    # Latest prediction
                    st.divider()
                    section("Latest Prediction (Most Recent Row)")
                    latest_feat = s_pred_ml.iloc[-1][feat_cols_available].fillna(0).values.reshape(1, -1)
                    pred_next = model.predict(latest_feat)[0]
                    current_close = s_pred_ml.iloc[-1]["Close"]
                    change_pct = (pred_next / current_close - 1) * 100

                    p1, p2, p3 = st.columns(3)
                    with p1:
                        st.markdown(kpi("Current Close", fmt_price(current_close)),
                                    unsafe_allow_html=True)
                    with p2:
                        st.markdown(kpi("Predicted Next Close", fmt_price(pred_next)),
                                    unsafe_allow_html=True)
                    with p3:
                        st.markdown(kpi("Expected Change", fmt_pct(change_pct), "", change_pct >= 0),
                                    unsafe_allow_html=True)

                    # Direction model
                    dir_key = "XGBoost (Direction)" if "XGBoost (Direction)" in models_dict else None
                    if dir_key:
                        dir_model = models_dict[dir_key]
                        dir_pred  = dir_model.predict(latest_feat)[0]
                        dir_prob  = dir_model.predict_proba(latest_feat)[0, 1]
                        dir_label = "📈 UP" if dir_pred == 1 else "📉 DOWN / FLAT"
                        st.metric("Direction", dir_label, f"Probability: {dir_prob:.2%}")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
            else:
                st.warning("Not enough data for this stock to generate predictions.")
        else:
            st.info("Feature columns not available. Run the notebook to generate predictions.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "📉 Model Performance":
    section("Model Performance & Comparison")

    metrics_path = PROJ_DIR / "outputs" / "metrics" / "model_comparison.csv"
    if metrics_path.exists():
        perf_df = pd.read_csv(metrics_path)

        st.dataframe(perf_df, use_container_width=True, hide_index=True)

        # Close regression models comparison
        close_rows = perf_df[perf_df["Task"] == "Close"].dropna(subset=["R2"])
        if not close_rows.empty:
            section("Regression Metrics — Next-Day Close")
            col_m1, col_m2, col_m3 = st.columns(3)
            for col_w, metric in zip([col_m1, col_m2, col_m3], ["MAE", "RMSE", "R2"]):
                with col_w:
                    if metric in close_rows.columns:
                        fig_m = px.bar(
                            close_rows, x="Model", y=metric,
                            text=metric, color="Model",
                            color_discrete_sequence=px.colors.qualitative.Set2,
                        )
                        fig_m.update_layout(**PLOTLY_TEMPLATE, showlegend=False, height=350,
                                             title=metric)
                        fig_m.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                        st.plotly_chart(fig_m, use_container_width=True)

        # Direction models
        dir_rows = perf_df[perf_df["Task"] == "Direction"].dropna(subset=["Dir_Accuracy"])
        if not dir_rows.empty:
            section("Classification Metrics — Direction")
            fig_dir = px.bar(
                dir_rows, x="Model", y=["Dir_Accuracy", "Dir_F1", "ROC_AUC"],
                barmode="group",
                labels={"value": "Score", "variable": "Metric"},
            )
            fig_dir.update_layout(**PLOTLY_TEMPLATE, height=400,
                                   title="Direction Prediction Metrics")
            st.plotly_chart(fig_dir, use_container_width=True)

        st.info(
            "Metrics reported on the chronological test set (last 15% of dates). "
            "Higher R² and lower MAE/RMSE are better for regression. "
            "Higher accuracy/F1/AUC are better for classification."
        )

    else:
        st.warning(
            "Model comparison file not found. "
            "Please run the Jupyter notebook to generate model metrics.\n\n"
            f"Expected path: `{metrics_path}`"
        )

    # ── Feature importance charts from saved figures ──
    section("Feature Importance")
    fi_path_xgb = PROJ_DIR / "outputs" / "figures" / "feature_importance_xgb.png"
    fi_path_rf  = PROJ_DIR / "outputs" / "figures" / "feature_importance_rf.png"

    col_fi1, col_fi2 = st.columns(2)
    with col_fi1:
        if fi_path_xgb.exists():
            st.image(str(fi_path_xgb), caption="XGBoost Feature Importance", use_container_width=True)
        else:
            st.info("XGBoost feature importance chart not found. Run the notebook first.")
    with col_fi2:
        if fi_path_rf.exists():
            st.image(str(fi_path_rf), caption="Random Forest Feature Importance", use_container_width=True)
        else:
            st.info("Random Forest feature importance chart not found. Run the notebook first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🗃️ Data Explorer":
    section("Data Explorer")

    with st.expander("🔧 Filter Options", expanded=True):
        col_de1, col_de2, col_de3 = st.columns(3)
        with col_de1:
            sel_de_tickers = st.multiselect("Stocks", ALL_TICKERS, default=ALL_TICKERS[:3])
        with col_de2:
            sel_de_sectors = st.multiselect("Sectors", ALL_SECTORS, default=[])
        with col_de3:
            yr_range = st.slider("Year Range", min_value=int(min(ALL_YEARS)),
                                  max_value=int(max(ALL_YEARS)),
                                  value=(2020, int(max(ALL_YEARS))))

    # Apply filters
    filtered = df.copy()
    if sel_de_tickers:
        filtered = filtered[filtered["Ticker"].isin(sel_de_tickers)]
    if sel_de_sectors:
        filtered = filtered[filtered["Sector"].isin(sel_de_sectors)]
    filtered = filtered[(filtered["Year"] >= yr_range[0]) & (filtered["Year"] <= yr_range[1])]

    # Column selector
    default_cols = ["Date", "Ticker", "Company_Name", "Sector", "Open", "High", "Low",
                    "Close", "Volume", "Daily_Return"]
    available_cols = [c for c in default_cols if c in filtered.columns]
    extra_cols     = [c for c in filtered.columns if c not in available_cols]
    sel_de_cols = st.multiselect(
        "Columns to display",
        available_cols + extra_cols,
        default=available_cols,
    )

    st.caption(f"Showing {len(filtered):,} rows after filters.")
    if sel_de_cols:
        st.dataframe(filtered[sel_de_cols].head(5000), use_container_width=True)
    else:
        st.dataframe(filtered.head(5000), use_container_width=True)

    # Download
    @st.cache_data
    def to_csv(data):
        return data.to_csv(index=False).encode("utf-8")

    csv_bytes = to_csv(filtered[sel_de_cols] if sel_de_cols else filtered)
    st.download_button(
        label="⬇️ Download Filtered Data (CSV)",
        data=csv_bytes,
        file_name="nifty50_filtered.csv",
        mime="text/csv",
    )

    # Summary statistics for filtered data
    with st.expander("📊 Summary Statistics"):
        numeric_filtered = filtered[sel_de_cols].select_dtypes(include=[np.number]) if sel_de_cols else filtered.select_dtypes(include=[np.number])
        st.dataframe(numeric_filtered.describe(), use_container_width=True)

    # Correlation heatmap on filtered
    with st.expander("🔥 Correlation Heatmap"):
        corr_cols_de = [c for c in ["Close", "Volume", "Daily_Return", "Roll_Vol_20",
                                     "SMA_20", "RSI_14", "MACD", "ATR_14"]
                        if c in filtered.columns]
        if len(corr_cols_de) >= 3:
            corr_m = filtered[corr_cols_de].corr()
            fig_ch = px.imshow(
                corr_m, text_auto=".2f", color_continuous_scale="RdYlGn",
                zmin=-1, zmax=1,
                title="Correlation Heatmap — Filtered Data",
            )
            fig_ch.update_layout(paper_bgcolor="white", height=500)
            st.plotly_chart(fig_ch, use_container_width=True)
            st.caption("NOTE: Correlation measures linear association only. It does not imply causation.")
        else:
            st.info("Not enough numeric columns for correlation heatmap.")


# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;font-size:11px;color:#888;'>"
    "Nifty 50 AI Market Analytics &nbsp;|&nbsp; "
    "IBM SkillsBuild Data Analytics with AI Academy Internship Program &nbsp;|&nbsp; "
    "Bharat Care × AICTE &nbsp;|&nbsp; "
    "Educational purposes only — Not financial advice"
    "</p>",
    unsafe_allow_html=True,
)
