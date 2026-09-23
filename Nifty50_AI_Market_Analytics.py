"""
Nifty 50 Stock Market Analytics and Multi-Task Machine Learning Prediction System
==================================================================================
IBM SkillsBuild Data Analytics with AI Academy Internship Program
Conducted by Bharat Care in association with AICTE

USAGE
-----
  # Run the full ML pipeline first (trains & saves models):
      python Nifty50_AI_Market_Analytics.py --pipeline

  # Launch the Streamlit dashboard:
      streamlit run Nifty50_AI_Market_Analytics.py

  # Run pipeline then immediately launch dashboard:
      python Nifty50_AI_Market_Analytics.py --pipeline --dashboard

DISCLAIMER
----------
This project is developed strictly for educational and research purposes.
It does not constitute financial advice. All predictions are model estimates
based on historical patterns and carry significant uncertainty.
Past performance does not guarantee future results.
"""

# ══════════════════════════════════════════════════════════════════════════════
# SITE-PACKAGES BOOTSTRAP
# Adds ALL site-packages directories for the running Python to sys.path so that
# packages installed via `pip` (user or global) are always importable regardless
# of how the script is launched (VS Code, streamlit run, bare python, etc.).
# ══════════════════════════════════════════════════════════════════════════════
import sys, site as _site

def _ensure_all_site_packages():
    """Guarantee every site-packages directory is on sys.path."""
    candidates = []
    # 1. user site-packages (pip install --user)
    try:
        candidates.append(_site.getusersitepackages())
    except Exception:
        pass
    # 2. all global site-packages directories
    try:
        candidates.extend(_site.getsitepackages())
    except Exception:
        pass
    for path in candidates:
        if path and path not in sys.path:
            sys.path.insert(0, path)

_ensure_all_site_packages()

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for pipeline mode
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve,
    ConfusionMatrixDisplay,
)
import xgboost as xgb
import joblib

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# PATHS  (all relative to this file's directory)
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR    = Path(__file__).parent
DB_DIR      = BASE_DIR.parent / "database"
MAIN_CSV    = DB_DIR / "nifty50_historical_data.csv"
SUMMARY_CSV = DB_DIR / "nifty50_summary_statistics.csv"
META_JSON   = DB_DIR / "metadata.json"

MODEL_DIR   = BASE_DIR / "models"
SCALER_DIR  = MODEL_DIR / "scalers"
OUTPUT_DIR  = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
CLEANED_DIR = OUTPUT_DIR / "cleaned_data"
CLEAN_CSV   = CLEANED_DIR / "nifty50_cleaned.csv"
PIPE_META   = METRICS_DIR / "pipeline_meta.json"

for _d in [MODEL_DIR, SCALER_DIR, OUTPUT_DIR, FIGURES_DIR, METRICS_DIR, CLEANED_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
RANDOM_STATE    = 42
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
LAG_PERIODS     = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [5, 10, 20]
FAST_MODE       = False          # set True to train on 5 tickers for quick testing
FAST_TICKERS    = ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "TCS.NS"]
LSTM_TICKER     = "RELIANCE.NS"
LSTM_SEQ_LEN    = 30

np.random.seed(RANDOM_STATE)

# ══════════════════════════════════════════════════════════════════════════════
# ██████████████████████  PIPELINE FUNCTIONS  ██████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────
def load_raw():
    if not MAIN_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found at {MAIN_CSV}.\n"
            "Download from: https://www.kaggle.com/datasets/kalyan197/"
            "nifty50-stocks1999-2026-daily-ohlcv-and-fundamentals"
        )
    print(f"Loading {MAIN_CSV} …")
    df = pd.read_csv(MAIN_CSV, low_memory=False)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA QUALITY AUDIT
# ─────────────────────────────────────────────────────────────────────────────
def data_quality_audit(df):
    print("\n" + "=" * 70)
    print("DATA QUALITY AUDIT — 20-POINT CHECKLIST")
    print("=" * 70)

    print(f"\n[1]  Dimensions : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\n[2]  Columns    : {list(df.columns)}")
    print(f"\n[3]  Dtypes:\n{df.dtypes.to_string()}")

    # [4] first / last
    print(f"\n[4]  First row  : {df.iloc[0].to_dict()}")
    print(f"     Last row   : {df.iloc[-1].to_dict()}")

    # [5] descriptive stats (summary only)
    print(f"\n[5]  Numeric describe (transposed):\n"
          f"{df.describe(include=[np.number]).T.to_string()}")

    # [6] missing
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    mr = pd.DataFrame({"Count": missing, "%": missing_pct})
    mr = mr[mr["Count"] > 0].sort_values("%", ascending=False)
    print(f"\n[6]  Missing values:\n{'  None' if mr.empty else mr.to_string()}")

    # [7][8] duplicates
    n_dup = df.duplicated().sum()
    n_dup_dt = df.duplicated(subset=["Date", "Ticker"]).sum()
    print(f"\n[7]  Fully duplicate rows     : {n_dup:,}")
    print(f"[8]  Duplicate Date-Ticker    : {n_dup_dt:,}")

    # [9] dates
    dates_parsed = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    n_inv_dates  = dates_parsed.isna().sum()
    print(f"\n[9]  Invalid dates            : {n_inv_dates:,}")
    print(f"     Date range              : {dates_parsed.min()} → {dates_parsed.max()}")

    # [10][11] price / volume
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            n_neg = (pd.to_numeric(df[col], errors="coerce") <= 0).sum()
            print(f"[10] {col:5s} zero/negative: {n_neg:,}")
    if "Volume" in df.columns:
        print(f"[11] Zero volume rows        : "
              f"{(pd.to_numeric(df['Volume'], errors='coerce') == 0).sum():,}")

    # [12] OHLC consistency
    nd = df[["Open", "High", "Low", "Close"]].apply(pd.to_numeric, errors="coerce")
    print(f"\n[12] High < Low  : {(nd['High'] < nd['Low']).sum():,}")
    print(f"     High < Open : {(nd['High'] < nd['Open']).sum():,}")
    print(f"     Low  > Close: {(nd['Low']  > nd['Close']).sum():,}")

    # [13] IQR outliers on Close
    close_n = pd.to_numeric(df["Close"], errors="coerce").dropna()
    Q1, Q3  = close_n.quantile(0.25), close_n.quantile(0.75)
    IQR     = Q3 - Q1
    n_out   = ((close_n < Q1 - 3 * IQR) | (close_n > Q3 + 3 * IQR)).sum()
    print(f"\n[13] Extreme outliers (3×IQR): {n_out:,} "
          f"(expected — price scale spans 25 yrs + splits)")

    # [14] categorical consistency
    print(f"\n[14] Unique tickers : {df['Ticker'].nunique()}")
    print(f"     Unique sectors : {df['Sector'].nunique() if 'Sector' in df.columns else 'N/A'}")

    # [15] global sort
    is_sorted = dates_parsed.is_monotonic_increasing
    print(f"\n[15] Globally date-sorted    : {is_sorted} "
          f"(per-ticker sort enforced in cleaning)")

    # [16][17] obs per stock
    grp = df.groupby("Ticker").agg(Count=("Close", "count"),
                                    First=("Date", "min"),
                                    Last=("Date", "max"))
    print(f"\n[16-17] Obs per stock — min {grp['Count'].min()}, "
          f"max {grp['Count'].max()}, mean {grp['Count'].mean():.0f}")

    # [18] missing business days (sample)
    if "RELIANCE.NS" in df["Ticker"].values:
        rel = pd.to_datetime(
            df[df["Ticker"] == "RELIANCE.NS"]["Date"], utc=True, errors="coerce"
        ).dt.tz_convert("Asia/Kolkata").dt.normalize().dropna().sort_values()
        exp_bd = pd.bdate_range(rel.min(), rel.max())
        miss_bd = exp_bd.difference(rel)
        print(f"\n[18] RELIANCE.NS — actual days: {len(rel)}, "
              f"business days in range: {len(exp_bd)}, "
              f"apparent gaps: {len(miss_bd)} (mostly Indian holidays)")

    # [19] leakage risk
    print("\n[19] Leakage-risk columns excluded from ML features:")
    for c in ["52Week_High", "52Week_Low", "MA_50", "MA_200", "Volatility_20D"]:
        if c in df.columns:
            print(f"     ⚠  {c}")

    print("\n[20] AUDIT COMPLETE. Proceeding to cleaning.\n")
    return int(n_inv_dates), int(n_dup_dt)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLEANING
# ─────────────────────────────────────────────────────────────────────────────
def clean_data(df_raw):
    print("Cleaning data …")
    df = df_raw.copy()
    n0 = len(df)

    # Step 1: parse dates
    df["Date"] = (pd.to_datetime(df["Date"], utc=True, errors="coerce")
                  .dt.tz_convert("Asia/Kolkata").dt.normalize())
    df = df.dropna(subset=["Date"])

    # Step 2: sort
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Step 3: drop full duplicates
    before = len(df); df = df.drop_duplicates()
    print(f"  Dropped {before - len(df):,} fully duplicate rows")

    # Step 4: drop duplicate Date-Ticker
    before = len(df); df = df.drop_duplicates(subset=["Date", "Ticker"], keep="first")
    print(f"  Dropped {before - len(df):,} duplicate Date-Ticker rows")

    # Step 5: numeric conversion
    num_cols = ["Open", "High", "Low", "Close", "Volume", "Dividend", "Stock_Split",
                "Daily_Return", "Volatility_20D", "MA_50", "MA_200", "Market_Cap",
                "PE_Ratio", "Forward_PE", "PEG_Ratio", "Price_to_Book",
                "Dividend_Yield", "EPS", "Beta", "52Week_High", "52Week_Low"]
    for c in [x for x in num_cols if x in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Step 6: forward-fill prices within ticker
    price_cols = ["Open", "High", "Low", "Close"]
    df[price_cols] = df.groupby("Ticker")[price_cols].transform(lambda x: x.ffill())
    df["Volume"] = df["Volume"].fillna(0)

    # Step 7: drop rows where Close still NaN
    before = len(df); df = df.dropna(subset=["Close"])
    print(f"  Dropped {before - len(df):,} rows with unresolvable NaN Close")

    # Step 8: drop zero/negative Close
    before = len(df); df = df[df["Close"] > 0]
    print(f"  Dropped {before - len(df):,} rows with Close ≤ 0")

    # Step 9: strip categorical strings
    for c in ["Ticker", "Company_Name", "Sector"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Step 10: recompute Daily_Return from clean prices
    df["Daily_Return"] = df.groupby("Ticker")["Close"].pct_change()

    # Step 11: fill fundamental NaN with ticker-wise median
    fund_cols = ["PE_Ratio", "Forward_PE", "PEG_Ratio", "Price_to_Book",
                 "Dividend_Yield", "EPS", "Beta", "Market_Cap"]
    for c in [x for x in fund_cols if x in df.columns]:
        df[c] = df.groupby("Ticker")[c].transform(lambda x: x.fillna(x.median()))

    # Optional fast-mode subsetting
    if FAST_MODE:
        df = df[df["Ticker"].isin(FAST_TICKERS)].reset_index(drop=True)
        print(f"  ⚡ FAST_MODE: subset to {FAST_TICKERS}")

    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    print(f"  Clean dataset: {len(df):,} rows (removed {n0 - len(df):,} total) "
          f"| {df['Ticker'].nunique()} tickers")
    df.to_csv(CLEAN_CSV, index=False)
    print(f"  Saved → {CLEAN_CSV}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. EDA CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def run_eda(df):
    print("\nRunning EDA …")

    # Market overview
    yearly = df.groupby("Year")["Close"].median()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(yearly.index, yearly.values, alpha=0.25, color="#1C77C3")
    ax.plot(yearly.index, yearly.values, color="#1C77C3", linewidth=2)
    ax.set_title("Median Close Price — All Nifty 50 Stocks (Yearly)", fontsize=13)
    ax.set_xlabel("Year"); ax.set_ylabel("Median Close (₹)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_market_overview.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Sector bar chart
    sector_stats = (df.groupby("Sector")
                    .agg(Avg_Return=("Daily_Return", "mean"),
                         Ann_Vol=("Daily_Return", lambda x: x.std() * np.sqrt(252)))
                    .reset_index().sort_values("Avg_Return", ascending=False))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    colors = ["#16a34a" if v > 0 else "#dc2626" for v in sector_stats["Avg_Return"]]
    axes[0].barh(sector_stats["Sector"], sector_stats["Avg_Return"] * 100,
                 color=colors); axes[0].set_title("Avg Daily Return by Sector (%)")
    axes[1].barh(sector_stats["Sector"], sector_stats["Ann_Vol"] * 100,
                 color="#1C77C3"); axes[1].set_title("Annualized Volatility by Sector (%)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_sector_analysis.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Monthly seasonality
    monthly = df.groupby("Month")["Daily_Return"].mean() * 100
    colors_m = ["#16a34a" if v > 0 else "#dc2626" for v in monthly]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(monthly.index, monthly.values, color=colors_m, edgecolor="white")
    ax.set_title("Average Daily Return by Month (Seasonal Pattern)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_monthly_seasonality.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Correlation heatmap
    corr_cols = [c for c in ["Open","High","Low","Close","Volume","Daily_Return",
                              "Volatility_20D","MA_50","MA_200","PE_Ratio","Beta","EPS"]
                 if c in df.columns]
    corr = df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.4, ax=ax, vmin=-1, vmax=1,
                annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_correlation_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"  EDA charts saved to {FIGURES_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df):
    print("\nEngineering features …")
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    def _per_ticker(g):
        g = g.copy()
        c = g["Close"]; h = g["High"]; lo = g["Low"]; v = g["Volume"]
        for lag in LAG_PERIODS:
            g[f"Close_Lag{lag}"]  = c.shift(lag)
            g[f"Return_Lag{lag}"] = c.pct_change(lag)
        for win in ROLLING_WINDOWS:
            g[f"SMA_{win}"]  = c.rolling(win).mean()
            g[f"STD_{win}"]  = c.rolling(win).std()
            g[f"Vol_{win}"]  = c.pct_change().rolling(win).std() * np.sqrt(252)
        g["Momentum_5"]    = c - c.shift(5)
        g["Momentum_10"]   = c - c.shift(10)
        g["HL_Range"]      = h - lo
        g["HL_Range_Pct"]  = (h - lo) / c.shift(1).replace(0, np.nan)
        sma50              = c.rolling(50).mean()
        sma200             = c.rolling(200).mean()
        g["Price_vs_SMA50"]  = (c - sma50)  / sma50.replace(0, np.nan)
        g["Price_vs_SMA200"] = (c - sma200) / sma200.replace(0, np.nan)
        g["Volume_Change"] = v.pct_change()
        g["Volume_SMA5"]   = v.rolling(5).mean()
        return g

    df = df.groupby("Ticker", group_keys=False).apply(_per_ticker)
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    print(f"  Features added. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def add_indicators(df):
    print("Adding technical indicators …")

    def _ind(g):
        g = g.copy().sort_values("Date")
        c = g["Close"]; h = g["High"]; lo = g["Low"]
        if TA_AVAILABLE and len(g) >= 200:
            g["SMA_20"]   = ta.trend.sma_indicator(c, 20)
            g["SMA_50"]   = ta.trend.sma_indicator(c, 50)
            g["SMA_100"]  = ta.trend.sma_indicator(c, 100)
            g["SMA_200"]  = ta.trend.sma_indicator(c, 200)
            g["EMA_20"]   = ta.trend.ema_indicator(c, 20)
            g["EMA_50"]   = ta.trend.ema_indicator(c, 50)
            g["RSI_14"]   = ta.momentum.RSIIndicator(c, 14).rsi()
            macd_obj      = ta.trend.MACD(c)
            g["MACD"]     = macd_obj.macd()
            g["MACD_Sig"] = macd_obj.macd_signal()
            g["MACD_Diff"]= macd_obj.macd_diff()
            bb            = ta.volatility.BollingerBands(c, 20)
            g["BB_High"]  = bb.bollinger_hband()
            g["BB_Low"]   = bb.bollinger_lband()
            g["BB_Mid"]   = bb.bollinger_mavg()
            g["BB_Width"] = (g["BB_High"] - g["BB_Low"]) / g["BB_Mid"].replace(0, np.nan)
            g["ATR_14"]   = ta.volatility.AverageTrueRange(h, lo, c, 14).average_true_range()
        else:
            g["SMA_20"]   = c.rolling(20).mean()
            g["SMA_50"]   = c.rolling(50).mean()
            g["SMA_100"]  = c.rolling(100).mean()
            g["SMA_200"]  = c.rolling(200).mean()
            g["EMA_20"]   = c.ewm(span=20, adjust=False).mean()
            g["EMA_50"]   = c.ewm(span=50, adjust=False).mean()
            delta         = c.diff()
            gain          = delta.clip(lower=0).rolling(14).mean()
            loss          = (-delta.clip(upper=0)).rolling(14).mean()
            g["RSI_14"]   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
            ema12         = c.ewm(span=12, adjust=False).mean()
            ema26         = c.ewm(span=26, adjust=False).mean()
            g["MACD"]     = ema12 - ema26
            g["MACD_Sig"] = g["MACD"].ewm(span=9, adjust=False).mean()
            g["MACD_Diff"]= g["MACD"] - g["MACD_Sig"]
            sma20         = c.rolling(20).mean(); std20 = c.rolling(20).std()
            g["BB_High"]  = sma20 + 2 * std20
            g["BB_Low"]   = sma20 - 2 * std20
            g["BB_Mid"]   = sma20
            g["BB_Width"] = (g["BB_High"] - g["BB_Low"]) / sma20.replace(0, np.nan)
            tr            = pd.concat([h - lo,
                                       (h - c.shift(1)).abs(),
                                       (lo - c.shift(1)).abs()], axis=1).max(axis=1)
            g["ATR_14"]   = tr.rolling(14).mean()
        return g

    df = df.groupby("Ticker", group_keys=False).apply(_ind)
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    print(f"  Indicators added. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. TARGET CREATION
# ─────────────────────────────────────────────────────────────────────────────
def create_targets(df):
    print("Creating prediction targets …")
    df["Next_Day_Close"]      = df.groupby("Ticker")["Close"].shift(-1)
    df["Next_Day_High"]       = df.groupby("Ticker")["High"].shift(-1)
    df["Next_Day_Low"]        = df.groupby("Ticker")["Low"].shift(-1)
    df["Next_Day_Return"]     = df.groupby("Ticker")["Daily_Return"].shift(-1)
    df["Roll_Vol_20"]         = df.groupby("Ticker")["Daily_Return"].transform(
                                    lambda x: x.rolling(20).std() * np.sqrt(252))
    df["Next_Day_Volatility"] = df.groupby("Ticker")["Roll_Vol_20"].shift(-1)
    df["Next_Day_Direction"]  = (df["Next_Day_Close"] > df["Close"]).astype(float)
    q33 = df["Roll_Vol_20"].quantile(0.33)
    q67 = df["Roll_Vol_20"].quantile(0.67)
    def _regime(v):
        if pd.isna(v): return np.nan
        return 0 if v <= q33 else (1 if v <= q67 else 2)
    df["Market_Regime"]       = df["Roll_Vol_20"].apply(_regime)
    df["Next_Day_Regime"]     = df.groupby("Ticker")["Market_Regime"].shift(-1)
    print("  Targets: Next_Day_Close/High/Low/Return/Volatility/Direction/Regime")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 8. SPLIT
# ─────────────────────────────────────────────────────────────────────────────
EXCLUDE_FROM_FEATURES = {
    "Date", "Ticker", "Company_Name", "Sector",
    "Next_Day_Close", "Next_Day_High", "Next_Day_Low", "Next_Day_Return",
    "Next_Day_Volatility", "Next_Day_Direction", "Next_Day_Regime", "Market_Regime",
    "52Week_High", "52Week_Low", "Roll_Vol_20", "Year", "Month",
    "MA_50", "MA_200", "Volatility_20D", "Daily_Return",
}

def prepare_split(df):
    print("\nPreparing train/val/test split …")
    df_ml = df.dropna(subset=["Next_Day_Close"]).copy()

    # Build feature list
    feat_cols = sorted([c for c in df_ml.columns
                        if c not in EXCLUDE_FROM_FEATURES
                        and df_ml[c].dtype in [np.float64, np.float32,
                                               np.int64, np.int32, float, int]])
    df_ml_num = df_ml[feat_cols].select_dtypes(include=[np.number])
    feat_cols = list(df_ml_num.columns)

    df_ml = df_ml.dropna(subset=feat_cols).copy()

    # Chronological split by date
    all_dates = df_ml["Date"].sort_values().unique()
    n         = len(all_dates)
    train_cut = all_dates[int(n * TRAIN_RATIO)]
    val_cut   = all_dates[int(n * (TRAIN_RATIO + VAL_RATIO))]

    train = df_ml[df_ml["Date"] <= train_cut]
    val   = df_ml[(df_ml["Date"] > train_cut) & (df_ml["Date"] <= val_cut)]
    test  = df_ml[df_ml["Date"] > val_cut]

    print(f"  Train : {train['Date'].min().date()} → {train['Date'].max().date()} "
          f"({len(train):,} rows)")
    print(f"  Val   : {val['Date'].min().date()} → {val['Date'].max().date()} "
          f"({len(val):,} rows)")
    print(f"  Test  : {test['Date'].min().date()} → {test['Date'].max().date()} "
          f"({len(test):,} rows)")
    print(f"  Features: {len(feat_cols)}")

    X_tr  = train[feat_cols].fillna(0).values
    X_val = val[feat_cols].fillna(0).values
    X_te  = test[feat_cols].fillna(0).values

    scaler = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_val_sc = scaler.transform(X_val)
    X_te_sc  = scaler.transform(X_te)
    joblib.dump(scaler, SCALER_DIR / "feature_scaler.joblib")

    tgt_scaler = StandardScaler()
    tgt_scaler.fit(train["Next_Day_Close"].values.reshape(-1, 1))
    joblib.dump(tgt_scaler, SCALER_DIR / "target_scaler.joblib")

    return (train, val, test,
            feat_cols, X_tr, X_val, X_te,
            X_tr_sc, X_val_sc, X_te_sc,
            train_cut, val_cut)


# ─────────────────────────────────────────────────────────────────────────────
# 9. TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────
def train_models(train, val, test, feat_cols,
                 X_tr, X_val, X_te,
                 X_tr_sc, X_val_sc, X_te_sc):
    print("\nTraining models …")
    results = {}
    n_est   = 50 if FAST_MODE else 200

    y_close_tr  = train["Next_Day_Close"].values
    y_close_val = val["Next_Day_Close"].values
    y_close_te  = test["Next_Day_Close"].values
    y_dir_tr    = train["Next_Day_Direction"].values
    y_dir_te    = test["Next_Day_Direction"].values
    y_high_tr   = train["Next_Day_High"].values
    y_high_te   = test["Next_Day_High"].values
    y_low_tr    = train["Next_Day_Low"].values
    y_low_te    = test["Next_Day_Low"].values

    def reg_metrics(y_true, y_pred, label):
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        print(f"  {label:<32} MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
        return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}

    def clf_metrics(y_true, y_pred, y_prob, label):
        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_prob)
        print(f"  {label:<32} Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
        return {"Dir_Accuracy": round(acc, 4), "Dir_F1": round(f1, 4), "ROC_AUC": round(auc, 4)}

    # ── Baseline ──
    baseline_pred = test["Close"].values
    baseline_dir  = (test["Close"].values > test["Close_Lag1"].values).astype(float) \
                    if "Close_Lag1" in test.columns else np.ones(len(test))
    m = reg_metrics(y_close_te, baseline_pred, "Baseline (prev-close)")
    m.update({"Task": "Close"})
    results["Baseline"] = m

    # ── Linear Regression (Close) ──
    lr = LinearRegression().fit(X_tr_sc, y_close_tr)
    lr_pred = lr.predict(X_te_sc)
    m = reg_metrics(y_close_te, lr_pred, "Linear Regression (Close)")
    m["Task"] = "Close"; results["Linear Regression"] = m

    # ── Logistic Regression (Direction) ──
    lgr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(X_tr_sc, y_dir_tr)
    lgr_pred = lgr.predict(X_te_sc)
    lgr_prob = lgr.predict_proba(X_te_sc)[:, 1]
    m = clf_metrics(y_dir_te, lgr_pred, lgr_prob, "Logistic Regression (Dir)")
    m["Task"] = "Direction"; results["Logistic Regression"] = m

    # ── Random Forest (Close) ──
    rf_reg = RandomForestRegressor(n_estimators=n_est, max_depth=12,
                                   min_samples_leaf=5, n_jobs=-1,
                                   random_state=RANDOM_STATE).fit(X_tr, y_close_tr)
    rf_pred = rf_reg.predict(X_te)
    m = reg_metrics(y_close_te, rf_pred, "Random Forest (Close)")
    m["Task"] = "Close"; results["Random Forest"] = m
    joblib.dump(rf_reg, MODEL_DIR / "rf_close_model.joblib")

    # ── Random Forest Classifier (Direction) ──
    rf_clf = RandomForestClassifier(n_estimators=n_est, max_depth=10,
                                    min_samples_leaf=10, n_jobs=-1,
                                    random_state=RANDOM_STATE).fit(X_tr, y_dir_tr)
    rf_dir_pred = rf_clf.predict(X_te)
    rf_dir_prob = rf_clf.predict_proba(X_te)[:, 1]
    m = clf_metrics(y_dir_te, rf_dir_pred, rf_dir_prob, "Random Forest (Direction)")
    m["Task"] = "Direction"; results["Random Forest Clf"] = m
    joblib.dump(rf_clf, MODEL_DIR / "rf_direction_model.joblib")

    # ── RF for High / Low ──
    for tgt_name, y_tr, y_te, mfile in [
        ("Next_Day_High", y_high_tr, y_high_te, "rf_high_model.joblib"),
        ("Next_Day_Low",  y_low_tr,  y_low_te,  "rf_low_model.joblib"),
    ]:
        m_obj = RandomForestRegressor(n_estimators=n_est, max_depth=12,
                                      min_samples_leaf=5, n_jobs=-1,
                                      random_state=RANDOM_STATE).fit(X_tr, y_tr)
        pred  = m_obj.predict(X_te)
        m     = reg_metrics(y_te, pred, f"Random Forest ({tgt_name})")
        m["Task"] = tgt_name; results[f"RF_{tgt_name}"] = m
        joblib.dump(m_obj, MODEL_DIR / mfile)

    # ── XGBoost (Close) ──
    xgb_reg = xgb.XGBRegressor(
        n_estimators=n_est, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
        random_state=RANDOM_STATE, early_stopping_rounds=20, verbosity=0
    )
    xgb_reg.fit(X_tr, y_close_tr, eval_set=[(X_val, y_close_val)], verbose=False)
    xgb_pred = xgb_reg.predict(X_te)
    m = reg_metrics(y_close_te, xgb_pred, "XGBoost (Close)")
    m["Task"] = "Close"; results["XGBoost"] = m
    joblib.dump(xgb_reg, MODEL_DIR / "xgb_close_model.joblib")

    # ── XGBoost Classifier (Direction) ──
    xgb_clf = xgb.XGBClassifier(
        n_estimators=n_est, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
        random_state=RANDOM_STATE, early_stopping_rounds=20,
        verbosity=0, eval_metric="logloss"
    )
    xgb_clf.fit(X_tr, y_dir_tr, eval_set=[(X_val, val["Next_Day_Direction"].values)],
                verbose=False)
    xgb_dir_pred = xgb_clf.predict(X_te)
    xgb_dir_prob = xgb_clf.predict_proba(X_te)[:, 1]
    m = clf_metrics(y_dir_te, xgb_dir_pred, xgb_dir_prob, "XGBoost (Direction)")
    m["Task"] = "Direction"; results["XGBoost Clf"] = m
    joblib.dump(xgb_clf, MODEL_DIR / "xgb_direction_model.joblib")

    # ── LSTM (optional) ──
    if LSTM_AVAILABLE and not FAST_MODE:
        _train_lstm(train, val, test, feat_cols, results)

    # Save comparison
    comp_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    comp_df.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    print(f"\n  Comparison table saved → {METRICS_DIR / 'model_comparison.csv'}")
    print("\n" + comp_df.to_string(index=False))

    return results, xgb_reg, xgb_clf, rf_reg, rf_clf, feat_cols, \
           y_close_te, xgb_pred, y_dir_te, xgb_dir_pred, xgb_dir_prob, \
           rf_dir_prob, test


def _train_lstm(train, val, test, feat_cols, results):
    print("  Training LSTM …")
    all_data = pd.concat([train, val, test]).sort_values(["Ticker", "Date"])
    s = all_data[all_data["Ticker"] == LSTM_TICKER].copy()
    if len(s) < LSTM_SEQ_LEN + 50:
        print(f"  Not enough data for LSTM on {LSTM_TICKER}"); return
    X_raw = s[feat_cols].fillna(0).values
    y_raw = s["Next_Day_Close"].values
    fs    = StandardScaler(); ts = StandardScaler()
    X_sc  = fs.fit_transform(X_raw)
    y_sc  = ts.fit_transform(y_raw.reshape(-1, 1)).ravel()
    Xs, ys = [], []
    for i in range(LSTM_SEQ_LEN, len(X_sc)):
        Xs.append(X_sc[i - LSTM_SEQ_LEN:i]); ys.append(y_sc[i])
    Xs, ys = np.array(Xs), np.array(ys)
    n = len(Xs)
    tr_e = int(n * TRAIN_RATIO); val_e = int(n * (TRAIN_RATIO + VAL_RATIO))
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LSTM_SEQ_LEN, Xs.shape[2])),
        Dropout(0.2), LSTM(32), Dropout(0.2), Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(Xs[:tr_e], ys[:tr_e],
              validation_data=(Xs[tr_e:val_e], ys[tr_e:val_e]),
              epochs=50, batch_size=32,
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
              verbose=0)
    pred_sc = model.predict(Xs[val_e:], verbose=0).ravel()
    pred    = ts.inverse_transform(pred_sc.reshape(-1, 1)).ravel()
    actual  = ts.inverse_transform(ys[val_e:].reshape(-1, 1)).ravel()
    mae_l = mean_absolute_error(actual, pred)
    rmse_l= np.sqrt(mean_squared_error(actual, pred))
    r2_l  = r2_score(actual, pred)
    print(f"  LSTM ({LSTM_TICKER}) MAE={mae_l:.4f} RMSE={rmse_l:.4f} R²={r2_l:.4f}")
    results["LSTM"] = {"Task": f"Close ({LSTM_TICKER})",
                       "MAE": round(mae_l, 4), "RMSE": round(rmse_l, 4), "R2": round(r2_l, 4)}
    model.save(str(MODEL_DIR / "lstm_close_model.h5"))
    joblib.dump(fs, SCALER_DIR / "lstm_feat_scaler.joblib")
    joblib.dump(ts, SCALER_DIR / "lstm_tgt_scaler.joblib")


# ─────────────────────────────────────────────────────────────────────────────
# 10. EVALUATION CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def save_eval_charts(y_close_te, xgb_pred, y_dir_te, xgb_dir_pred,
                     xgb_dir_prob, rf_dir_prob,
                     xgb_reg, rf_reg, feat_cols, test):
    print("\nSaving evaluation charts …")

    # Actual vs predicted
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    axes[0].plot(test["Date"].values[:500], y_close_te[:500],
                 color="navy", lw=1, label="Actual", alpha=0.8)
    axes[0].plot(test["Date"].values[:500], xgb_pred[:500],
                 color="orange", lw=1, label="XGBoost Pred", alpha=0.8)
    axes[0].set_title("Actual vs Predicted — Next-Day Close (first 500 test pts)")
    axes[0].legend()
    axes[1].scatter(y_close_te[:2000], xgb_pred[:2000], alpha=0.15, s=8, color="steelblue")
    mn, mx = y_close_te[:2000].min(), y_close_te[:2000].max()
    axes[1].plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect")
    axes[1].set_title("Actual vs Predicted Scatter"); axes[1].legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "actual_vs_predicted_close.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Confusion matrix + ROC
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cm = confusion_matrix(y_dir_te, xgb_dir_pred)
    ConfusionMatrixDisplay(cm, display_labels=["DOWN/FLAT", "UP"]).plot(
        ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("XGBoost — Confusion Matrix (Direction)")
    auc_xgb = roc_auc_score(y_dir_te, xgb_dir_prob)
    auc_rf  = roc_auc_score(y_dir_te, rf_dir_prob)
    fpr_x, tpr_x, _ = roc_curve(y_dir_te, xgb_dir_prob)
    fpr_r, tpr_r, _ = roc_curve(y_dir_te, rf_dir_prob)
    axes[1].plot(fpr_x, tpr_x, label=f"XGBoost (AUC={auc_xgb:.3f})")
    axes[1].plot(fpr_r, tpr_r, label=f"RF (AUC={auc_rf:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[1].set_title("ROC Curve — Direction"); axes[1].legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "classification_evaluation.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Feature importances
    for model_obj, fname, title in [
        (xgb_reg, "feature_importance_xgb.png", "XGBoost Feature Importance (Close)"),
        (rf_reg,  "feature_importance_rf.png",  "Random Forest Feature Importance (Close)"),
    ]:
        fi = pd.Series(model_obj.feature_importances_, index=feat_cols)
        top20 = fi.sort_values(ascending=False).head(20).sort_values()
        fig, ax = plt.subplots(figsize=(14, 7))
        top20.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
        ax.set_title(title)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / fname, dpi=120, bbox_inches="tight")
        plt.close(fig)

    # Residuals
    residuals = y_close_te - xgb_pred
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].scatter(xgb_pred[:3000], residuals[:3000], alpha=0.15, s=8, color="steelblue")
    axes[0].axhline(0, color="red", ls="--", lw=1)
    axes[0].set_title("Residuals vs Predicted")
    axes[1].hist(residuals, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", ls="--", lw=1)
    axes[1].set_title("Prediction Error Distribution")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "residual_analysis.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Charts saved to {FIGURES_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. SAVE PIPELINE META
# ─────────────────────────────────────────────────────────────────────────────
def save_pipeline_meta(df, feat_cols, train_cut, val_cut):
    meta = {
        "feature_cols": feat_cols,
        "tickers":      sorted(df["Ticker"].unique().tolist()),
        "sectors":      sorted(df["Sector"].unique().tolist()),
        "date_min":     str(df["Date"].min().date()),
        "date_max":     str(df["Date"].max().date()),
        "n_rows":       len(df),
        "random_state": RANDOM_STATE,
        "train_cut":    str(train_cut.date()) if hasattr(train_cut, "date") else str(train_cut),
        "val_cut":      str(val_cut.date())   if hasattr(val_cut,   "date") else str(val_cut),
        "lstm_seq_len": LSTM_SEQ_LEN,
    }
    with open(PIPE_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Pipeline meta saved → {PIPE_META}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline():
    print("=" * 70)
    print("NIFTY 50 AI MARKET ANALYTICS — FULL PIPELINE")
    print("=" * 70)
    df_raw = load_raw()
    data_quality_audit(df_raw)
    df     = clean_data(df_raw)
    run_eda(df)
    df     = engineer_features(df)
    df     = add_indicators(df)
    df     = create_targets(df)
    (train, val, test,
     feat_cols, X_tr, X_val, X_te,
     X_tr_sc, X_val_sc, X_te_sc,
     train_cut, val_cut)           = prepare_split(df)
    (results, xgb_reg, xgb_clf,
     rf_reg, rf_clf, feat_cols,
     y_close_te, xgb_pred,
     y_dir_te, xgb_dir_pred,
     xgb_dir_prob, rf_dir_prob,
     test_df)                      = train_models(train, val, test, feat_cols,
                                                  X_tr, X_val, X_te,
                                                  X_tr_sc, X_val_sc, X_te_sc)
    save_eval_charts(y_close_te, xgb_pred, y_dir_te, xgb_dir_pred,
                     xgb_dir_prob, rf_dir_prob, xgb_reg, rf_reg, feat_cols, test_df)
    save_pipeline_meta(df, feat_cols, train_cut, val_cut)
    print("\n✅ Pipeline complete!")


# ══════════════════════════════════════════════════════════════════════════════
# ██████████████████████  STREAMLIT DASHBOARD  █████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
# This section runs ONLY when the file is invoked via `streamlit run`.
# The `_is_streamlit()` guard below ensures the pipeline code above never
# accidentally triggers Streamlit widgets when running `python ... --pipeline`.
# ══════════════════════════════════════════════════════════════════════════════

def _is_streamlit() -> bool:
    """Return True when Streamlit's runtime context is active."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def run_dashboard():
    import streamlit as st

    # ── Page config ────────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Nifty 50 AI Market Analytics",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── CSS ────────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
      [data-testid="stSidebar"] { background-color: #0B132B; }
      [data-testid="stSidebar"] * { color: #E8EAF0 !important; }
      [data-testid="stSidebar"] .stSelectbox label,
      [data-testid="stSidebar"] .stMultiSelect label { color: #A0AEC0 !important; }
      .sidebar-title  { font-size:22px;font-weight:700;color:#1C77C3;padding:12px 0 4px; }
      .sidebar-subtitle { font-size:11px;color:#7C8DB0;margin-bottom:16px; }
      .kpi-card { background:#F5F7FA;border:1px solid #E5E7EB;border-radius:10px;
                  padding:16px 20px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.07); }
      .kpi-label { font-size:11px;color:#57606A;text-transform:uppercase;
                   letter-spacing:.08em;margin-bottom:6px; }
      .kpi-value { font-size:26px;font-weight:700;color:#1f2328;line-height:1.1; }
      .kpi-delta-pos { font-size:13px;color:#16a34a; }
      .kpi-delta-neg { font-size:13px;color:#dc2626; }
      .section-title { font-size:22px;font-weight:700;color:#0B132B;
                       border-left:4px solid #1C77C3;padding-left:12px;margin:20px 0 12px; }
      .disclaimer { background:#FFF8E1;border-left:4px solid #F59E0B;
                    padding:12px 16px;border-radius:6px;font-size:13px;color:#78350F; }
      div[data-testid="stMetricValue"] { font-size:22px !important; }
    </style>""", unsafe_allow_html=True)

    # ── Dashboard-local paths (same project root) ──────────────────────────────
    _PROJ   = BASE_DIR          # Nifty50_AI_Market_Analytics/
    _DB     = DB_DIR            # ../database/
    _MCOMP  = METRICS_DIR / "model_comparison.csv"
    _FI_XGB = FIGURES_DIR / "feature_importance_xgb.png"
    _FI_RF  = FIGURES_DIR / "feature_importance_rf.png"

    # ── Colour palette & layout template ──────────────────────────────────────
    PAL = {
        "navy":"#0B132B","blue":"#1C77C3","teal":"#2CA6A4",
        "green":"#16a34a","red":"#dc2626","orange":"#ea580c","purple":"#7c3aed",
    }
    PT = dict(paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFBFC",
              font=dict(family="Segoe UI, sans-serif", size=13, color="#1f2328"),
              xaxis=dict(gridcolor="#E5E7EB", linecolor="#D1D5DB"),
              yaxis=dict(gridcolor="#E5E7EB", linecolor="#D1D5DB"))

    # ── Small UI helpers ───────────────────────────────────────────────────────
    def _kpi(label, value, delta="", positive=True):
        dcls  = "kpi-delta-pos" if positive else "kpi-delta-neg"
        dhtml = f'<div class="{dcls}">{delta}</div>' if delta else ""
        return (f'<div class="kpi-card">'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div>{dhtml}</div>')

    def _sec(title):
        st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

    def _fp(v):  return "N/A" if pd.isna(v) else f"₹{v:,.2f}"
    def _pct(v):
        if pd.isna(v): return "N/A"
        return f"+{v:.2f}%" if v >= 0 else f"{v:.2f}%"

    # ── Data loaders (cached) ──────────────────────────────────────────────────
    @st.cache_data(show_spinner="Loading market data…")
    def _load_df():
        csv = CLEAN_CSV if CLEAN_CSV.exists() else MAIN_CSV
        if not csv.exists():
            return None, f"CSV not found at {csv}. Run --pipeline first."
        df = pd.read_csv(csv, low_memory=False)
        try:
            df["Date"] = (pd.to_datetime(df["Date"], utc=True, errors="coerce")
                          .dt.tz_convert("Asia/Kolkata").dt.normalize())
        except Exception:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        df["Year"]  = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        for c in ["Open","High","Low","Close","Volume","Daily_Return"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "Daily_Return" not in df.columns or df["Daily_Return"].isna().mean() > 0.5:
            df["Daily_Return"] = df.groupby("Ticker")["Close"].pct_change()
        df["Roll_Vol_20"] = df.groupby("Ticker")["Daily_Return"].transform(
            lambda x: x.rolling(20).std() * np.sqrt(252))
        return df, None

    @st.cache_resource(show_spinner="Loading ML models…")
    def _load_models():
        _models, _meta = {}, {}
        files = {
            "XGBoost (Close)":        MODEL_DIR / "xgb_close_model.joblib",
            "Random Forest (Close)":  MODEL_DIR / "rf_close_model.joblib",
            "XGBoost (Direction)":    MODEL_DIR / "xgb_direction_model.joblib",
            "Random Forest (Dir)":    MODEL_DIR / "rf_direction_model.joblib",
            "RF (High)":              MODEL_DIR / "rf_high_model.joblib",
            "RF (Low)":               MODEL_DIR / "rf_low_model.joblib",
        }
        for name, path in files.items():
            if path.exists():
                try: _models[name] = joblib.load(path)
                except Exception as e: st.warning(f"Could not load {name}: {e}")
        if PIPE_META.exists():
            with open(PIPE_META) as f:
                _meta = json.load(f)
        return _models, _meta

    # ── Load data ──────────────────────────────────────────────────────────────
    df, load_err = _load_df()
    if load_err or df is None:
        st.error(f"**Data not found.**\n\n{load_err or ''}\n\n"
                 f"Run:  `python Nifty50_AI_Market_Analytics.py --pipeline`")
        st.stop()

    ALL_TICKERS  = sorted(df["Ticker"].unique())
    ALL_SECTORS  = sorted(df["Sector"].unique())
    ALL_YEARS    = sorted(df["Year"].dropna().unique().astype(int))
    TNAME        = (df[["Ticker","Company_Name"]].drop_duplicates()
                    .set_index("Ticker")["Company_Name"].to_dict())

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="sidebar-title">📈 Nifty 50 Analytics</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="sidebar-subtitle">IBM SkillsBuild × Bharat Care × AICTE</div>',
                    unsafe_allow_html=True)
        st.divider()
        PAGE = st.radio("Navigation", [
            "🏠 Executive Overview",
            "🔍 Stock Explorer",
            "📅 Year Explorer",
            "🏭 Sector Analysis",
            "📊 Technical Analysis",
            "🤖 ML Predictions",
            "📉 Model Performance",
            "🗃️ Data Explorer",
        ], label_visibility="collapsed")
        st.divider()
        st.caption("⚠️ Educational purposes only. Not financial advice.")

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — EXECUTIVE OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    if PAGE == "🏠 Executive Overview":
        st.title("Nifty 50 AI Market Analytics")
        st.caption("IBM SkillsBuild Data Analytics with AI Academy | Bharat Care × AICTE")
        _sec("Market Snapshot")
        latest_date = df["Date"].max()
        latest      = df[df["Date"] == latest_date]
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.markdown(_kpi("Companies",    str(df["Ticker"].nunique())),  unsafe_allow_html=True)
        c2.markdown(_kpi("Sectors",      str(df["Sector"].nunique())), unsafe_allow_html=True)
        c3.markdown(_kpi("Total Records",f"{len(df):,}"),               unsafe_allow_html=True)
        c4.markdown(_kpi("Date Range",   f"{df['Date'].min().year}–{df['Date'].max().year}"),
                    unsafe_allow_html=True)
        avg_ret = latest["Daily_Return"].mean() * 100
        c5.markdown(_kpi("Avg Return Today", _pct(avg_ret), "", avg_ret >= 0),
                    unsafe_allow_html=True)
        st.divider()
        _sec("Market-Wide Price Trend (Median Close)")
        ym = df.groupby("Year")["Close"].median().reset_index()
        fig = px.area(ym, x="Year", y="Close",
                      labels={"Close":"Median Close (₹)"},
                      color_discrete_sequence=[PAL["blue"]])
        fig.update_layout(**PT, title="Median Close — All Nifty 50 Stocks")
        fig.update_traces(line_color=PAL["blue"], fillcolor="rgba(28,119,195,0.12)")
        st.plotly_chart(fig, use_container_width=True)
        cg,cl,cv = st.columns(3)
        with cg:
            _sec("🟢 Top Gainers")
            tg = (latest.nlargest(5,"Daily_Return")[["Ticker","Close","Daily_Return"]]
                  .assign(Return=lambda x:(x["Daily_Return"]*100).round(2)))
            st.dataframe(tg[["Ticker","Close","Return"]].rename(columns={"Return":"Return (%)"})\
                         .reset_index(drop=True), hide_index=True, use_container_width=True)
        with cl:
            _sec("🔴 Top Losers")
            tl = (latest.nsmallest(5,"Daily_Return")[["Ticker","Close","Daily_Return"]]
                  .assign(Return=lambda x:(x["Daily_Return"]*100).round(2)))
            st.dataframe(tl[["Ticker","Close","Return"]].rename(columns={"Return":"Return (%)"})\
                         .reset_index(drop=True), hide_index=True, use_container_width=True)
        with cv:
            _sec("📦 Volume Leaders")
            st.dataframe(latest.nlargest(5,"Volume")[["Ticker","Close","Volume"]]
                         .reset_index(drop=True), hide_index=True, use_container_width=True)
        _sec("Sector Overview")
        ss = (latest.groupby("Sector")
              .agg(Avg_Return=("Daily_Return","mean"),Companies=("Ticker","nunique"))
              .reset_index().sort_values("Avg_Return", ascending=False))
        ss["Avg_Return_%"] = (ss["Avg_Return"]*100).round(3)
        fig = px.bar(ss, x="Sector", y="Avg_Return_%",
                     color="Avg_Return_%",
                     color_continuous_scale=["#dc2626","#e5e7eb","#16a34a"],
                     color_continuous_midpoint=0, text="Avg_Return_%")
        fig.update_layout(**PT, title="Sector Avg Return — Latest Day", coloraxis_showscale=False)
        fig.update_traces(texttemplate="%{text:.3f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — STOCK EXPLORER
    # ══════════════════════════════════════════════════════════════════════════
    elif PAGE == "🔍 Stock Explorer":
        _sec("Stock Explorer")
        cs, cys, cye = st.columns([2,1,1])
        sel_tk  = cs.selectbox("Stock",  ALL_TICKERS, format_func=lambda t:f"{t} — {TNAME.get(t,'')}")
        yr_s    = cys.selectbox("From Year", ALL_YEARS, index=0)
        yr_e    = cye.selectbox("To Year",   ALL_YEARS, index=len(ALL_YEARS)-1)
        s = df[(df["Ticker"]==sel_tk)&(df["Year"]>=yr_s)&(df["Year"]<=yr_e)].copy().sort_values("Date")
        if s.empty: st.warning("No data for this selection."); st.stop()
        lr = s.iloc[-1]; pr = s.iloc[-2] if len(s)>1 else lr
        ret_t = (lr["Close"]/pr["Close"]-1)*100
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.markdown(_kpi("Latest Close",  _fp(lr["Close"])),                  unsafe_allow_html=True)
        c2.markdown(_kpi("Daily Return",  _pct(ret_t),"",ret_t>=0),           unsafe_allow_html=True)
        c3.markdown(_kpi("Period High",   _fp(s["High"].max())),               unsafe_allow_html=True)
        c4.markdown(_kpi("Period Low",    _fp(s["Low"].min())),                unsafe_allow_html=True)
        vol_a = s["Daily_Return"].std()*np.sqrt(252)*100
        c5.markdown(_kpi("Ann. Volatility",_pct(vol_a)),                      unsafe_allow_html=True)
        st.divider()
        _sec(f"{sel_tk} — OHLC Candlestick")
        fig = go.Figure(go.Candlestick(x=s["Date"],open=s["Open"],high=s["High"],
                                       low=s["Low"],close=s["Close"],name="OHLC",
                                       increasing_line_color=PAL["green"],
                                       decreasing_line_color=PAL["red"]))
        for sc,col,nm in [("SMA_20",PAL["orange"],"SMA 20"),
                           ("SMA_50",PAL["blue"],  "SMA 50"),
                           ("SMA_200",PAL["purple"],"SMA 200")]:
            if sc in s.columns and s[sc].notna().sum()>5:
                fig.add_trace(go.Scatter(x=s["Date"],y=s[sc],mode="lines",
                                         name=nm,line=dict(color=col,width=1.2)))
        fig.update_layout(**PT, title=f"{sel_tk} Candlestick with Moving Averages",
                          xaxis_rangeslider_visible=False, height=500)
        st.plotly_chart(fig, use_container_width=True)
        _sec("Volume")
        vc = ["green" if r>=0 else "red" for r in s["Daily_Return"].fillna(0)]
        fig2 = go.Figure(go.Bar(x=s["Date"],y=s["Volume"],marker_color=vc,opacity=0.7))
        fig2.update_layout(**PT, title="Daily Volume", height=280)
        st.plotly_chart(fig2, use_container_width=True)
        cr,crv = st.columns(2)
        with cr:
            _sec("Daily Returns")
            rc = ["green" if r>=0 else "red" for r in s["Daily_Return"].fillna(0)]
            fig3 = go.Figure(go.Bar(x=s["Date"],y=s["Daily_Return"]*100,
                                    marker_color=rc,opacity=0.8))
            fig3.update_layout(**PT, title="Daily Return (%)", height=300, yaxis_title="Return (%)")
            st.plotly_chart(fig3, use_container_width=True)
        with crv:
            _sec("Rolling Volatility (20-Day)")
            rv = s["Daily_Return"].rolling(20).std()*np.sqrt(252)*100
            fig4 = go.Figure(go.Scatter(x=s["Date"],y=rv,mode="lines",
                                        line=dict(color=PAL["purple"],width=1.5)))
            fig4.update_layout(**PT, title="20-Day Ann. Volatility (%)", height=300,
                               yaxis_title="Volatility (%)")
            st.plotly_chart(fig4, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3 — YEAR EXPLORER
    # ══════════════════════════════════════════════════════════════════════════
    elif PAGE == "📅 Year Explorer":
        _sec("Historical Year Explorer")
        ct,cy1,cy2 = st.columns([2,1,1])
        sel_tky = ct.selectbox("Stock", ALL_TICKERS,
                                format_func=lambda t:f"{t} — {TNAME.get(t,'')}")
        yr_a = cy1.selectbox("Year A", ALL_YEARS, index=max(0,len(ALL_YEARS)-3))
        yr_b = cy2.selectbox("Year B", ALL_YEARS, index=len(ALL_YEARS)-1)
        def _yr_sum(tk,yr):
            s = df[(df["Ticker"]==tk)&(df["Year"]==yr)].sort_values("Date")
            if s.empty: return None
            return {"Year":yr,"Open (₹)":round(s["Open"].iloc[0],2),
                    "Close (₹)":round(s["Close"].iloc[-1],2),
                    "High (₹)":round(s["High"].max(),2),"Low (₹)":round(s["Low"].min(),2),
                    "Return (%)":round((s["Close"].iloc[-1]/s["Open"].iloc[0]-1)*100,2),
                    "Volatility (%)":round(s["Daily_Return"].std()*np.sqrt(252)*100,2),
                    "Avg Volume":int(s["Volume"].mean()),"Trading Days":len(s)}
        rows = [r for r in [_yr_sum(sel_tky,yr_a),_yr_sum(sel_tky,yr_b)] if r]
        if rows: st.dataframe(pd.DataFrame(rows).T, use_container_width=True)
        _sec(f"{sel_tky} — Annual Return History")
        sy = df[df["Ticker"]==sel_tky].sort_values("Date")
        annual = (sy.groupby("Year")
                  .apply(lambda g:(g["Close"].iloc[-1]/g["Open"].iloc[0]-1)*100)
                  .reset_index(name="Annual_Return_%"))
        bc = [PAL["green"] if v>=0 else PAL["red"] for v in annual["Annual_Return_%"]]
        fig = go.Figure(go.Bar(x=annual["Year"],y=annual["Annual_Return_%"],
                                marker_color=bc,text=annual["Annual_Return_%"].round(1),
                                texttemplate="%{text}%",textposition="outside"))
        fig.update_layout(**PT, title=f"{sel_tky} — Annual Return (%)",
                          yaxis_title="Return (%)", height=420)
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
        _sec("Full Price History")
        fig2 = go.Figure(go.Scatter(x=sy["Date"],y=sy["Close"],mode="lines",
                                    line=dict(color=PAL["blue"],width=1.5),name="Close"))
        for yr_s,col in [(yr_a,PAL["orange"]),(yr_b,PAL["teal"])]:
            yd = sy[sy["Year"]==yr_s]
            if not yd.empty:
                fig2.add_vrect(x0=yd["Date"].min(),x1=yd["Date"].max(),
                               fillcolor=col,opacity=0.07,line_width=0,
                               annotation_text=str(yr_s),annotation_position="top left")
        fig2.update_layout(**PT, title=f"{sel_tky} — Price History", height=420,
                           yaxis_title="Close (₹)")
        st.plotly_chart(fig2, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 4 — SECTOR ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    elif PAGE == "🏭 Sector Analysis":
        _sec("Sector Analysis")
        sel_sec = st.selectbox("Select Sector", ["All Sectors"]+ALL_SECTORS)
        dfs = df if sel_sec=="All Sectors" else df[df["Sector"]==sel_sec]
        sp = (dfs.groupby("Sector")
              .agg(Companies=("Ticker","nunique"),
                   Avg_Daily_Return=("Daily_Return","mean"),
                   Volatility=("Daily_Return","std"),
                   Median_Cap=("Market_Cap","median"))
              .reset_index())
        sp["Avg_Return_%"] = (sp["Avg_Daily_Return"]*100).round(4)
        sp["Ann_Vol_%"]    = (sp["Volatility"]*np.sqrt(252)*100).round(2)
        sp = sp.sort_values("Avg_Return_%",ascending=False)
        ca,cb = st.columns(2)
        with ca:
            _sec("Sector Avg Daily Return")
            fig = px.bar(sp,x="Sector",y="Avg_Return_%",color="Avg_Return_%",
                         color_continuous_scale=["#dc2626","#e5e7eb","#16a34a"],
                         color_continuous_midpoint=0,text="Avg_Return_%")
            fig.update_layout(**PT, coloraxis_showscale=False, height=400)
            fig.update_traces(texttemplate="%{text:.4f}%")
            st.plotly_chart(fig, use_container_width=True)
        with cb:
            _sec("Annualized Volatility")
            fig2 = px.bar(sp,x="Sector",y="Ann_Vol_%",color="Ann_Vol_%",
                          color_continuous_scale="Reds",text="Ann_Vol_%")
            fig2.update_layout(**PT, coloraxis_showscale=False, height=400)
            fig2.update_traces(texttemplate="%{text:.1f}%")
            st.plotly_chart(fig2, use_container_width=True)
        if "Market_Cap" in df.columns:
            _sec("Market Capitalization Treemap")
            cd = (df.groupby(["Sector","Ticker"])["Market_Cap"].median()
                  .reset_index().dropna(subset=["Market_Cap"]))
            cd["Cap_Bn"] = cd["Market_Cap"]/1e9
            fig3 = px.treemap(cd, path=["Sector","Ticker"], values="Cap_Bn",
                              title="Median Market Cap (₹ Billion)",
                              color="Cap_Bn", color_continuous_scale="Blues")
            fig3.update_layout(paper_bgcolor="white")
            st.plotly_chart(fig3, use_container_width=True)
        _sec("Return Distribution by Sector")
        samp = dfs.dropna(subset=["Daily_Return"])
        if len(samp)>100000: samp = samp.sample(100000, random_state=42)
        fig4 = px.box(samp,x="Sector",y="Daily_Return",color="Sector",points=False)
        fig4.update_layout(**PT, showlegend=False, height=450,
                           yaxis_title="Daily Return",
                           title="Daily Return Distribution by Sector")
        st.plotly_chart(fig4, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 5 — TECHNICAL ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    elif PAGE == "📊 Technical Analysis":
        _sec("Technical Analysis")
        ct1,ct2 = st.columns([2,1])
        sel_ta = ct1.selectbox("Stock", ALL_TICKERS, key="ta_tk",
                                format_func=lambda t:f"{t} — {TNAME.get(t,'')}")
        ta_per = ct2.selectbox("Period",["1Y","2Y","5Y","10Y","All"],index=2)
        sta = df[df["Ticker"]==sel_ta].copy().sort_values("Date")
        ld  = sta["Date"].max()
        pm  = {"1Y":365,"2Y":730,"5Y":1825,"10Y":3650,"All":99999}
        sta = sta[sta["Date"]>=(ld-pd.Timedelta(days=pm.get(ta_per,99999)))]
        st.sidebar.markdown("**Indicator Controls**")
        sh_sma  = st.sidebar.checkbox("SMA (20/50/200)", value=True)
        sh_ema  = st.sidebar.checkbox("EMA (20/50)",     value=False)
        sh_bb   = st.sidebar.checkbox("Bollinger Bands", value=True)
        sh_rsi  = st.sidebar.checkbox("RSI",             value=True)
        sh_macd = st.sidebar.checkbox("MACD",            value=True)
        sh_vol  = st.sidebar.checkbox("Volume",          value=True)
        rows_n  = 1+int(sh_rsi)+int(sh_macd)+int(sh_vol)
        rh      = [0.5]+[0.17]*(rows_n-1); rh=[r/sum(rh) for r in rh]
        ri      = {"price":1}; nr=2
        if sh_vol:  ri["volume"]=nr; nr+=1
        if sh_rsi:  ri["rsi"]=nr;    nr+=1
        if sh_macd: ri["macd"]=nr;   nr+=1
        stitles = [f"{sel_ta} Price"] \
                  +(["Volume"]   if sh_vol  else []) \
                  +(["RSI (14)"] if sh_rsi  else []) \
                  +(["MACD"]     if sh_macd else [])
        fig = make_subplots(rows=rows_n,cols=1,shared_xaxes=True,
                            row_heights=rh,vertical_spacing=0.04,
                            subplot_titles=stitles)
        fig.add_trace(go.Candlestick(x=sta["Date"],open=sta["Open"],high=sta["High"],
                                     low=sta["Low"],close=sta["Close"],name="OHLC",
                                     increasing_line_color=PAL["green"],
                                     decreasing_line_color=PAL["red"]),row=1,col=1)
        if sh_sma:
            for sc,col,nm in [("SMA_20","#f97316","SMA 20"),
                               ("SMA_50",PAL["blue"],"SMA 50"),
                               ("SMA_200",PAL["purple"],"SMA 200")]:
                if sc in sta.columns:
                    fig.add_trace(go.Scatter(x=sta["Date"],y=sta[sc],mode="lines",
                                             name=nm,line=dict(color=col,width=1.2)),row=1,col=1)
        if sh_ema:
            for sc,col,nm in [("EMA_20","#14b8a6","EMA 20"),("EMA_50","#6366f1","EMA 50")]:
                if sc in sta.columns:
                    fig.add_trace(go.Scatter(x=sta["Date"],y=sta[sc],mode="lines",
                                             name=nm,line=dict(color=col,width=1,dash="dot")),row=1,col=1)
        if sh_bb and "BB_High" in sta.columns:
            fig.add_trace(go.Scatter(x=sta["Date"],y=sta["BB_High"],mode="lines",name="BB High",
                                     line=dict(color="gray",width=0.8,dash="dash")),row=1,col=1)
            fig.add_trace(go.Scatter(x=sta["Date"],y=sta["BB_Low"],mode="lines",name="BB Low",
                                     fill="tonexty",fillcolor="rgba(150,150,150,0.1)",
                                     line=dict(color="gray",width=0.8,dash="dash")),row=1,col=1)
        if sh_vol:
            vc = ["green" if r>=0 else "red" for r in sta["Daily_Return"].fillna(0)]
            fig.add_trace(go.Bar(x=sta["Date"],y=sta["Volume"],marker_color=vc,
                                 name="Volume",opacity=0.6),row=ri["volume"],col=1)
        if sh_rsi and "RSI_14" in sta.columns:
            fig.add_trace(go.Scatter(x=sta["Date"],y=sta["RSI_14"],mode="lines",
                                     name="RSI",line=dict(color=PAL["teal"],width=1.2)),
                          row=ri["rsi"],col=1)
            fig.add_hline(y=70,row=ri["rsi"],col=1,line_color="red",line_dash="dash",line_width=0.8)
            fig.add_hline(y=30,row=ri["rsi"],col=1,line_color="green",line_dash="dash",line_width=0.8)
        if sh_macd and "MACD" in sta.columns:
            fig.add_trace(go.Scatter(x=sta["Date"],y=sta["MACD"],mode="lines",
                                     name="MACD",line=dict(color=PAL["blue"],width=1.2)),
                          row=ri["macd"],col=1)
            if "MACD_Sig" in sta.columns:
                fig.add_trace(go.Scatter(x=sta["Date"],y=sta["MACD_Sig"],mode="lines",
                                         name="Signal",line=dict(color=PAL["red"],width=1)),
                              row=ri["macd"],col=1)
            if "MACD_Diff" in sta.columns:
                mc = ["green" if v>=0 else "red" for v in sta["MACD_Diff"].fillna(0)]
                fig.add_trace(go.Bar(x=sta["Date"],y=sta["MACD_Diff"],marker_color=mc,
                                     name="MACD Hist",opacity=0.5),row=ri["macd"],col=1)
        fig.update_layout(paper_bgcolor="white",plot_bgcolor="#FAFBFC",
                          font=dict(family="Segoe UI, sans-serif",size=12),
                          height=720,showlegend=True,xaxis_rangeslider_visible=False,
                          legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="right",x=1))
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 6 — ML PREDICTIONS
    # ══════════════════════════════════════════════════════════════════════════
    elif PAGE == "🤖 ML Predictions":
        _sec("Machine Learning Predictions")
        st.markdown(
            '<div class="disclaimer">⚠️ <strong>Disclaimer:</strong> All predictions are '
            "statistical estimates for educational purposes only. Not financial advice. "
            "Past model performance does not guarantee future results.</div>",
            unsafe_allow_html=True)
        st.divider()
        models_d, meta = _load_models()
        cp,cm = st.columns([2,1])
        sel_ptk = cp.selectbox("Stock", ALL_TICKERS, key="pred_tk",
                                format_func=lambda t:f"{t} — {TNAME.get(t,'')}")
        sel_mn  = cm.selectbox("Model", list(models_d.keys()) if models_d else ["No models loaded"])
        sp = df[df["Ticker"]==sel_ptk].sort_values("Date")
        if sp.empty:
            st.warning("No data for this stock.")
        elif not models_d:
            st.info("No models found. Run:  `python Nifty50_AI_Market_Analytics.py --pipeline`")
        else:
            feat_cols = [c for c in meta.get("feature_cols",[]) if c in df.columns]
            if feat_cols and sel_mn in models_d:
                model  = models_d[sel_mn]
                sp_ml  = sp.dropna(subset=["Close"]+feat_cols).copy()
                if len(sp_ml) > 50:
                    te_s   = sp_ml.iloc[int(len(sp_ml)*0.8):]
                    Xp     = te_s[feat_cols].fillna(0).values
                    try:
                        yp     = model.predict(Xp)
                        ya     = te_s.groupby("Ticker")["Close"].shift(-1).iloc[:-1].values
                        yp     = yp[:len(ya)]; dp = te_s["Date"].values[:len(ya)]
                        mae_v  = mean_absolute_error(ya, yp)
                        r2_v   = r2_score(ya, yp)
                        _sec("Actual vs Predicted — Close Price")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=dp,y=ya,mode="lines",name="Actual",
                                                 line=dict(color=PAL["navy"],width=1.5)))
                        fig.add_trace(go.Scatter(x=dp,y=yp,mode="lines",name="Predicted",
                                                 line=dict(color=PAL["orange"],width=1.5,dash="dot")))
                        fig.update_layout(**PT, height=400,
                                          title=f"{sel_ptk} — Actual vs Predicted Next-Day Close",
                                          yaxis_title="Price (₹)")
                        st.plotly_chart(fig, use_container_width=True)
                        mc1,mc2 = st.columns(2)
                        mc1.markdown(_kpi("MAE", _fp(mae_v)), unsafe_allow_html=True)
                        mc2.markdown(_kpi("R²", f"{r2_v:.4f}"), unsafe_allow_html=True)
                        st.divider(); _sec("Latest Prediction")
                        lf   = sp_ml.iloc[-1][feat_cols].fillna(0).values.reshape(1,-1)
                        pn   = model.predict(lf)[0]
                        cc   = sp_ml.iloc[-1]["Close"]
                        chg  = (pn/cc-1)*100
                        p1,p2,p3 = st.columns(3)
                        p1.markdown(_kpi("Current Close",    _fp(cc)),             unsafe_allow_html=True)
                        p2.markdown(_kpi("Predicted Next",   _fp(pn)),             unsafe_allow_html=True)
                        p3.markdown(_kpi("Expected Change",  _pct(chg),"",chg>=0), unsafe_allow_html=True)
                        if "XGBoost (Direction)" in models_d:
                            dm  = models_d["XGBoost (Direction)"]
                            dp_ = dm.predict(lf)[0]
                            prb = dm.predict_proba(lf)[0,1]
                            st.metric("Direction",
                                      "📈 UP" if dp_==1 else "📉 DOWN / FLAT",
                                      f"Probability: {prb:.2%}")
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                else:
                    st.warning("Not enough data for this stock.")
            else:
                st.info("Feature metadata missing. Run the pipeline first.")

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 7 — MODEL PERFORMANCE
    # ══════════════════════════════════════════════════════════════════════════
    elif PAGE == "📉 Model Performance":
        _sec("Model Performance & Comparison")
        if _MCOMP.exists():
            pf = pd.read_csv(_MCOMP)
            st.dataframe(pf, use_container_width=True, hide_index=True)
            cr = pf[pf["Task"]=="Close"].dropna(subset=["R2"])
            if not cr.empty:
                _sec("Regression Metrics — Next-Day Close")
                cm1,cm2,cm3 = st.columns(3)
                for cw,met in zip([cm1,cm2,cm3],["MAE","RMSE","R2"]):
                    if met in cr.columns:
                        f = px.bar(cr,x="Model",y=met,text=met,color="Model",
                                   color_discrete_sequence=px.colors.qualitative.Set2)
                        f.update_layout(**PT,showlegend=False,height=350,title=met)
                        f.update_traces(texttemplate="%{text:.3f}",textposition="outside")
                        cw.plotly_chart(f,use_container_width=True)
            dr = pf[pf["Task"]=="Direction"].dropna(subset=["Dir_Accuracy"])
            if not dr.empty:
                _sec("Classification Metrics — Direction")
                f2 = px.bar(dr,x="Model",y=["Dir_Accuracy","Dir_F1","ROC_AUC"],
                            barmode="group",labels={"value":"Score","variable":"Metric"})
                f2.update_layout(**PT,height=400,title="Direction Prediction Metrics")
                st.plotly_chart(f2, use_container_width=True)
            st.info("Metrics on chronological test set (last 15% of dates).")
        else:
            st.warning(f"No metrics file found at `{_MCOMP}`. Run the pipeline first.")
        _sec("Feature Importance")
        cfi1,cfi2 = st.columns(2)
        with cfi1:
            if _FI_XGB.exists():
                st.image(str(_FI_XGB), caption="XGBoost Feature Importance",
                         use_container_width=True)
            else:
                st.info("XGBoost chart not found. Run the pipeline first.")
        with cfi2:
            if _FI_RF.exists():
                st.image(str(_FI_RF), caption="Random Forest Feature Importance",
                         use_container_width=True)
            else:
                st.info("RF chart not found. Run the pipeline first.")

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 8 — DATA EXPLORER
    # ══════════════════════════════════════════════════════════════════════════
    elif PAGE == "🗃️ Data Explorer":
        _sec("Data Explorer")
        with st.expander("🔧 Filter Options", expanded=True):
            cd1,cd2,cd3 = st.columns(3)
            sel_tks  = cd1.multiselect("Stocks",  ALL_TICKERS, default=ALL_TICKERS[:3])
            sel_secs = cd2.multiselect("Sectors", ALL_SECTORS, default=[])
            yr_rng   = cd3.slider("Year Range",
                                   min_value=int(min(ALL_YEARS)),
                                   max_value=int(max(ALL_YEARS)),
                                   value=(2020,int(max(ALL_YEARS))))
        filt = df.copy()
        if sel_tks:  filt = filt[filt["Ticker"].isin(sel_tks)]
        if sel_secs: filt = filt[filt["Sector"].isin(sel_secs)]
        filt = filt[(filt["Year"]>=yr_rng[0])&(filt["Year"]<=yr_rng[1])]
        def_cols = ["Date","Ticker","Company_Name","Sector","Open","High","Low",
                    "Close","Volume","Daily_Return"]
        avail  = [c for c in def_cols if c in filt.columns]
        extra  = [c for c in filt.columns if c not in avail]
        sel_cols = st.multiselect("Columns to display", avail+extra, default=avail)
        st.caption(f"Showing {len(filt):,} rows after filters.")
        st.dataframe((filt[sel_cols] if sel_cols else filt).head(5000),
                     use_container_width=True)

        @st.cache_data
        def _to_csv(data): return data.to_csv(index=False).encode("utf-8")

        st.download_button("⬇️ Download Filtered Data (CSV)",
                           data=_to_csv(filt[sel_cols] if sel_cols else filt),
                           file_name="nifty50_filtered.csv", mime="text/csv")
        with st.expander("📊 Summary Statistics"):
            nd = (filt[sel_cols] if sel_cols else filt).select_dtypes(include=[np.number])
            st.dataframe(nd.describe(), use_container_width=True)
        with st.expander("🔥 Correlation Heatmap"):
            cc = [c for c in ["Close","Volume","Daily_Return","Roll_Vol_20",
                               "SMA_20","RSI_14","MACD","ATR_14"] if c in filt.columns]
            if len(cc) >= 3:
                fig = px.imshow(filt[cc].corr(), text_auto=".2f",
                                color_continuous_scale="RdYlGn", zmin=-1, zmax=1,
                                title="Correlation Heatmap — Filtered Data")
                fig.update_layout(paper_bgcolor="white", height=500)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Correlation ≠ causation.")
            else:
                st.info("Not enough numeric columns for heatmap.")

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<p style='text-align:center;font-size:11px;color:#888;'>"
        "Nifty 50 AI Market Analytics &nbsp;|&nbsp; "
        "IBM SkillsBuild Data Analytics with AI Academy &nbsp;|&nbsp; "
        "Bharat Care × AICTE &nbsp;|&nbsp; Educational purposes only — Not financial advice"
        "</p>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nifty 50 AI Market Analytics — pipeline + dashboard in one file"
    )
    parser.add_argument("--pipeline",  action="store_true",
                        help="Run the full ML pipeline (data audit → cleaning → EDA → "
                             "feature engineering → model training → evaluation charts)")
    parser.add_argument("--dashboard", action="store_true",
                        help="After running the pipeline, launch the Streamlit dashboard")
    args = parser.parse_args()

    if args.pipeline:
        run_pipeline()
        if args.dashboard:
            import subprocess
            subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
    else:
        # Called without arguments: print usage hint
        print(__doc__)
        print("\nHint: if you reached this via `streamlit run`, the dashboard is already active.")

# Streamlit entry — runs when invoked via `streamlit run Nifty50_AI_Market_Analytics.py`
if _is_streamlit():
    run_dashboard()
