
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import requests
import plotly.graph_objects as go
import streamlit as st
import os

def fetch_alpha_vantage(symbol: str, api_key: str, years: int = 3) -> pd.DataFrame:
    url = "https://www.alphavantage.co/query"
    pars = {"function": "TIME_SERIES_DAILY", "symbol": symbol, "outputsize": "full", "apikey": api_key}
    r = requests.get(url, params=pars, timeout=30)
    d = r.json()
    key = "Time Series (Daily)"
    if key not in d:
        msg = d.get("Note") or d.get("Error Message") or "Unknown error"
        raise RuntimeError(msg)
    f = pd.DataFrame.from_dict(d[key], orient="index").apply(pd.to_numeric, errors="coerce")
    f.index = pd.to_datetime(f.index)
    f = f.sort_index()
    rename_map = {"1. open": "Open", "2. high": "High", "3. low": "Low", "4. close": "Close", "5. volume": "Volume"}
    f = f.rename(columns=rename_map)
    cutoff = datetime.now() - timedelta(days=365 * years + 5)
    f = f[f.index >= cutoff]
    keep_cols = ["Open", "High", "Low", "Close", "Volume"]
    existing = [c for c in keep_cols if c in f.columns]
    f = f[existing].dropna(how="any")
    if f.empty:
        raise RuntimeError(f"No recent data returned for {symbol} after trimming to {years} years.")
    return f

def compute_returns(prices: pd.Series, log=True) -> pd.Series:
    if log:
        rets = np.log(prices).diff()
    else:
        rets = prices.pct_change()
    return rets.dropna()

def fit_student_t(returns: pd.Series):
    df, loc, scale = stats.t.fit(returns.values)
    return df, loc, scale

def var_cvar_t(alpha: float, df: float, loc: float, scale: float):
    q = stats.t.ppf(alpha, df, loc=loc, scale=scale)
    xs = np.linspace(q - 10 * scale, q, 4000)
    pdf = stats.t.pdf((xs - loc) / scale, df) / scale
    es = (np.trapz(xs * pdf, xs) / alpha)
    return q, es

def sharpe_approx(returns: pd.Series, trading_days=252):
    mu_d = returns.mean()
    sig_d = returns.std(ddof=1)
    mu_ann = mu_d * trading_days
    sig_ann = sig_d * np.sqrt(trading_days)
    sharpe = 0.0 if sig_ann == 0 else mu_ann / sig_ann
    return mu_d, sig_d, mu_ann, sig_ann, sharpe

def rolling_vol(returns: pd.Series, window: int = 30, trading_days=252) -> pd.Series:
    return returns.rolling(window).std(ddof=1) * np.sqrt(trading_days)

def drawdown_stats(prices: pd.Series):
    peaks = prices.cummax()
    dd = prices / peaks - 1.0
    max_dd = dd.min()
    current_dd = dd.iloc[-1]
    in_dd = dd < 0
    durations = []
    count = 0
    for v in in_dd.values:
        if v:
            count += 1
        else:
            if count > 0:
                durations.append(count)
            count = 0
    if count > 0:
        durations.append(count)
    longest = int(max(durations) if durations else 0)
    return dd, float(max_dd), float(current_dd), longest

def regime_labels(rolling_vol_series: pd.Series, low_q=0.33, high_q=0.66):
    rv = rolling_vol_series.dropna()
    if rv.empty:
        return "N/A", np.nan, np.nan
    v = rv.iloc[-1]
    lo = rv.quantile(low_q)
    hi = rv.quantile(high_q)
    if v <= lo:
        r = "Low Vol"
    elif v >= hi:
        r = "High Vol"
    else:
        r = "Medium Vol"
    return r, float(lo), float(hi)

def trend_label(prices: pd.Series, fast=50, slow=200):
    ma_fast = prices.rolling(fast).mean()
    ma_slow = prices.rolling(slow).mean()
    if np.isnan(ma_fast.iloc[-1]) or np.isnan(ma_slow.iloc[-1]):
        return "N/A", ma_fast, ma_slow
    if ma_fast.iloc[-1] > ma_slow.iloc[-1]:
        return f"Uptrend (MA{fast}>{slow})", ma_fast, ma_slow
    elif ma_fast.iloc[-1] < ma_slow.iloc[-1]:
        return f"Downtrend (MA{fast}<{slow})", ma_fast, ma_slow
    else:
        return f"Neutral (MA{fast}≈{slow})", ma_fast, ma_slow

def simulate_paths_t(spot: float, days: int, n_paths: int, df: float, loc: float, scale: float, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rets = stats.t.rvs(df, loc=loc, scale=scale, size=(days, n_paths), random_state=rng)
    paths = np.empty_like(rets)
    paths[0, :] = spot * np.exp(rets[0, :])
    for t in range(1, days):
        paths[t, :] = paths[t - 1, :] * np.exp(rets[t, :])
    return paths

def t_pdf_overlay(returns: pd.Series, df: float, loc: float, scale: float, bins=50):
    hist_y, hist_x = np.histogram(returns, bins=bins, density=True)
    mid = 0.5 * (hist_x[1:] + hist_x[:-1])
    x_grid = np.linspace(mid.min(), mid.max(), 400)
    pdf = stats.t.pdf((x_grid - loc) / scale, df) / scale
    return mid, hist_y, x_grid, pdf

st.set_page_config(page_title="Price-Only Risk Dashboard", layout="wide")
st.title("Price-Only Risk Dashboard (Student’s t)")

with st.sidebar:
    st.header("Inputs")
    symbol = st.text_input("Ticker", value="AAPL").strip().upper()
    years = st.slider("History (years)", 2, 10, 3, 1)
    window_vol = st.slider("Rolling Vol Window (days)", 10, 90, 30, 5)
    ma_fast = st.slider("Fast MA (days)", 10, 100, 50, 5)
    ma_slow = st.slider("Slow MA (days)", 100, 300, 200, 10)
    alpha_var = st.selectbox("VaR/CVaR Tail (α)", options=[0.01, 0.025, 0.05], index=2)
    paths_to_show = st.slider("Monte Carlo Paths (display)", 5, 100, 25, 5)
    big_runs = st.select_slider("Monte Carlo Finals (histogram)", options=[10000, 20000, 50000, 100000], value=50000)
    sim_days = st.select_slider("Simulation Horizon (days)", options=[126, 189, 252], value=252)
    seed = st.number_input("Random Seed", value=42, step=1)
    api_key_input = st.text_input("Alpha Vantage API Key", value=os.getenv("ALPHAVANTAGE_API_KEY", ""), type="password", help="Leave blank if configured in environment as ALPHAVANTAGE_API_KEY")
    run = st.button("Run / Refresh", type="primary")

if not run:
    st.info("Set parameters in the sidebar and click Run / Refresh.")
    st.stop()

api_key = api_key_input or os.getenv("ALPHAVANTAGE_API_KEY")
if not api_key:
    st.error("Alpha Vantage API key is required. Enter it in the sidebar or set ALPHAVANTAGE_API_KEY in your environment.")
    st.stop()

with st.spinner(f"Fetching {years}y of daily prices for {symbol}..."):
    try:
        px = fetch_alpha_vantage(symbol, api_key, years=years)
    except Exception as e:
        st.error(f"Alpha Vantage error for {symbol}: {e}")
        st.stop()

prices = px["Close"].copy()
prices.index = pd.to_datetime(prices.index)
if getattr(prices.index, "tz", None) is not None:
    prices = prices.tz_localize(None)

rets = compute_returns(prices, log=True)
df_t, loc_t, sc_t = fit_student_t(rets)
mu_d, sig_d, mu_ann, sig_ann, sharpe = sharpe_approx(rets, trading_days=252)
VaR_d, CVaR_d = var_cvar_t(alpha_var, df_t, loc_t, sc_t)
rv = rolling_vol(rets, window=window_vol, trading_days=252)
vol_regime, vol_lo, vol_hi = regime_labels(rv)
dd_series, max_dd, curr_dd, dd_days = drawdown_stats(prices)
trend_regime, ma_f, ma_s = trend_label(prices, fast=ma_fast, slow=ma_slow)
paths = simulate_paths_t(prices.iloc[-1], days=sim_days, n_paths=paths_to_show, df=df_t, loc=loc_t, scale=sc_t, seed=seed)
finals = simulate_paths_t(prices.iloc[-1], days=sim_days, n_paths=big_runs, df=df_t, loc=loc_t, scale=sc_t, seed=seed+1)[-1, :]
low, high = np.percentile(finals, [1.25, 98.75])
trim = finals[(finals >= low) & (finals <= high)]
mid, hist_y, x_grid, pdf = t_pdf_overlay(rets, df_t, loc_t, sc_t, bins=50)

colA, colB, colC, colD = st.columns(4)
colA.metric("Spot", f"${prices.iloc[-1]:,.2f}")
colB.metric("Ann. Mean (μ)", f"{100*mu_ann:.2f}%")
colC.metric("Ann. Vol (σ)", f"{100*sig_ann:.2f}%")
colD.metric("Sharpe (rf≈0)", f"{sharpe:.2f}")

colE, colF, colG, colH = st.columns(4)
colE.metric(f"{int((1-alpha_var)*100)}% 1-Day VaR", f"{100*VaR_d:.2f}%")
colF.metric(f"{int((1-alpha_var)*100)}% 1-Day CVaR", f"{100*CVaR_d:.2f}%")
colG.metric("Skewness", f"{stats.skew(rets):.2f}")
colH.metric("Excess Kurtosis", f"{stats.kurtosis(rets, fisher=True):.2f}")

jb_stat, jb_p = stats.jarque_bera(rets)
st.caption(f"Normality (Jarque–Bera): JB={jb_stat:.2f}, p-value={jb_p:.4f}")

price_fig = go.Figure()
price_fig.add_trace(go.Scatter(x=prices.index, y=prices, name="Close", mode="lines"))
if ma_f.notna().sum() > 0:
    price_fig.add_trace(go.Scatter(x=ma_f.index, y=ma_f, name=f"MA {ma_fast}", mode="lines"))
if ma_s.notna().sum() > 0:
    price_fig.add_trace(go.Scatter(x=ma_s.index, y=ma_s, name=f"MA {ma_slow}", mode="lines"))
price_fig.update_layout(title=f"{symbol} Price & Trend — {trend_regime}", xaxis_title="Date", yaxis_title="Price ($)", template="plotly_white", legend_title_text="")
st.plotly_chart(price_fig, use_container_width=True)

dd_fig = go.Figure()
dd_fig.add_trace(go.Scatter(x=dd_series.index, y=100*dd_series, fill="tozeroy", name="Drawdown (%)", mode="lines"))
dd_fig.update_layout(title=f"{symbol} Drawdown (Max: {100*max_dd:.1f}%, Current: {100*curr_dd:.1f}%, Longest: {dd_days} days)", xaxis_title="Date", yaxis_title="Drawdown (%)", template="plotly_white")
st.plotly_chart(dd_fig, use_container_width=True)

rv_fig = go.Figure()
rv_fig.add_trace(go.Scatter(x=rv.index, y=100*rv, name=f"Rolling Vol ({window_vol}d, annualized)", mode="lines"))
if not np.isnan(vol_lo):
    rv_fig.add_hline(y=100*vol_lo, line_dash="dot", annotation_text="Low quantile", annotation_position="top left")
if not np.isnan(vol_hi):
    rv_fig.add_hline(y=100*vol_hi, line_dash="dot", annotation_text="High quantile", annotation_position="bottom left")
rv_fig.update_layout(title=f"{symbol} Volatility — Regime: {vol_regime}", xaxis_title="Date", yaxis_title="Vol (%)", template="plotly_white")
st.plotly_chart(rv_fig, use_container_width=True)

dist_fig = go.Figure()
dist_fig.add_trace(go.Bar(x=mid, y=hist_y, name="Returns (hist, density)"))
dist_fig.add_trace(go.Scatter(x=x_grid, y=pdf, name="Fitted t-PDF", mode="lines"))
dist_fig.update_layout(title=f"{symbol} Daily Log Returns — Fitted Student’s t (df={df_t:.1f})", xaxis_title="Daily Log Return", yaxis_title="Density", template="plotly_white")
st.plotly_chart(dist_fig, use_container_width=True)

path_fig = go.Figure()
for i in range(min(paths_to_show, paths.shape[1])):
    path_fig.add_trace(go.Scatter(x=np.arange(1, sim_days+1), y=paths[:, i], mode="lines", line=dict(width=1), name=f"Path {i+1}", hovertemplate="Day %{x}<br>$%{y:.2f}<extra></extra>"))
path_fig.update_layout(title=f"{symbol} Monte Carlo Simulated Price Paths — {sim_days} Trading Days, {paths_to_show} Runs", xaxis_title="Day", yaxis_title="Price ($)", template="plotly_white", showlegend=False)
st.plotly_chart(path_fig, use_container_width=True)

h = go.Figure()
h.add_trace(go.Histogram(x=trim, nbinsx=30, marker_line_color="black", marker_line_width=1, opacity=0.9, hovertemplate="Terminal Price: $%{x:.2f}<br>Count: %{y}<extra></extra>"))
h.update_layout(title=dict(text=f"{symbol} — 1-Year (≈{sim_days}d) Terminal Price Distribution<br><sup>Spot ${prices.iloc[-1]:.2f} · {big_runs:,} Sims · t-driven</sup>", x=0.5), xaxis_title="Simulated 1-Year Price", yaxis_title="Count", template="plotly_white", bargap=0.1)
st.plotly_chart(h, use_container_width=True)

st.caption("Notes: Student’s t returns capture fat tails. VaR/CVaR are 1-day return measures; Monte Carlo uses t-distributed daily log returns. Risk-free assumed ≈ 0 for Sharpe. Data source: Alpha Vantage.")
