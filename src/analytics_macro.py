from __future__ import annotations

from pathlib import Path
import datetime as dt
import warnings

import numpy as np
import pandas as pd

from analytics.portfolio import load_portfolio_series as _load_portfolio_series

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "macro_cache"


def compute_returns(prices: pd.DataFrame | pd.Series, freq: str) -> pd.DataFrame | pd.Series:
    prices = prices.sort_index()
    if freq == "Weekly":
        prices = prices.resample("W-FRI").last()
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return returns


def rolling_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std()


def rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    aligned = pd.concat([a, b], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    return aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])


def rolling_beta(port_ret: pd.Series, bench_ret: pd.Series, window: int) -> pd.Series:
    aligned = pd.concat([port_ret, bench_ret], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    y = aligned.iloc[:, 0]
    x = aligned.iloc[:, 1]
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return cov / var


def beta_and_r2(port_ret: pd.Series, x_ret: pd.Series) -> tuple[float, float]:
    aligned = pd.concat([port_ret, x_ret], axis=1).dropna()
    if len(aligned) < 2:
        return np.nan, np.nan
    y = aligned.iloc[:, 0].to_numpy()
    x = aligned.iloc[:, 1].to_numpy()
    if np.allclose(x.var(), 0.0):
        return np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 0.0 if ss_tot == 0 else 1 - (ss_res / ss_tot)
    return slope, r2


def max_drawdown(price_series: pd.Series) -> tuple[float, pd.Series]:
    if price_series.empty:
        return np.nan, pd.Series(dtype=float)
    roll_max = price_series.cummax()
    drawdown = (price_series / roll_max) - 1.0
    return drawdown.min(), drawdown


def normalize_to_100(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    if prices.empty:
        return prices
    first = prices.iloc[0]
    return (prices / first) * 100.0


def load_portfolio_series() -> pd.Series:
    return _load_portfolio_series()


def load_fred_series(
    path: str,
    date_col: str = "DATE",
    value_col: str | None = None,
) -> pd.Series | None:
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if date_col not in df.columns:
        return None
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    if df.empty:
        return None
    if value_col and value_col in df.columns:
        series = pd.to_numeric(df[value_col], errors="coerce")
    else:
        numeric_cols = [c for c in df.columns if c != date_col]
        best = None
        best_non_na = -1
        for c in numeric_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            non_na = s.notna().sum()
            if non_na > best_non_na:
                best_non_na = non_na
                best = c
        if best is None:
            return None
        series = pd.to_numeric(df[best], errors="coerce")
    series = series.dropna()
    if series.empty:
        return None
    return series


def load_ticker_prices(
    tickers: list[str],
    start: dt.date | None = None,
    end: dt.date | None = None,
    cache_days: int = 1,
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tickers = [t.upper() for t in tickers]
    cached = {}
    missing = []

    for ticker in tickers:
        cache_path = CACHE_DIR / f"{ticker}.csv"
        if cache_path.exists():
            age_days = (dt.datetime.now() - dt.datetime.fromtimestamp(cache_path.stat().st_mtime)).days
            try:
                series = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")["Close"]
                series = series.sort_index()
            except Exception:
                series = None
            if series is not None and not series.empty:
                if age_days <= cache_days or cache_days <= 0:
                    cached[ticker] = series
                else:
                    cached[ticker] = series
                    missing.append(ticker)
            else:
                missing.append(ticker)
        else:
            missing.append(ticker)

    if missing:
        try:
            import yfinance as yf  # local import to avoid hard dependency in tests
            df = yf.download(
                missing,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )["Close"]
        except Exception as exc:
            warnings.warn(f"[WARN] Failed to download {missing}: {exc}")
            df = pd.DataFrame()

        if isinstance(df, pd.Series):
            df = df.to_frame(name=missing[0])

        for ticker in missing:
            if ticker in df.columns:
                series = df[ticker].dropna().sort_index()
            else:
                series = pd.Series(dtype=float)
            if series.empty:
                warnings.warn(f"[WARN] No data for {ticker}; skipping.")
                continue
            cache_path = CACHE_DIR / f"{ticker}.csv"
            series.to_frame("Close").to_csv(cache_path)
            cached[ticker] = series

    if not cached:
        return pd.DataFrame()
    combined = pd.concat(cached, axis=1)
    if isinstance(combined.columns, pd.MultiIndex):
        combined.columns = combined.columns.get_level_values(0)
    combined = combined.sort_index()
    if start:
        combined = combined[combined.index >= pd.to_datetime(start)]
    if end:
        combined = combined[combined.index <= pd.to_datetime(end)]
    return combined
