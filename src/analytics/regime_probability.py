from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from analytics.constants import ANALYSIS_END, ANALYSIS_START
from analytics.regimes import compute_regime_features, label_regimes, load_proxy_prices, returns_from_prices


PROXIES = ["SPY", "QQQ", "HYG", "TLT", "USO", "UUP", "GLD", "TIP"]
RISK_OFF_LABEL = "Risk-Off / Credit Stress"


@dataclass
class SplitConfig:
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_weekly_returns(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    prices = load_proxy_prices(start, end)
    if prices.empty:
        return pd.DataFrame()
    daily_ret = prices.pct_change().replace([np.inf, -np.inf], np.nan)
    weekly = (1 + daily_ret).resample("W-FRI").prod() - 1
    return weekly.dropna(how="all")


def make_weekly_regimes(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    prices = load_proxy_prices(start, end)
    features = compute_regime_features(prices, freq="Weekly")
    labels = label_regimes(features, "Balanced")
    return labels.dropna()


def make_labels(regimes: pd.Series, horizon_weeks: int) -> pd.Series:
    if regimes.empty:
        return pd.Series(dtype=int)
    y = []
    idx = regimes.index
    for i in range(len(idx) - horizon_weeks):
        future = regimes.iloc[i + 1:i + 1 + horizon_weeks]
        y.append(int((future == RISK_OFF_LABEL).any()))
    return pd.Series(y, index=idx[:-horizon_weeks])


def build_features(weekly_returns: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    if weekly_returns.empty:
        return pd.DataFrame()
    df = pd.DataFrame(index=weekly_returns.index)
    df["spy_4w"] = weekly_returns.get("SPY", pd.Series(index=df.index)).rolling(4).sum()
    df["spy_12w"] = weekly_returns.get("SPY", pd.Series(index=df.index)).rolling(12).sum()
    df["qqq_4w"] = weekly_returns.get("QQQ", pd.Series(index=df.index)).rolling(4).sum()
    df["qqq_12w"] = weekly_returns.get("QQQ", pd.Series(index=df.index)).rolling(12).sum()
    df["spy_vol_8w"] = weekly_returns.get("SPY", pd.Series(index=df.index)).rolling(8).std()
    df["qqq_vol_8w"] = weekly_returns.get("QQQ", pd.Series(index=df.index)).rolling(8).std()
    df["spy_dd_26w"] = (1 + weekly_returns.get("SPY", pd.Series(index=df.index))).rolling(26).apply(lambda x: (x.prod() / x.cummax().max()) - 1, raw=False)
    df["hyg_dd_26w"] = (1 + weekly_returns.get("HYG", pd.Series(index=df.index))).rolling(26).apply(lambda x: (x.prod() / x.cummax().max()) - 1, raw=False)
    df["hyg_4w"] = weekly_returns.get("HYG", pd.Series(index=df.index)).rolling(4).sum()
    if "HYG" in weekly_returns.columns and "TLT" in weekly_returns.columns:
        df["hyg_tlt_4w"] = (weekly_returns["HYG"] - weekly_returns["TLT"]).rolling(4).sum()
    df["uup_4w"] = weekly_returns.get("UUP", pd.Series(index=df.index)).rolling(4).sum()
    df["tlt_4w"] = weekly_returns.get("TLT", pd.Series(index=df.index)).rolling(4).sum()
    df["uso_4w"] = weekly_returns.get("USO", pd.Series(index=df.index)).rolling(4).sum()
    if "USO" in weekly_returns.columns and "TIP" in weekly_returns.columns:
        df["uso_tip_4w"] = (weekly_returns["USO"] - weekly_returns["TIP"]).rolling(4).sum()

    regimes = regimes.reindex(df.index).ffill()
    df["regime"] = regimes
    df["weeks_in_regime"] = regimes.groupby((regimes != regimes.shift()).cumsum()).cumcount() + 1
    df = pd.get_dummies(df, columns=["regime"], prefix="regime")
    df = df.dropna()
    return df


def default_splits() -> SplitConfig:
    return SplitConfig(
        train_end=pd.Timestamp("2021-12-31"),
        val_end=pd.Timestamp("2022-12-31"),
        test_start=ANALYSIS_START,
        test_end=ANALYSIS_END,
    )


def resolve_splits(index: pd.DatetimeIndex, splits: SplitConfig) -> SplitConfig:
    if index.empty:
        return splits
    min_dt = index.min()
    max_dt = index.max()
    train_end = min(splits.train_end, max_dt)
    val_end = min(splits.val_end, max_dt)
    test_start = max(splits.test_start, min_dt)
    test_end = min(splits.test_end, max_dt)
    if test_start > test_end:
        test_start = index[int(len(index) * 0.7)]
        test_end = max_dt
    if train_end >= test_start:
        train_end = index[int(len(index) * 0.6)]
    if val_end <= train_end:
        val_end = index[int(len(index) * 0.8)]
    return SplitConfig(train_end=train_end, val_end=val_end, test_start=test_start, test_end=test_end)


def split_by_time(X: pd.DataFrame, y: pd.Series, splits: SplitConfig) -> dict:
    train_mask = X.index <= splits.train_end
    val_mask = (X.index > splits.train_end) & (X.index <= splits.val_end)
    test_mask = (X.index >= splits.test_start) & (X.index <= splits.test_end)
    return {
        "X_train": X.loc[train_mask],
        "y_train": y.loc[train_mask],
        "X_val": X.loc[val_mask],
        "y_val": y.loc[val_mask],
        "X_test": X.loc[test_mask],
        "y_test": y.loc[test_mask],
    }


def train_model(X: pd.DataFrame, y: pd.Series, model_name: str) -> tuple[object, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    if model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_scaled, y.values)
        return model, scaler
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_scaled, y.values)
    return model, scaler


def predict_proba(model: LogisticRegression, scaler: StandardScaler, X: pd.DataFrame) -> pd.Series:
    X_scaled = scaler.transform(X.values)
    proba = model.predict_proba(X_scaled)[:, 1]
    return pd.Series(proba, index=X.index)


def evaluate_model(y_true: pd.Series, y_prob: pd.Series) -> dict:
    if y_true.empty:
        return {}
    metrics = {
        "prevalence": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if y_true.nunique() > 1 else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if y_true.nunique() > 1 else np.nan,
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
    return metrics
