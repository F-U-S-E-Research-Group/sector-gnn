import pandas as pd
import numpy as np


def zscore_by_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score each column independently over time.
    """
    mean = df.mean()
    std = df.std().replace(0, np.nan)
    return (df - mean) / std


def build_features(prices: pd.DataFrame, volumes: pd.DataFrame):
    """
    Compute feature matrices aligned by date.
    Returns a dict of DataFrames.
    """
    prices = prices.sort_index()
    volumes = volumes.sort_index()

    # Daily returns
    daily_returns = prices.pct_change()

    # Feature engineering
    ret_5d = prices.pct_change(5)
    ret_10d = prices.pct_change(10)

    vol_5d = daily_returns.rolling(5).std()
    avg_vol_5d = volumes.rolling(5).mean()

    ma_10 = prices.rolling(10).mean()
    dist_ma_10 = prices - ma_10

    features = {
        "ret_5d": ret_5d,
        "ret_10d": ret_10d,
        "vol_5d": vol_5d,
        "avg_vol_5d": avg_vol_5d,
        "dist_ma_10": dist_ma_10,
        "daily_returns": daily_returns,
        "prices": prices,
    }

    # Align dates (drop NaNs from rolling windows)
    valid_index = features["ret_10d"].dropna().index
    for k in features:
        features[k] = features[k].loc[valid_index]

    # Z-score the actual input features
    for k in ["ret_5d", "ret_10d", "vol_5d", "avg_vol_5d", "dist_ma_10"]:
        features[k] = zscore_by_column(features[k])

    return features
