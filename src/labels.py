import pandas as pd
import numpy as np


def compute_forward_returns(prices: pd.DataFrame, forward_days: int):
    """
    Compute forward cumulative returns:
    (P[t+forward_days] / P[t]) - 1
    """
    forward_prices = prices.shift(-forward_days)
    forward_returns = (forward_prices / prices) - 1.0
    return forward_returns


def make_direction_labels(prices: pd.DataFrame, forward_days: int):
    """
    Convert forward returns into binary direction labels.
    1 = positive return
    0 = zero or negative return
    """
    fwd_returns = compute_forward_returns(prices, forward_days)
    labels = (fwd_returns > 0).astype(int)
    return labels
