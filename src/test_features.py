import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

print("TEST_FEATURES SCRIPT STARTED")

import pandas as pd
from config import Config
from download_data import download_sector_data
from features import build_features





if __name__ == "__main__":
    cfg = Config()

    prices, volumes = download_sector_data(cfg)
    feats = build_features(prices, volumes)

    print("Feature shapes:")
    for k, v in feats.items():
        print(f"{k}: {v.shape}")

    print("\nSample feature values:")
    print(feats["ret_5d"].head())
