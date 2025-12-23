import numpy as np
import torch
from torch_geometric.data import Data

from config import Config, SECTORS
from features import build_features
from labels import make_direction_labels


class SectorGraphDataset:
    def __init__(self, prices, volumes, cfg: Config):
        self.cfg = cfg

        # Build features + labels
        self.features = build_features(prices, volumes)
        self.labels = make_direction_labels(prices, cfg.forward_days)

        # Align dates across features and labels
        self.dates = self.features["prices"].index.intersection(
            self.labels.index
        )

        # We also need enough past data for correlations
        self.start_idx = cfg.corr_window
        self.dates = self.dates[self.start_idx : -cfg.forward_days]

        self.num_sectors = len(SECTORS)

    def __len__(self):
        return len(self.dates)

    def _build_graph(self, returns_window):
        corr = returns_window.corr().values
        edge_src = []
        edge_dst = []
        edge_weight = []

        for i in range(self.num_sectors):
            for j in range(self.num_sectors):
                if i == j:
                    continue
                w = corr[i, j]
                if np.isfinite(w) and abs(w) > self.cfg.corr_threshold:
                    edge_src.append(i)
                    edge_dst.append(j)
                    edge_weight.append(w)

        # Fallback: fully connected graph with zero weights
        if len(edge_src) == 0:
            for i in range(self.num_sectors):
                for j in range(self.num_sectors):
                    if i != j:
                        edge_src.append(i)
                        edge_dst.append(j)
                        edge_weight.append(0.0)

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)

        return edge_index, edge_weight

    def __getitem__(self, idx):
        date = self.dates[idx]

        # Node features
        x = np.stack(
            [
                self.features["ret_5d"].loc[date].values,
                self.features["ret_10d"].loc[date].values,
                self.features["vol_5d"].loc[date].values,
                self.features["avg_vol_5d"].loc[date].values,
                self.features["dist_ma_10"].loc[date].values,
            ],
            axis=1,
        )

        x = torch.tensor(x, dtype=torch.float)

        # Labels
        y = self.labels.loc[date].values
        y = torch.tensor(y, dtype=torch.long)

        # Rolling window for correlations
        date_idx = self.features["prices"].index.get_loc(date)
        returns_window = self.features["daily_returns"].iloc[
            date_idx - self.cfg.corr_window : date_idx
        ]

        edge_index, edge_weight = self._build_graph(returns_window)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
        )

        data.date = str(date)
        return data
