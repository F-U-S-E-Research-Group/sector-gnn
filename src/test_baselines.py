from config import Config
from download_data import download_sector_data
from dataset import SectorGraphDataset
from baselines import always_up_baseline


if __name__ == "__main__":
    cfg = Config()

    prices, volumes = download_sector_data(cfg)
    dataset = SectorGraphDataset(prices, volumes, cfg)

    acc = always_up_baseline(dataset)
    print(f"Always-Up baseline accuracy: {acc:.3f}")
