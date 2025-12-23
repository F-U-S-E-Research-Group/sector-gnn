from config import Config
from download_data import download_sector_data
from labels import make_direction_labels


if __name__ == "__main__":
    cfg = Config()

    prices, _ = download_sector_data(cfg)
    labels = make_direction_labels(prices, cfg.forward_days)

    print("Label shape:", labels.shape)
    print("Sample labels:")
    print(labels.head())

    # Class balance check
    positive_rate = labels.mean().mean()
    print(f"\nOverall positive rate: {positive_rate:.3f}")
