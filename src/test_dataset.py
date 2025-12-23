from config import Config
from download_data import download_sector_data
from dataset import SectorGraphDataset


if __name__ == "__main__":
    cfg = Config()

    prices, volumes = download_sector_data(cfg)
    dataset = SectorGraphDataset(prices, volumes, cfg)

    print("Dataset length:", len(dataset))

    sample = dataset[0]
    print("\nSample graph:")
    print("x shape:", sample.x.shape)
    print("y shape:", sample.y.shape)
    print("edge_index shape:", sample.edge_index.shape)
    print("edge_weight shape:", sample.edge_weight.shape)
    print("date:", sample.date)
