import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split


from config import Config
from download_data import download_sector_data
from dataset import SectorGraphDataset
from mlp import MLP


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for data in loader:
        x = data.x.view(-1, data.x.size(-1))  # flatten nodes
        y = data.y.view(-1).float()

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def eval_epoch(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            x = data.x.view(-1, data.x.size(-1))
            y = data.y.view(-1)

            logits = model(x)
            preds = (logits > 0).long()

            correct += (preds == y).sum().item()
            total += y.numel()

    return correct / total


if __name__ == "__main__":
    cfg = Config()

    prices, volumes = download_sector_data(cfg)
    dataset = SectorGraphDataset(prices, volumes, cfg)

    train_size = int(len(dataset) * cfg.train_fraction)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    model = MLP(input_dim=5, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(cfg.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_acc = eval_epoch(model, val_loader)

        print(f"Epoch {epoch+1:02d} | Train loss: {train_loss:.4f} | Val acc: {val_acc:.3f}")
