import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from config import Config
from download_data import download_sector_data
from dataset import SectorGraphDataset
from gnn import SectorGCN


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index, batch.edge_weight)
        y = batch.y.view(-1).float()

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
        for batch in loader:
            logits = model(batch.x, batch.edge_index, batch.edge_weight)
            preds = (logits > 0).long()
            y = batch.y.view(-1)

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
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = SectorGCN(input_dim=5, hidden_dim=cfg.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    best_val = 0.0

    for epoch in range(cfg.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_acc = eval_epoch(model, val_loader)
        best_val = max(best_val, val_acc)

        print(f"Epoch {epoch+1:02d} | Train loss: {train_loss:.4f} | Val acc: {val_acc:.3f}")

    print(f"\nBest validation accuracy: {best_val:.3f}")
