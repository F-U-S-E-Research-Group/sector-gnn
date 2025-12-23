import torch
import numpy as np


def always_up_baseline(dataset):
    """
    Predict 1 for every node in every graph.
    Returns overall accuracy.
    """
    total = 0
    correct = 0

    for data in dataset:
        y_true = data.y
        y_pred = torch.ones_like(y_true)

        correct += (y_true == y_pred).sum().item()
        total += y_true.numel()

    accuracy = correct / total
    return accuracy
