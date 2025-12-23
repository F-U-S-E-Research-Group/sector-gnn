import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [batch_size * num_nodes, num_features]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)
