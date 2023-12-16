import torch
import torch.nn as nn


class FixedFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=129)
        self.max_pool = nn.MaxPool1d(kernel_size=3)
        self.batch_norm = nn.BatchNorm1d(128)
        self.l_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.batch_norm(x)
        x = self.l_relu(x)
        return x
