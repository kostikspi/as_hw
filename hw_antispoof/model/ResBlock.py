import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F


class FMS(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, channels)

    def forward(self, x):
        res = x
        x = self.avgpool(x).view(x.size(0), -1)
        x = self.fc(x)
        x = F.sigmoid(x).view(x.size(0), x.size(1), -1)
        x = res * x + x
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.l_relu = nn.LeakyReLU()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.bn(x)
        x = self.l_relu(x)
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sample=False):
        super().__init__()
        self.sample = sample
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.l_relu = nn.LeakyReLU()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        # self.conv1 = ConvBlock(in_channels, out_channels)
        # self.conv2 = ConvBlock(out_channels, out_channels)
        if self.sample:
            self.sample_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.max_pool = nn.MaxPool1d(3)
        self.FMS = FMS(out_channels)

    def forward(self, x):
        res = x
        x = self.bn1(x)
        x = self.l_relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.l_relu(x)
        x = self.conv2(x)
        if self.sample:
            res = self.sample_conv(res)
        x = x + res
        x = self.max_pool(x)
        x = self.FMS(x)
        return x
