from typing import Union

from torch import nn, Tensor
import torch.nn.functional as F
import torch

from hw_asr.base import BaseModel

from hw_asr.model.SincConv import SincConv_fast
from hw_asr.model.ResBlock import ResBlock


class RawNet(BaseModel):
    def __init__(self, do_abs, sinc_hz=0, sinc_band=0, gru_layers=3, do_bn_gru=True, grad_sinc=False):
        super().__init__()
        self.do_abs = do_abs
        # HINT #5
        self.sinc_filter = SincConv_fast(kernel_size=129, out_channels=20, min_low_hz=sinc_hz,
                                         min_band_hz=sinc_band, grad_sinc=grad_sinc)
        self.res_blocks1 = nn.Sequential(*[ResBlock(20, 20, sample=True),
                                           ResBlock(20, 20)])
        self.res_blocks2 = nn.Sequential(*[ResBlock(20, 128, sample=True) if i == 0
                                           else ResBlock(128, 128)
                                           for i in range(4)])
        self.gru = nn.GRU(input_size=128, hidden_size=1024, num_layers=gru_layers)
        self.fc1 = nn.Linear(in_features=1024, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=2)
        self.bn1 = nn.BatchNorm1d(20)
        self.bn2 = nn.BatchNorm1d(128)
        self.maxpool = nn.MaxPool1d(3)
        self.do_bn = do_bn_gru

    def forward(self, wave, **batch) -> Union[Tensor, dict]:

        x = self.sinc_filter(wave)

        # HINT #1
        if self.do_abs:
            x = torch.abs(x)  # 3, 128, 63872

        x = self.maxpool(x)

        x = self.bn1(x)
        x = F.relu(x)

        x = self.res_blocks1(x)
        x = self.res_blocks2(x)
        # HINT #2
        if self.do_bn:
            x = self.bn2(x)
            x = F.leaky_relu(x).mT
        else:
            x = x.mT

        x, _ = self.gru(x)
        x = self.fc1(x[:, -1, :])
        x = self.fc2(x)
        return x

    def transform_input_lengths(self, input_lengths):
        pass
