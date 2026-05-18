import time

import torch
from torch import nn

from .utils import SKUnit, regression


class OriginalHPE3D(nn.Module):
    def __init__(self):
        super(OriginalHPE3D, self).__init__()
        num_lay = 64
        hidden_reg = 32

        self.skunit1 = SKUnit(
            in_features=3,
            mid_features=num_lay,
            out_features=num_lay,
            dim1=114,
            dim2=10,
            pool_dim="freq-chan",
            M=2,
            G=64,
            r=4,
            stride=1,
            L=32,
        )
        self.skunit2 = SKUnit(
            in_features=num_lay,
            mid_features=num_lay * 2,
            out_features=num_lay * 2,
            dim1=57,
            dim2=8,
            pool_dim="freq-chan",
            M=2,
            G=64,
            r=4,
            stride=1,
            L=32,
        )
        self.regression = regression(
            input_dim=7168, output_dim=51, hidden_dim=hidden_reg
        )

    def forward(self, x):
        batch = x.shape[0]
        time_start = time.time()

        pool = torch.nn.AvgPool2d((2, 2))
        x = self.skunit1(x)
        x = pool(x)

        out = self.skunit2(x)
        out = pool(out)

        pose = self.regression(out)
        pose = pose.reshape(batch, 17, 3)

        time_sum = time.time() - time_start
        return pose, time_sum
