import time

import torch
import torch.nn.functional as F
from torch import nn

from .utils import ChannelTransformer, regression


DEFAULT_DSKNET_TRANS_MMFI_3D_CONFIG = {
    "num_lay": 128,
    "hidden_reg": 32,
    "sk_m": 3,
    "sk_g": 32,
    "sk_r": 4,
    "sk_l": 32,
    "transformer_layers": 1,
    "transformer_heads": 3,
}


def normalize_dsknet_trans_mmfi_3d_config(model_config=None):
    config = dict(DEFAULT_DSKNET_TRANS_MMFI_3D_CONFIG)
    if model_config:
        config.update(model_config)

    int_keys = (
        "num_lay",
        "hidden_reg",
        "sk_m",
        "sk_g",
        "sk_r",
        "sk_l",
        "transformer_layers",
        "transformer_heads",
    )
    for key in int_keys:
        config[key] = int(config[key])
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive, got {config[key]}.")

    num_lay = config["num_lay"]
    if num_lay % config["sk_g"] != 0 or (num_lay * 2) % config["sk_g"] != 0:
        raise ValueError(
            f"sk_g={config['sk_g']} must divide num_lay={num_lay} "
            f"and 2*num_lay={num_lay * 2}."
        )
    return config


def get_dsknet_trans_mmfi_3d_model_config(checkpoint):
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model_config"), dict):
        return normalize_dsknet_trans_mmfi_3d_config(checkpoint["model_config"])
    return normalize_dsknet_trans_mmfi_3d_config()


class PhaseCDSKConv(nn.Module):
    """Author-aligned dual selective kernel convolution used by Phase C.

    This is a standalone 3D-port implementation of the DSKConv logic from
    model/sknet_trans_mmfi.py. It keeps channel selection, frequency selection,
    ChannelTransformer, and final width pooling.
    """

    def __init__(
        self,
        features,
        img_size,
        m=3,
        groups=32,
        reduction=4,
        min_bottleneck=32,
        stride=1,
        transformer_layers=1,
        transformer_heads=3,
    ):
        super().__init__()
        if features % groups != 0:
            raise ValueError(
                f"groups={groups} must divide features={features} in PhaseCDSKConv."
            )
        if len(img_size) != 2:
            raise ValueError(f"img_size must be [height, width], got {img_size}.")

        bottleneck = max(int(features / reduction), min_bottleneck)
        self.features = int(features)
        self.img_size = [int(img_size[0]), int(img_size[1])]
        self.m = int(m)

        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        features,
                        features,
                        kernel_size=3,
                        stride=stride,
                        padding=1 + branch_idx,
                        dilation=1 + branch_idx,
                        groups=groups,
                        bias=False,
                    ),
                    nn.BatchNorm2d(features),
                    nn.ReLU(inplace=True),
                )
                for branch_idx in range(m)
            ]
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(features, bottleneck, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True),
        )
        self.fcs = nn.ModuleList(
            [nn.Conv2d(bottleneck, features, kernel_size=1) for _ in range(m)]
        )
        self.softmax = nn.Softmax(dim=1)
        self.norm = nn.BatchNorm2d(features)
        self.transformer = ChannelTransformer(
            vis=False,
            img_size=self.img_size,
            channel_num=features,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
        )

    def forward(self, x):
        feats = torch.stack([conv(x) for conv in self.convs], dim=1)

        feats_u = feats.sum(dim=1)
        feats_s = self.gap(feats_u)
        feats_z = self.fc(feats_s)
        channel_attention = torch.stack([fc(feats_z) for fc in self.fcs], dim=1)
        channel_attention = self.softmax(channel_attention)
        feats_channel = (feats * channel_attention).sum(dim=1)

        feats_frequency = feats.sum(dim=2)
        frequency_attention = F.adaptive_avg_pool2d(
            feats_frequency, (feats_frequency.size(2), 1)
        )
        frequency_attention = self.softmax(frequency_attention)
        feats_frequency = (feats * frequency_attention.unsqueeze(2)).sum(dim=1)

        feats_v = torch.cat([feats_channel, feats_frequency], dim=3)
        if list(feats_v.shape[2:4]) != self.img_size:
            raise RuntimeError(
                f"PhaseCDSKConv expected transformer input spatial size "
                f"{self.img_size}, got {list(feats_v.shape[2:4])}."
            )

        feats_v = self.norm(feats_v)
        feats_v, _ = self.transformer(feats_v)
        return F.avg_pool2d(feats_v, kernel_size=(1, 2))


class PhaseCDSKUnit(nn.Module):
    """Basic 2D CNN + pooling + PhaseCDSKConv block."""

    def __init__(
        self,
        in_features,
        mid_features,
        out_features,
        img_size,
        m=3,
        groups=32,
        reduction=4,
        min_bottleneck=32,
        stride=1,
        transformer_layers=1,
        transformer_heads=3,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True),
        )
        self.pooling = nn.AvgPool2d((2, 2))
        self.conv2_dsk = PhaseCDSKConv(
            mid_features,
            img_size=img_size,
            m=m,
            groups=groups,
            reduction=reduction,
            min_bottleneck=min_bottleneck,
            stride=stride,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
        )
        self.norm = nn.BatchNorm2d(mid_features)
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_features, out_features, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_features),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.pooling(out)
        out = self.conv2_dsk(out)
        out = self.norm(out)
        out = self.conv3(out)
        return out


class DSKNetTransMMFI3D(nn.Module):
    """3D port of the author's DSKNetTransMMFI model.

    The CSI backbone follows model/sknet_trans_mmfi.py. The only intended
    architectural change is the regression head output: 17*2 -> 17*3.
    """

    def __init__(self, **model_config):
        super().__init__()
        self.model_config = normalize_dsknet_trans_mmfi_3d_config(model_config)
        num_lay = self.model_config["num_lay"]
        hidden_reg = self.model_config["hidden_reg"]

        common_kwargs = {
            "m": self.model_config["sk_m"],
            "groups": self.model_config["sk_g"],
            "reduction": self.model_config["sk_r"],
            "min_bottleneck": self.model_config["sk_l"],
            "stride": 1,
            "transformer_layers": self.model_config["transformer_layers"],
            "transformer_heads": self.model_config["transformer_heads"],
        }

        self.skunit1 = PhaseCDSKUnit(
            in_features=3,
            mid_features=num_lay,
            out_features=num_lay,
            img_size=[57, 10],
            **common_kwargs,
        )
        self.norm = nn.BatchNorm2d(num_lay)
        self.skunit2 = PhaseCDSKUnit(
            in_features=num_lay,
            mid_features=num_lay * 2,
            out_features=num_lay * 2,
            img_size=[28, 4],
            **common_kwargs,
        )
        self.final_pool = nn.AvgPool2d((2, 2))
        self.regression = regression(input_dim=3584, output_dim=51, hidden_dim=hidden_reg)

    def get_model_config(self):
        return dict(self.model_config)

    def forward_features(self, x):
        out = self.skunit1(x)
        out = self.norm(out)
        out = self.skunit2(out)
        return self.final_pool(out)

    def forward(self, x):
        batch = x.shape[0]
        time_start = time.time()

        features = self.forward_features(x)
        pose = self.regression(features)
        pose = pose.reshape(batch, 17, 3)

        time_sum = time.time() - time_start
        return pose, time_sum
