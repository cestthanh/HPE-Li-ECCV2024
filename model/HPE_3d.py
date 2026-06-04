import re
import time

import torch
from torch import nn

from .utils import SKUnit, regression


DEFAULT_HPE3D_MODEL_CONFIG = {
    "sk_m": 4,
    "sk_g": 32,
    "sk_r": 32,
    "sk_l": 32,
    "sk_use_min_l": True,
    "sk_layout": "stack",
}


def normalize_hpe3d_model_config(model_config=None):
    config = dict(DEFAULT_HPE3D_MODEL_CONFIG)
    if model_config:
        config.update(model_config)

    for key in ("sk_m", "sk_g", "sk_r", "sk_l"):
        config[key] = int(config[key])
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive, got {config[key]}.")

    config["sk_use_min_l"] = bool(config["sk_use_min_l"])
    if config["sk_layout"] not in {"legacy_view", "stack"}:
        raise ValueError(f"Unsupported sk_layout: {config['sk_layout']}")
    if 64 % config["sk_g"] != 0 or 128 % config["sk_g"] != 0:
        raise ValueError(
            f"sk_g={config['sk_g']} must divide both 64 and 128 channels."
        )
    return config


def strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def infer_hpe3d_model_config(state_dict):
    """Infer legacy HPE-Li-3D architecture dimensions from state tensor shapes."""
    state_dict = strip_module_prefix(state_dict)
    branch_pattern = re.compile(r"^skunit1\.conv2_sk\.0\.fcs\.(\d+)\.weight$")
    branch_indices = [
        int(match.group(1))
        for key in state_dict
        if (match := branch_pattern.match(key)) is not None
    ]
    if not branch_indices:
        raise ValueError("Cannot infer HPE-Li-3D branch count from checkpoint.")

    conv1 = state_dict["skunit1.conv2_sk.0.convs.0.0.weight"]
    conv2 = state_dict["skunit2.conv2_sk.0.convs.0.0.weight"]
    sk_g_1 = 64 // int(conv1.shape[1])
    sk_g_2 = 128 // int(conv2.shape[1])
    if sk_g_1 != sk_g_2:
        raise ValueError(
            f"Checkpoint uses inconsistent SK groups: {sk_g_1} and {sk_g_2}."
        )

    d1 = int(state_dict["skunit1.conv2_sk.0.fc.0.weight"].shape[0])
    d2 = int(state_dict["skunit2.conv2_sk.0.fc.0.weight"].shape[0])
    ratio_1 = 64 // d1 if 64 % d1 == 0 else None
    ratio_2 = 128 // d2 if 128 % d2 == 0 else None
    if ratio_1 is not None and ratio_1 == ratio_2:
        sk_r = ratio_1
        sk_l = 32
        sk_use_min_l = False
    else:
        # New checkpoints save model_config. This fallback only needs to
        # reconstruct a shape-compatible model if metadata is unavailable.
        sk_r = 32
        sk_l = min(d1, d2)
        sk_use_min_l = True

    return normalize_hpe3d_model_config(
        {
            "sk_m": max(branch_indices) + 1,
            "sk_g": sk_g_1,
            "sk_r": sk_r,
            "sk_l": sk_l,
            "sk_use_min_l": sk_use_min_l,
            # Every checkpoint created before model_config metadata used the
            # historical view-based forward pass.
            "sk_layout": "legacy_view",
        }
    )


def get_hpe3d_model_config(checkpoint, state_dict=None):
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model_config"), dict):
        return normalize_hpe3d_model_config(checkpoint["model_config"])

    if state_dict is None and isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                state_dict = value
                break
        if state_dict is None and checkpoint and all(
            hasattr(value, "shape") for value in checkpoint.values()
        ):
            state_dict = checkpoint

    if state_dict is None:
        raise ValueError("Cannot determine HPE-Li-3D model configuration.")
    return infer_hpe3d_model_config(state_dict)


class OriginalHPE3D(nn.Module):
    def __init__(self, **model_config):
        super(OriginalHPE3D, self).__init__()
        self.model_config = normalize_hpe3d_model_config(model_config)
        num_lay = 64
        hidden_reg = 32

        sk_kwargs = {
            "M": self.model_config["sk_m"],
            "G": self.model_config["sk_g"],
            "r": self.model_config["sk_r"],
            "L": self.model_config["sk_l"],
            "use_min_l": self.model_config["sk_use_min_l"],
            "layout": self.model_config["sk_layout"],
        }
        self.skunit1 = SKUnit(
            in_features=3,
            mid_features=num_lay,
            out_features=num_lay,
            dim1=114,
            dim2=10,
            pool_dim="freq-chan",
            stride=1,
            **sk_kwargs,
        )
        self.skunit2 = SKUnit(
            in_features=num_lay,
            mid_features=num_lay * 2,
            out_features=num_lay * 2,
            dim1=57,
            dim2=8,
            pool_dim="freq-chan",
            stride=1,
            **sk_kwargs,
        )
        self.regression = regression(
            input_dim=7168, output_dim=51, hidden_dim=hidden_reg
        )

    def get_model_config(self):
        return dict(self.model_config)

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
