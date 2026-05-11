#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:16:35 2024

@author: jackson-devworks
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

experiment_config = {
    "mmfi_config": str(PROJECT_ROOT / "dataset_lib" / "config.yaml"),
    "dataset_root": os.getenv("MMFI_DATASET_ROOT", str(PROJECT_ROOT / "data" / "mmfi" / "dataset")),
    "noise_level": [0.0],
    "mode": 0,  # Mode 0: no denoiser layer, Mode 1: have AE denoiser layers, Mode 2: use traditional filter to denoise
    "epoch": int(os.getenv("HPE_LI_EPOCHS", "60")),
    "checkpoint": os.getenv("HPE_LI_CHECKPOINT", str(PROJECT_ROOT / "output")),
}

denoiser_config = {
    "epoch": int(os.getenv("HPE_LI_DENOISER_EPOCHS", "20")),
    "mode": 1,  # Mode 0: 1 stage AE, Mode 1: stacked AE
    "previous_encoder": os.getenv(
        "HPE_LI_PREVIOUS_ENCODER",
        str(PROJECT_ROOT / "output" / "SPN" / "FourLayerDenosing" / "Encoder-DecoderReconstructor"),
    ),
    "checkpoint": os.getenv(
        "HPE_LI_DENOISER_CHECKPOINT",
        str(PROJECT_ROOT / "output" / "AWGN" / "FiveLayerDenosing" / "Encoder-DecoderReconstructor"),
    ),
}
