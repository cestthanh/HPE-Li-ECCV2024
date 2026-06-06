import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import tools.demo_inference_3d as demo
from model.dsknet_trans_mmfi_3d import (
    DSKNetTransMMFI3D,
    get_dsknet_trans_mmfi_3d_model_config,
)


def load_phase_c_model(checkpoint_path, device):
    checkpoint = demo.load_checkpoint(checkpoint_path, device)
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint.to(device)
        model.eval()
        return model, checkpoint

    state_dict = demo.strip_module_prefix(demo.extract_state_dict(checkpoint))
    model_config = get_dsknet_trans_mmfi_3d_model_config(checkpoint)
    model = DSKNetTransMMFI3D(**model_config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def main():
    demo.load_model = load_phase_c_model
    demo.main()


if __name__ == "__main__":
    main()
