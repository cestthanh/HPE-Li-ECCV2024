import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import OriginalHPE3D
from utils.eval_3d import compute_3d_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase B smoke test: HPE-Li-3D shape check and one-batch overfit."
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Use the first MMFi train batch instead of a synthetic fixed batch.",
    )
    parser.add_argument(
        "--dataset-root",
        default=os.getenv("MMFI_DATASET_ROOT", str(PROJECT_ROOT / "data" / "mmfi" / "dataset")),
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "dataset_lib" / "config.yaml"),
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def make_synthetic_batch(batch_size, device):
    csi = torch.rand(batch_size, 3, 114, 10, device=device)

    xyz = torch.empty(batch_size, 17, 3, device=device)
    xyz[..., 0] = 0.08 + 0.50 * torch.rand(batch_size, 17, device=device)
    xyz[..., 1] = -0.77 + 1.50 * torch.rand(batch_size, 17, device=device)
    xyz[..., 2] = 2.60 + 0.65 * torch.rand(batch_size, 17, device=device)
    return csi, xyz


def make_real_batch(args, device):
    import yaml

    from dataset_lib import make_dataloader, make_dataset

    with open(args.config, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    config["train_loader"] = dict(config["train_loader"])
    config["train_loader"]["batch_size"] = args.batch_size
    config["train_loader"]["num_workers"] = 0

    train_dataset, _ = make_dataset(args.dataset_root, config)
    generator = torch.Generator().manual_seed(args.seed)
    loader = make_dataloader(
        train_dataset,
        is_training=True,
        generator=generator,
        **config["train_loader"],
    )
    batch = next(iter(loader))
    csi = batch["input_wifi-csi"].to(device).float()
    xyz = batch["output"][:, :, 0:3].to(device).float()
    return csi, xyz


def print_metrics(prefix, pred_pose, gt_pose):
    metrics = compute_3d_metrics(pred_pose, gt_pose)
    print(
        f"{prefix}: "
        f"mpjpe={metrics['mpjpe_mm']:.3f}mm, "
        f"pa_mpjpe={metrics['pa_mpjpe_mm']:.3f}mm, "
        f"pck50={metrics['pck_50mm']:.3f}, "
        f"pck100={metrics['pck_100mm']:.3f}, "
        f"pa_invalid={metrics['pa_mpjpe_invalid_count']}",
        flush=True,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    model = OriginalHPE3D().to(device)
    criterion = torch.nn.SmoothL1Loss(beta=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.real_data:
        csi, gt_pose = make_real_batch(args, device)
        source = "real MMFi batch"
    else:
        csi, gt_pose = make_synthetic_batch(args.batch_size, device)
        source = "synthetic fixed batch"

    model.train()
    pred_pose, _ = model(csi)
    loss = criterion(pred_pose, gt_pose)
    print(f"source={source}")
    print(f"device={device}")
    print(f"csi_shape={tuple(csi.shape)}")
    print(f"gt_pose_shape={tuple(gt_pose.shape)}")
    print(f"pred_pose_shape={tuple(pred_pose.shape)}")
    print(f"initial_loss={loss.item():.6f}")
    print_metrics("initial_metrics", pred_pose, gt_pose)

    for step in range(1, args.steps + 1):
        pred_pose, _ = model(csi)
        loss = criterion(pred_pose, gt_pose)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step == args.steps or step % max(1, args.steps // 5) == 0:
            print(f"step={step}, loss={loss.item():.6f}", flush=True)

    model.eval()
    with torch.no_grad():
        pred_pose, _ = model(csi)
        final_loss = criterion(pred_pose, gt_pose)

    print(f"final_loss={final_loss.item():.6f}")
    print_metrics("final_metrics", pred_pose, gt_pose)


if __name__ == "__main__":
    main()
