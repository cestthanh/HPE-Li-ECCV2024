import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_lib import make_dataloader, make_dataset
from model import OriginalHPE3D
from utils.eval_3d import MMFI_17_JOINT_NAMES, compute_3d_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an HPE-Li-3D checkpoint and export per-joint MPJPE."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset-root",
        default=os.getenv(
            "MMFI_DATASET_ROOT", str(PROJECT_ROOT / "data" / "mmfi" / "dataset")
        ),
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "dataset_lib" / "config.yaml"),
        help="Fallback config path if checkpoint does not include one.",
    )
    parser.add_argument(
        "--split-to-use",
        default=None,
        choices=["random_split", "cross_scene_split", "cross_subject_split", "manual_split"],
        help="Override config split_to_use. Usually not needed for checkpoints from train_3d_baseline.py.",
    )
    parser.add_argument(
        "--eval-split",
        default="test",
        choices=["val", "test", "eval_all"],
        help="Which half of the eval dataset to evaluate. train_3d_baseline.py reports test.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--output-json",
        default=None,
        help="Path to save full metrics JSON. Defaults to checkpoint directory/per_joint_metrics.json.",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Path to save Markdown per-joint table. Defaults to checkpoint directory/per_joint_mpjpe.md.",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_config(args, checkpoint):
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        config = copy.deepcopy(checkpoint["config"])
    else:
        with open(args.config, "r") as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)

    if args.split_to_use is not None:
        config["split_to_use"] = args.split_to_use

    for key in ("train_loader", "val_loader", "test_loader"):
        config[key] = dict(config[key])
        config[key]["num_workers"] = args.num_workers

    config["val_loader"]["batch_size"] = args.batch_size
    config["test_loader"]["batch_size"] = args.batch_size
    return config


def make_eval_loader(dataset_root, config, args):
    _, eval_dataset = make_dataset(dataset_root, config)

    if args.eval_split == "eval_all":
        selected_dataset = eval_dataset
    else:
        val_indices, test_indices = train_test_split(
            list(range(len(eval_dataset))), test_size=0.5, random_state=41
        )
        selected_indices = val_indices if args.eval_split == "val" else test_indices
        selected_dataset = Subset(eval_dataset, selected_indices)

    generator = torch.Generator().manual_seed(args.seed)
    loader = make_dataloader(
        selected_dataset,
        is_training=False,
        generator=generator,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=config["test_loader"].get("pin_memory", False),
    )
    return loader, selected_dataset


def load_model(checkpoint, device):
    model = OriginalHPE3D().to(device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError(
            "Unsupported checkpoint format. Expected a dict with model_state_dict."
        )
    model.eval()
    return model


def evaluate(model, loader, device, max_batches=None):
    pred_chunks = []
    gt_chunks = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="evaluate")):
            if max_batches is not None and batch_idx >= max_batches:
                break
            csi_data = batch["input_wifi-csi"].to(device).float()
            gt_pose = batch["output"][:, :, 0:3].to(device).float()
            pred_pose, _ = model(csi_data)
            pred_chunks.append(pred_pose.detach().cpu().numpy())
            gt_chunks.append(gt_pose.detach().cpu().numpy())

    if not pred_chunks:
        raise RuntimeError("No batches were evaluated.")

    pred_all = np.concatenate(pred_chunks, axis=0)
    gt_all = np.concatenate(gt_chunks, axis=0)
    metrics = compute_3d_metrics(pred_all, gt_all)
    metrics["num_samples"] = int(pred_all.shape[0])
    metrics["num_batches"] = int(len(pred_chunks))
    return metrics


def make_markdown_table(metrics):
    lines = [
        "| Joint | MPJPE (mm) |",
        "|---|---:|",
    ]
    for name, value in zip(MMFI_17_JOINT_NAMES, metrics["per_joint_mpjpe_mm"]):
        lines.append(f"| {name} | {value:.3f} |")
    lines.append(f"| Average | {metrics['mpjpe_mm']:.3f} |")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = load_config(args, checkpoint)
    loader, selected_dataset = make_eval_loader(args.dataset_root, config, args)
    model = load_model(checkpoint, device)
    metrics = evaluate(model, loader, device, max_batches=args.max_batches)

    metrics["checkpoint"] = str(checkpoint_path)
    metrics["dataset_root"] = args.dataset_root
    metrics["split_to_use"] = config["split_to_use"]
    metrics["eval_split"] = args.eval_split

    output_json = (
        Path(args.output_json)
        if args.output_json is not None
        else checkpoint_path.parent / f"per_joint_metrics_{args.eval_split}.json"
    )
    output_md = (
        Path(args.output_md)
        if args.output_md is not None
        else checkpoint_path.parent / f"per_joint_mpjpe_{args.eval_split}.md"
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as fd:
        json.dump(metrics, fd, indent=2)
    with open(output_md, "w") as fd:
        fd.write(make_markdown_table(metrics))

    print(
        "eval_split=%s samples=%d mpjpe=%.3f pa_mpjpe=%.3f "
        "pck50=%.3f pck100=%.3f"
        % (
            args.eval_split,
            len(selected_dataset),
            metrics["mpjpe_mm"],
            metrics["pa_mpjpe_mm"],
            metrics["pck_50mm"],
            metrics["pck_100mm"],
        ),
        flush=True,
    )
    print(f"saved_json={output_json}", flush=True)
    print(f"saved_md={output_md}", flush=True)
    print(make_markdown_table(metrics), flush=True)


if __name__ == "__main__":
    main()
