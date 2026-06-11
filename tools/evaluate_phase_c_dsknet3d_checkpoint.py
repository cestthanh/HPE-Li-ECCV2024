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
from dataset_lib.splits import split_eval_dataset_by_sequence
from model.dsknet_trans_mmfi_3d import (
    DSKNetTransMMFI3D,
    get_dsknet_trans_mmfi_3d_model_config,
)
from utils.eval_3d import MMFI_17_JOINT_NAMES, compute_3d_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a Phase C DSKNetTransMMFI3D checkpoint."
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
        choices=[
            "random_split",
            "cross_scene_split",
            "cross_subject_split",
            "manual_split",
        ],
        help="Override config split_to_use. Usually not needed for Phase C checkpoints.",
    )
    parser.add_argument(
        "--eval-split",
        default="test",
        choices=["val", "test", "eval_all"],
        help="Which eval partition to evaluate.",
    )
    parser.add_argument(
        "--eval-partition-unit",
        default="auto",
        choices=["auto", "sequence", "frame"],
        help=(
            "Use sequence-level splitting for new checkpoints or legacy frame-level "
            "splitting. Auto reads checkpoint metadata and defaults to frame for old "
            "checkpoints."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-graphpose-md", default=None)
    parser.add_argument("--method-name", default="DSKNetTransMMFI3D")
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


def get_eval_partition_unit(checkpoint, requested_unit):
    if requested_unit != "auto":
        return requested_unit
    if isinstance(checkpoint, dict):
        metadata = checkpoint.get("eval_split_metadata")
        if not metadata:
            metrics = checkpoint.get("metrics")
            if isinstance(metrics, dict):
                metadata = metrics.get("eval_split_metadata")
        if isinstance(metadata, dict) and metadata.get("split_unit") == "sequence":
            return "sequence"
    return "frame"


def make_eval_loader(dataset_root, config, args, partition_unit):
    _, eval_dataset = make_dataset(dataset_root, config)

    if args.eval_split == "eval_all":
        selected_dataset = eval_dataset
        eval_split_metadata = {
            "split_unit": "none",
            "selection": "eval_all",
            "num_frames": len(eval_dataset),
        }
    elif partition_unit == "sequence":
        val_dataset, test_dataset, eval_split_metadata = (
            split_eval_dataset_by_sequence(
                eval_dataset, test_size=0.5, random_state=41
            )
        )
        selected_dataset = val_dataset if args.eval_split == "val" else test_dataset
        eval_split_metadata = dict(eval_split_metadata)
        eval_split_metadata["selection"] = args.eval_split
    else:
        val_indices, test_indices = train_test_split(
            list(range(len(eval_dataset))), test_size=0.5, random_state=41
        )
        selected_indices = val_indices if args.eval_split == "val" else test_indices
        selected_dataset = Subset(eval_dataset, selected_indices)
        eval_split_metadata = {
            "split_unit": "frame",
            "selection": args.eval_split,
            "test_size": 0.5,
            "random_state": 41,
            "num_frames": len(selected_indices),
        }

    generator = torch.Generator().manual_seed(args.seed)
    loader = make_dataloader(
        selected_dataset,
        is_training=False,
        generator=generator,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=config["test_loader"].get("pin_memory", False),
    )
    return loader, selected_dataset, eval_split_metadata


def strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict")
        if isinstance(state_dict, dict):
            return strip_module_prefix(state_dict)
    raise ValueError("Unsupported checkpoint format. Expected model_state_dict.")


def load_model(checkpoint, device):
    state_dict = extract_state_dict(checkpoint)
    model_config = get_dsknet_trans_mmfi_3d_model_config(checkpoint)
    model = DSKNetTransMMFI3D(**model_config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_pose_normalization(checkpoint):
    if isinstance(checkpoint, dict):
        pose_stats = checkpoint.get("pose_normalization")
        if pose_stats:
            return pose_stats
        metrics = checkpoint.get("metrics")
        if isinstance(metrics, dict) and metrics.get("pose_normalization"):
            return metrics["pose_normalization"]
    return {"enabled": False}


def make_pose_stats_tensors(pose_stats, device):
    if not pose_stats or not pose_stats.get("enabled", False):
        return None
    return {
        "mean": torch.tensor(pose_stats["mean_xyz"], device=device).view(1, 1, 3),
        "std": torch.tensor(pose_stats["std_xyz"], device=device).view(1, 1, 3),
    }


def denormalize_pose(pose, pose_stats_tensors):
    if pose_stats_tensors is None:
        return pose
    return pose * pose_stats_tensors["std"] + pose_stats_tensors["mean"]


def evaluate(model, loader, device, pose_stats_tensors=None, max_batches=None):
    pred_chunks = []
    gt_chunks = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="evaluate")):
            if max_batches is not None and batch_idx >= max_batches:
                break
            csi_data = batch["input_wifi-csi"].to(device).float()
            gt_pose = batch["output"][:, :, 0:3].to(device).float()
            pred_pose, _ = model(csi_data)
            pred_pose = denormalize_pose(pred_pose, pose_stats_tensors)
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


def make_graphpose_markdown_table(metrics, method_name):
    lines = [
        "| Method | g_PCK@10 | g_PCK@20 | g_PCK@30 | g_PCK@40 | g_PCK@50 | MPJPE | PA-MPJPE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {method_name} "
            f"| {metrics['g_PCK@10']:.1f} "
            f"| {metrics['g_PCK@20']:.1f} "
            f"| {metrics['g_PCK@30']:.1f} "
            f"| {metrics['g_PCK@40']:.1f} "
            f"| {metrics['g_PCK@50']:.1f} "
            f"| {metrics['mpjpe_mm']:.1f} "
            f"| {metrics['pa_mpjpe_mm']:.1f} |"
        ),
        "",
        "`g_PCK@10`..`g_PCK@50` use thresholds 0.1..0.5 of the MMFi body scale, not millimeters.",
        "The corrected MMFi body scale is the ground-truth distance between R.Hip (index 1) and L.Shoulder (index 11).",
        "These values are not directly comparable to legacy GraphPose-Fi results computed with indices (5, 12).",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = load_config(args, checkpoint)
    partition_unit = get_eval_partition_unit(
        checkpoint, args.eval_partition_unit
    )
    loader, selected_dataset, eval_split_metadata = make_eval_loader(
        args.dataset_root, config, args, partition_unit
    )
    model = load_model(checkpoint, device)
    pose_stats = get_pose_normalization(checkpoint)
    pose_stats_tensors = make_pose_stats_tensors(pose_stats, device)

    metrics = evaluate(
        model,
        loader,
        device,
        pose_stats_tensors=pose_stats_tensors,
        max_batches=args.max_batches,
    )
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["dataset_root"] = args.dataset_root
    metrics["split_to_use"] = config["split_to_use"]
    metrics["eval_split"] = args.eval_split
    metrics["model_name"] = "DSKNetTransMMFI3D"
    metrics["model_config"] = model.get_model_config()
    metrics["pose_normalization"] = pose_stats
    metrics["eval_split_metadata"] = eval_split_metadata

    output_json = (
        Path(args.output_json)
        if args.output_json is not None
        else checkpoint_path.parent / f"phase_c_metrics_{args.eval_split}.json"
    )
    output_md = (
        Path(args.output_md)
        if args.output_md is not None
        else checkpoint_path.parent / f"phase_c_per_joint_mpjpe_{args.eval_split}.md"
    )
    output_graphpose_md = (
        Path(args.output_graphpose_md)
        if args.output_graphpose_md is not None
        else checkpoint_path.parent / f"phase_c_graphpose_benchmark_{args.eval_split}.md"
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_graphpose_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as fd:
        json.dump(metrics, fd, indent=2)
    with open(output_md, "w") as fd:
        fd.write(make_markdown_table(metrics))
    with open(output_graphpose_md, "w") as fd:
        fd.write(make_graphpose_markdown_table(metrics, args.method_name))

    print(
        "eval_split=%s samples=%d mpjpe=%.3f pa_mpjpe=%.3f "
        "pck50mm=%.3f pck100mm=%.3f g_PCK@10=%.3f g_PCK@20=%.3f "
        "g_PCK@30=%.3f g_PCK@40=%.3f g_PCK@50=%.3f normalize_pose=%s "
        "model_config=%s"
        % (
            args.eval_split,
            len(selected_dataset),
            metrics["mpjpe_mm"],
            metrics["pa_mpjpe_mm"],
            metrics["pck_50mm"],
            metrics["pck_100mm"],
            metrics["g_PCK@10"],
            metrics["g_PCK@20"],
            metrics["g_PCK@30"],
            metrics["g_PCK@40"],
            metrics["g_PCK@50"],
            pose_stats.get("enabled", False),
            model.get_model_config(),
        ),
        flush=True,
    )
    print(f"saved_json={output_json}", flush=True)
    print(f"saved_md={output_md}", flush=True)
    print(f"saved_graphpose_md={output_graphpose_md}", flush=True)
    print(
        "collapse_diagnostics: axis_mae_mm=%s root_mpjpe=%.3f "
        "root_centered_mpjpe=%.3f constant_mean_pose_mpjpe=%.3f "
        "constant_mean_pose_pa_mpjpe=%.3f pa_gain_over_constant=%.3f"
        % (
            metrics["axis_mae_mm_by_name"],
            metrics["root_mpjpe_mm"],
            metrics["root_centered_mpjpe_mm"],
            metrics["constant_mean_pose_mpjpe_mm"],
            metrics["constant_mean_pose_pa_mpjpe_mm"],
            metrics["pa_mpjpe_gain_over_constant_mean_pose_mm"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
