import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import tools.demo_inference_3d as demo
from dataset_lib.mmfi import MMFi_Database, MMFi_Dataset, decode_config
from tools.demo_inference_phase_c_3d import load_phase_c_model
from utils.eval_3d import compute_3d_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether MMFi action-specific demo failures come from "
            "split membership, loader mismatch, GT motion, or model prediction."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "dataset_lib" / "config.yaml")
    )
    parser.add_argument("--split-to-use", default=None)
    parser.add_argument("--scene", default="E01")
    parser.add_argument("--subject", default="S05")
    parser.add_argument(
        "--actions",
        default=None,
        help="Comma-separated actions. Defaults to all action directories for scene/subject.",
    )
    parser.add_argument("--max-frames", type=int, default=297)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--skip-direct-compare",
        action="store_true",
        help="Skip direct-vs-dataset loader comparison to make the scan faster.",
    )
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def load_config(path, split_to_use=None):
    with open(path, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    config = copy.deepcopy(config)
    if split_to_use is not None:
        config["split_to_use"] = split_to_use
    return config


def list_actions(dataset_root, scene, subject, actions_arg):
    if actions_arg:
        return [action.strip() for action in actions_arg.split(",") if action.strip()]

    subject_dir = Path(dataset_root) / scene / subject
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

    return [
        path.name
        for path in sorted(subject_dir.iterdir(), key=lambda path: demo.natural_key(path.name))
        if path.is_dir() and path.name.startswith("A")
    ]


def get_split_membership(config, subject, action):
    decoded = decode_config(config)
    train_actions = decoded["train_dataset"]["data_form"].get(subject, [])
    eval_actions = decoded["val_dataset"]["data_form"].get(subject, [])
    if action in train_actions:
        return "train"
    if action in eval_actions:
        return "eval"
    return "not_in_split"


def load_with_dataset_loader(database, subject, action, max_frames):
    dataset = MMFi_Dataset(
        database,
        data_unit="frame",
        modality="wifi-csi",
        split="diagnostic",
        data_form={subject: [action]},
    )
    csi_frames = []
    gt_frames = []
    frame_indices = []
    for dataset_idx in range(len(dataset)):
        if max_frames is not None and len(csi_frames) >= max_frames:
            break
        sample = dataset[dataset_idx]
        csi_frames.append(np.asarray(sample["input_wifi-csi"], dtype=np.float32))
        gt_frames.append(np.asarray(demo.tensor_to_numpy(sample["output"]), dtype=np.float32)[:, 0:3])
        frame_indices.append(int(sample.get("idx", dataset_idx)))

    if not csi_frames:
        raise RuntimeError(f"No frames loaded for {subject}/{action}.")

    return {
        "csi": np.stack(csi_frames, axis=0).astype(np.float32),
        "gt": np.stack(gt_frames, axis=0).astype(np.float32),
        "first_idx": int(frame_indices[0]),
        "last_idx": int(frame_indices[-1]),
        "num_frames": int(len(csi_frames)),
    }


def load_with_direct_loader(dataset_root, scene, subject, action, max_frames):
    sequence_root = Path(dataset_root) / scene / subject / action
    csi_path = sequence_root / "wifi-csi"
    gt_path = sequence_root / "ground_truth.npy"
    return {
        "csi": demo.load_csi_sequence(csi_path, normalize=True, max_frames=max_frames),
        "gt": demo.load_gt_sequence(gt_path, max_frames=max_frames),
    }


def compare_loaders(dataset_payload, direct_payload):
    count = min(
        len(dataset_payload["csi"]),
        len(dataset_payload["gt"]),
        len(direct_payload["csi"]),
        len(direct_payload["gt"]),
    )
    if count == 0:
        return {
            "checked": False,
            "reason": "No overlapping frames.",
        }

    dataset_csi = dataset_payload["csi"][:count]
    direct_csi = direct_payload["csi"][:count]
    dataset_gt = dataset_payload["gt"][:count]
    direct_gt = direct_payload["gt"][:count]
    if dataset_csi.shape != direct_csi.shape or dataset_gt.shape != direct_gt.shape:
        return {
            "checked": True,
            "num_compared_frames": int(count),
            "shape_mismatch": {
                "dataset_csi": list(dataset_csi.shape),
                "direct_csi": list(direct_csi.shape),
                "dataset_gt": list(dataset_gt.shape),
                "direct_gt": list(direct_gt.shape),
            },
        }

    return {
        "checked": True,
        "num_compared_frames": int(count),
        "csi_max_abs_diff": float(np.max(np.abs(dataset_csi - direct_csi))),
        "csi_mean_abs_diff": float(np.mean(np.abs(dataset_csi - direct_csi))),
        "gt_max_abs_diff": float(np.max(np.abs(dataset_gt - direct_gt))),
        "gt_mean_abs_diff": float(np.mean(np.abs(dataset_gt - direct_gt))),
    }


def compute_gt_summary(gt_pose):
    gt_pose = np.asarray(gt_pose, dtype=np.float64)
    if len(gt_pose) < 2:
        return {
            "gt_mean_step_mm": 0.0,
            "oracle_sequence_mean_mpjpe_mm": 0.0,
        }

    step = np.linalg.norm(np.diff(gt_pose, axis=0), axis=-1) * 1000.0
    mean_pose = gt_pose.mean(axis=0, keepdims=True)
    oracle = np.linalg.norm(gt_pose - mean_pose, axis=-1) * 1000.0
    return {
        "gt_mean_step_mm": float(step.mean()),
        "oracle_sequence_mean_mpjpe_mm": float(oracle.mean()),
    }


def evaluate_model_on_sequence(model, device, batch_size, pose_stats_tensors, csi, gt):
    pred, output_dims = demo.run_inference(
        model,
        csi,
        device,
        batch_size,
        pose_stats_tensors,
    )
    count = min(len(pred), len(gt))
    pred = pred[:count]
    gt = gt[:count]
    metrics = compute_3d_metrics(pred, gt)
    diagnostics = demo.compute_demo_diagnostics(pred, gt, output_dims)
    return {
        "mpjpe_mm": float(metrics["mpjpe_mm"]),
        "pa_mpjpe_mm": float(metrics["pa_mpjpe_mm"]),
        "pck_100mm": float(metrics["pck_100mm"]),
        "root_centered_mpjpe_mm": float(metrics["root_centered_mpjpe_mm"]),
        "axis_mae_mm": metrics["axis_mae_mm_by_name"],
        "coord_corr_mean": diagnostics["temporal_correlation"]["mean"],
        "model_gain_over_oracle_mean_mm": float(
            diagnostics["mpjpe_gain_over_constant_sequence_mean_mm"]
        ),
        "warnings": diagnostics["warnings"],
    }


def print_table(rows, has_model):
    header = [
        "action",
        "split",
        "frames",
        "csi_diff",
        "gt_diff",
        "gt_step",
        "oracle",
    ]
    if has_model:
        header.extend(["mpjpe", "pa", "pck100", "z_mae", "corr", "gain"])
    print(" ".join(f"{name:>10}" for name in header), flush=True)

    for row in rows:
        loader = row.get("loader_consistency") or {}
        values = [
            row["action"],
            row["split_membership"],
            str(row["dataset_frames"]),
            format_optional(loader.get("csi_max_abs_diff")),
            format_optional(loader.get("gt_max_abs_diff")),
            f"{row['gt_summary']['gt_mean_step_mm']:.1f}",
            f"{row['gt_summary']['oracle_sequence_mean_mpjpe_mm']:.1f}",
        ]
        if has_model:
            model = row.get("model_metrics") or {}
            axis_mae = model.get("axis_mae_mm") or {}
            values.extend(
                [
                    format_optional(model.get("mpjpe_mm")),
                    format_optional(model.get("pa_mpjpe_mm")),
                    format_optional(model.get("pck_100mm")),
                    format_optional(axis_mae.get("z")),
                    format_optional(model.get("coord_corr_mean"), precision=3),
                    format_optional(model.get("model_gain_over_oracle_mean_mm")),
                ]
            )
        print(" ".join(f"{value:>10}" for value in values), flush=True)


def format_optional(value, precision=1):
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{precision}f}"


def main():
    args = parse_args()
    config = load_config(args.config, split_to_use=args.split_to_use)
    actions = list_actions(args.dataset_root, args.scene, args.subject, args.actions)
    database = MMFi_Database(str(Path(args.dataset_root)))

    model = None
    pose_stats_tensors = None
    device = None
    if args.checkpoint:
        device = resolve_device(args.device)
        model, checkpoint = load_phase_c_model(args.checkpoint, device)
        pose_stats = demo.get_pose_normalization(checkpoint)
        pose_stats_tensors = demo.make_pose_stats_tensors(pose_stats, device)
        print(
            f"checkpoint={args.checkpoint} device={device} "
            f"pose_normalization={pose_stats.get('enabled', False)}",
            flush=True,
        )

    rows = []
    for action in actions:
        row = {
            "scene": args.scene,
            "subject": args.subject,
            "action": action,
            "split_membership": get_split_membership(config, args.subject, action),
        }
        try:
            dataset_payload = load_with_dataset_loader(
                database, args.subject, action, args.max_frames
            )
            row["dataset_frames"] = dataset_payload["num_frames"]
            row["first_frame_idx"] = dataset_payload["first_idx"]
            row["last_frame_idx"] = dataset_payload["last_idx"]
            row["gt_summary"] = compute_gt_summary(dataset_payload["gt"])

            if not args.skip_direct_compare:
                direct_payload = load_with_direct_loader(
                    args.dataset_root,
                    args.scene,
                    args.subject,
                    action,
                    args.max_frames,
                )
                row["loader_consistency"] = compare_loaders(
                    dataset_payload, direct_payload
                )
            else:
                row["loader_consistency"] = {"checked": False, "reason": "skipped"}

            if model is not None:
                row["model_metrics"] = evaluate_model_on_sequence(
                    model,
                    device,
                    args.batch_size,
                    pose_stats_tensors,
                    dataset_payload["csi"],
                    dataset_payload["gt"],
                )
        except Exception as exc:
            row["error"] = str(exc)
            row["dataset_frames"] = 0
            row["gt_summary"] = {
                "gt_mean_step_mm": float("nan"),
                "oracle_sequence_mean_mpjpe_mm": float("nan"),
            }
        rows.append(row)

    print(
        f"dataset_root={args.dataset_root} scene={args.scene} subject={args.subject} "
        f"split_to_use={config['split_to_use']} max_frames={args.max_frames}",
        flush=True,
    )
    print_table(rows, has_model=model is not None)

    errors = [row for row in rows if row.get("error")]
    if errors:
        print("errors:", flush=True)
        for row in errors:
            print(f"  {row['action']}: {row['error']}", flush=True)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fd:
            json.dump(rows, fd, indent=2)
        print(f"saved_json={output_path}", flush=True)


if __name__ == "__main__":
    main()
