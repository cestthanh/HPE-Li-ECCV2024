import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_lib.specs import get_dataset_spec
from dataset_lib.wipose import WiPoseDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect WiPose CSI/SkeletonPoints before Phase B 3D training."
    )
    parser.add_argument(
        "--dataset-root",
        default=os.getenv("WIPOSE_DATASET_ROOT", str(PROJECT_ROOT / "data" / "wipose")),
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "dataset_lib" / "wipose_config.yaml"),
    )
    parser.add_argument("--split", default=None, help="Override config split, e.g. Train or Test.")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as fd:
        return yaml.load(fd, Loader=yaml.FullLoader)


def summarize_array(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def make_dataset(args, config):
    train_cfg = dict(config.get("train_dataset", {}))
    split = args.split or train_cfg.get("split", "Train")
    return WiPoseDataset(
        args.dataset_root,
        split=split,
        preprocessed_root=config.get("preprocessed_root")
        or os.getenv("WIPOSE_PREPROCESSED_ROOT"),
        num_joints=int(config.get("num_joints", 18)),
        pose_scale=float(config.get("pose_scale", 1.0)),
        normalize_csi=bool(config.get("normalize_csi", True)),
    )


def collect_stats(dataset, max_samples):
    csi_chunks = []
    pose_chunks = []
    file_paths = []
    count = min(len(dataset), max_samples)
    for idx in range(count):
        sample = dataset[idx]
        csi = sample["input_wifi-csi"].detach().cpu().numpy()
        pose = sample["output"].detach().cpu().numpy()
        csi_chunks.append(csi)
        pose_chunks.append(pose)
        file_paths.append(sample.get("file_path", str(idx)))

    if not csi_chunks:
        raise RuntimeError("No WiPose samples found for inspection.")

    csi_all = np.stack(csi_chunks, axis=0)
    pose_all = np.stack(pose_chunks, axis=0)
    return {
        "num_samples_checked": int(count),
        "first_file": file_paths[0],
        "csi_shape": list(csi_all.shape[1:]),
        "pose_shape": list(pose_all.shape[1:]),
        "csi_stats": summarize_array(csi_all),
        "pose_stats": summarize_array(pose_all),
        "pose_axis_stats": {
            "x": summarize_array(pose_all[:, :, 0]),
            "y": summarize_array(pose_all[:, :, 1]),
            "z_or_confidence": summarize_array(pose_all[:, :, 2]),
        },
    }


def make_markdown(payload):
    lines = [
        "# WiPose 3D Data Inspection",
        "",
        f"- dataset_root: `{payload['dataset_root']}`",
        f"- config: `{payload['config']}`",
        f"- split: `{payload['split']}`",
        f"- samples_total: `{payload['samples_total']}`",
        f"- samples_checked: `{payload['num_samples_checked']}`",
        f"- first_file: `{payload['first_file']}`",
        f"- csi_shape: `{payload['csi_shape']}`",
        f"- pose_shape: `{payload['pose_shape']}`",
        f"- pose_scale: `{payload['pose_scale']}`",
        f"- pose_unit_after_scale: `{payload['pose_unit_after_scale']}`",
        "",
        "## Pose Axis Stats",
        "",
        "| Axis | Min | Max | Mean | Std |",
        "|---|---:|---:|---:|---:|",
    ]
    for axis, stats in payload["pose_axis_stats"].items():
        lines.append(
            f"| {axis} | {stats['min']:.6f} | {stats['max']:.6f} | "
            f"{stats['mean']:.6f} | {stats['std']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- If `z_or_confidence` has a very small discrete range, verify whether it is depth or confidence.",
            "- If pose values are already millimeters, do not multiply by 1000 again in metrics.",
            "- If pose values are meters after `pose_scale`, MPJPE can be reported in mm by multiplying errors by 1000.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    config = load_config(args.config)
    spec = get_dataset_spec("wipose")
    dataset = make_dataset(args, config)
    stats = collect_stats(dataset, args.num_samples)
    payload = {
        "dataset_name": "wipose",
        "dataset_root": args.dataset_root,
        "config": args.config,
        "split": dataset.split,
        "samples_total": len(dataset),
        "spec_num_joints": spec.num_joints,
        "spec_joint_names": list(spec.joint_names),
        "spec_skeleton_edges": [list(edge) for edge in spec.skeleton_edges],
        "pose_scale": float(config.get("pose_scale", 1.0)),
        "pose_unit_after_scale": config.get("pose_unit_after_scale", "unknown"),
        **stats,
    }

    print(json.dumps(payload, indent=2), flush=True)

    if args.output_json is not None:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as fd:
            json.dump(payload, fd, indent=2)
        print(f"saved_json={output_json}", flush=True)

    if args.output_md is not None:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        with open(output_md, "w") as fd:
            fd.write(make_markdown(payload))
        print(f"saved_md={output_md}", flush=True)


if __name__ == "__main__":
    main()
