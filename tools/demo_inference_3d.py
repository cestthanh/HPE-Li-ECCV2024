import argparse
import glob
import http.server
import json
import mimetypes
import os
import re
import sys
import threading
import time
import warnings
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import OriginalHPE3D, get_hpe3d_model_config
from utils.eval_3d import compute_3d_metrics


# Edit these defaults for the sequence you use most often.
DEFAULT_DATASET_ROOT = r"G:\My Drive\MMFi_lite"
DEFAULT_SCENE = "E01"
DEFAULT_SUBJECT = "S05"
DEFAULT_ACTION = "A06"
DEFAULT_RGB_ROOT = r"D:\RGB_image-MMFi\MMFi_Defaced_RGB"
DEFAULT_CHECKPOINT = r"D:\Thesis_Docs\HPE-Li-ECCV2024\checkpoints\phase_b_s1_random_20e_package\best_s1_random_20e.pt"
DEFAULT_DEVICE = "auto"
DEFAULT_INFERENCE_BATCH_SIZE = 16
DEFAULT_PORT = 8082
DEFAULT_HOST = "localhost"

HTML_PATH = Path(__file__).with_name("demo_inference_3d_web.html")

warnings.filterwarnings(
    "ignore",
    message=r".*urllib3 .* or chardet .*/charset_normalizer .* doesn't match a supported version!.*",
)
try:
    from requests import RequestsDependencyWarning

    warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
except Exception:
    pass


MMFI_17_JOINT_NAMES = [
    "Bot Torso",
    "R.Hip",
    "R.Knee",
    "R.Foot",
    "L.Hip",
    "L.Knee",
    "L.Foot",
    "Center Torso",
    "Upper Torso",
    "Neck Base",
    "Center Head",
    "L.Shoulder",
    "L.Elbow",
    "L.Hand",
    "R.Shoulder",
    "R.Elbow",
    "R.Hand",
]

MMFI_17_BONES = [
    [0, 1],
    [1, 2],
    [2, 3],
    [0, 4],
    [4, 5],
    [5, 6],
    [0, 7],
    [7, 8],
    [8, 9],
    [9, 10],
    [8, 11],
    [11, 12],
    [12, 13],
    [8, 14],
    [14, 15],
    [15, 16],
]

GT_JOINT_COLORS = [
    "#00E5FF",
    "#40C4FF",
    "#40C4FF",
    "#40C4FF",
    "#69F0AE",
    "#69F0AE",
    "#69F0AE",
    "#00E5FF",
    "#00E5FF",
    "#00E5FF",
    "#00E5FF",
    "#69F0AE",
    "#69F0AE",
    "#69F0AE",
    "#40C4FF",
    "#40C4FF",
    "#40C4FF",
]

PRED_JOINT_COLORS = [
    "#FF6D00",
    "#FFD740",
    "#FFD740",
    "#FFD740",
    "#FF4081",
    "#FF4081",
    "#FF4081",
    "#FF6D00",
    "#FF6D00",
    "#FF6D00",
    "#FF6D00",
    "#FF4081",
    "#FF4081",
    "#FF4081",
    "#FFD740",
    "#FFD740",
    "#FFD740",
]

GT_EDGE_COLORS = [
    "#40C4FF",
    "#40C4FF",
    "#40C4FF",
    "#69F0AE",
    "#69F0AE",
    "#69F0AE",
    "#00E5FF",
    "#00E5FF",
    "#00E5FF",
    "#00E5FF",
    "#69F0AE",
    "#69F0AE",
    "#69F0AE",
    "#40C4FF",
    "#40C4FF",
    "#40C4FF",
]

PRED_EDGE_COLORS = [
    "#FFD740",
    "#FFD740",
    "#FFD740",
    "#FF4081",
    "#FF4081",
    "#FF4081",
    "#FF6D00",
    "#FF6D00",
    "#FF6D00",
    "#FF6D00",
    "#FF4081",
    "#FF4081",
    "#FF4081",
    "#FFD740",
    "#FFD740",
    "#FFD740",
]


STATE = {
    "ready": False,
    "status": "init",
    "error": None,
    "gt_frames": None,
    "pred_frames": None,
    "frame_names": [],
    "mpjpe_frames_mm": None,
    "diagnostics": {},
    "paths": {},
    "rgb_source": None,
    "pose_normalization": {"enabled": False},
}
STATE_LOCK = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Serve a browser-based MMFi demo with synchronized RGB, GT 3D pose, "
            "and CSI inference 3D pose."
        )
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument("--action", default=DEFAULT_ACTION)
    parser.add_argument("--rgb-root", default=DEFAULT_RGB_ROOT)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--csi-path", default=None)
    parser.add_argument("--gt-path", default=None)
    parser.add_argument(
        "--video-path",
        default=None,
        help="RGB frame directory or video file. Defaults to rgb-root/scene/subject/action/rgb.",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_INFERENCE_BATCH_SIZE)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load inputs and run inference, then exit without starting the server.",
    )
    parser.add_argument(
        "--no-csi-normalize",
        action="store_true",
        help="Skip MMFi per-frame CSI min-max normalization.",
    )
    return parser.parse_args()


def natural_key(path):
    return [
        int(chunk) if chunk.isdigit() else chunk.lower()
        for chunk in re.split(r"(\d+)", str(path))
    ]


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def resolve_sequence_paths(args):
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    csi_path = Path(args.csi_path) if args.csi_path else None
    gt_path = Path(args.gt_path) if args.gt_path else None
    video_path = Path(args.video_path) if args.video_path else None

    if args.dataset_root and args.scene and args.subject and args.action:
        sequence_root = Path(args.dataset_root) / args.scene / args.subject / args.action
        csi_path = csi_path or sequence_root / "wifi-csi"
        gt_path = gt_path or sequence_root / "ground_truth.npy"

    if video_path is None and args.rgb_root and args.scene and args.subject and args.action:
        candidates = [
            Path(args.rgb_root) / args.scene / args.subject / args.action / "rgb",
            Path(args.rgb_root) / args.scene / args.subject / args.action,
            Path(args.rgb_root) / args.subject / args.action / "rgb",
            Path(args.rgb_root) / args.subject / args.action,
        ]
        video_path = next((candidate for candidate in candidates if candidate.exists()), None)

    if csi_path is None or gt_path is None:
        raise ValueError("Missing CSI/GT path. Supply --csi-path and --gt-path.")
    if not csi_path.exists():
        raise FileNotFoundError(f"CSI path not found: {csi_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"GT path not found: {gt_path}")

    return csi_path, gt_path, video_path


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint, dict) and all(hasattr(v, "shape") for v in checkpoint.values()):
        return checkpoint
    raise ValueError("Unsupported checkpoint format. Expected model_state_dict or raw state_dict.")


def strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_model(checkpoint_path, device):
    checkpoint = load_checkpoint(checkpoint_path, device)
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint.to(device)
        model.eval()
        return model, checkpoint

    state_dict = strip_module_prefix(extract_state_dict(checkpoint))
    model_config = get_hpe3d_model_config(checkpoint, state_dict=state_dict)
    model = OriginalHPE3D(**model_config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


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


def model_output_to_xyz(output):
    if isinstance(output, tuple):
        output = output[0]

    if output.ndim == 2:
        if output.shape[1] == 34:
            output = output.reshape(output.shape[0], 17, 2)
        elif output.shape[1] == 51:
            output = output.reshape(output.shape[0], 17, 3)
        else:
            raise ValueError(f"Unsupported flat model output shape: {tuple(output.shape)}")

    if output.ndim != 3 or output.shape[1] != 17:
        raise ValueError(f"Unsupported model output shape: {tuple(output.shape)}")

    if output.shape[2] == 2:
        z_pad = torch.zeros(
            output.shape[0],
            output.shape[1],
            1,
            dtype=output.dtype,
            device=output.device,
        )
        return torch.cat([output, z_pad], dim=-1), 2

    if output.shape[2] >= 3:
        return output[:, :, 0:3], 3

    raise ValueError(f"Unsupported model output shape: {tuple(output.shape)}")


def normalize_csi_frame(frame):
    frame = np.asarray(frame, dtype=np.float32)
    frame[np.isinf(frame)] = np.nan
    if frame.ndim != 3:
        raise ValueError(f"Expected CSI frame with 3 dimensions, got {frame.shape}.")

    for idx in range(frame.shape[-1]):
        channel = frame[:, :, idx]
        if np.isnan(channel).any():
            valid = channel[~np.isnan(channel)]
            fill_value = float(valid.mean()) if valid.size else 0.0
            channel[np.isnan(channel)] = fill_value

    min_value = np.nanmin(frame)
    max_value = np.nanmax(frame)
    denom = max_value - min_value
    if not np.isfinite(denom) or denom <= 0:
        return np.zeros_like(frame, dtype=np.float32)
    return ((frame - min_value) / denom).astype(np.float32)


def ensure_csi_frame_shape(frame, source):
    frame = np.asarray(frame, dtype=np.float32)
    if frame.shape == (3, 114, 10):
        return frame
    if frame.ndim == 3 and frame.shape[-1] == 3 and frame.shape[:2] == (114, 10):
        return np.transpose(frame, (2, 0, 1))
    raise ValueError(
        f"CSI frame from {source} has shape {frame.shape}. "
        "Expected (3,114,10) or (114,10,3)."
    )


def load_mat_csi(path, normalize=True):
    import scipy.io as scio

    data = scio.loadmat(path)
    if "CSIamp" not in data:
        keys = sorted(key for key in data if not key.startswith("__"))
        raise KeyError(f"{path} does not contain CSIamp. Available keys: {keys}")
    frame = data["CSIamp"]
    if normalize:
        frame = normalize_csi_frame(frame)
    return ensure_csi_frame_shape(frame, path)


def load_npy_csi(path, normalize=True, max_frames=None):
    array = np.load(path)
    if array.ndim == 3:
        frames = [array]
    elif array.ndim == 4:
        frames = [array[idx] for idx in range(min(len(array), max_frames or len(array)))]
    else:
        raise ValueError(f"CSI npy must have shape (N,3,114,10) or (3,114,10), got {array.shape}.")

    processed = []
    for idx, frame in enumerate(frames):
        if normalize:
            frame = normalize_csi_frame(frame)
        processed.append(ensure_csi_frame_shape(frame, f"{path}[{idx}]"))
    return np.stack(processed, axis=0).astype(np.float32)


def load_csi_sequence(path, normalize=True, max_frames=None):
    path = Path(path)
    if path.is_dir():
        mat_files = sorted(glob.glob(str(path / "frame*.mat")), key=natural_key)
        if not mat_files:
            mat_files = sorted(glob.glob(str(path / "*.mat")), key=natural_key)
        if not mat_files:
            raise FileNotFoundError(f"No .mat CSI frames found in {path}.")
        if max_frames is not None:
            mat_files = mat_files[:max_frames]
        frames = [load_mat_csi(mat_path, normalize=normalize) for mat_path in mat_files]
        return np.stack(frames, axis=0).astype(np.float32)

    if path.suffix.lower() == ".mat":
        return load_mat_csi(path, normalize=normalize)[None, ...]
    if path.suffix.lower() == ".npy":
        return load_npy_csi(path, normalize=normalize, max_frames=max_frames)
    raise ValueError(f"Unsupported CSI path extension: {path.suffix}")


def load_gt_sequence(path, max_frames=None):
    gt = np.load(path).astype(np.float32)
    if gt.ndim == 2 and gt.shape == (17, 3):
        gt = gt[None, ...]
    if gt.ndim != 3 or gt.shape[1] != 17 or gt.shape[2] < 3:
        raise ValueError(f"GT must have shape (N,17,3), got {gt.shape}.")
    gt = gt[:, :, 0:3]
    if max_frames is not None:
        gt = gt[:max_frames]
    return gt


def run_inference(model, csi_sequence, device, batch_size, pose_stats_tensors):
    pred_chunks = []
    output_dims = None
    with torch.no_grad():
        for start in range(0, len(csi_sequence), batch_size):
            end = min(start + batch_size, len(csi_sequence))
            csi = torch.from_numpy(csi_sequence[start:end]).to(device).float()
            pred, batch_output_dims = model_output_to_xyz(model(csi))
            if batch_output_dims == 3:
                pred = denormalize_pose(pred, pose_stats_tensors)
            output_dims = output_dims or batch_output_dims
            pred_chunks.append(pred.detach().cpu().numpy())
    if not pred_chunks:
        raise RuntimeError("No CSI frames were available for inference.")
    return np.concatenate(pred_chunks, axis=0).astype(np.float32), output_dims


def make_frame_names(scene, subject, action, count):
    prefix = "_".join(part for part in (scene, subject, action) if part)
    if not prefix:
        prefix = "sequence"
    return [f"{prefix} frame{i + 1:03d}" for i in range(count)]


def compute_mpjpe_frames_mm(pred_pose, gt_pose, output_dims):
    count = min(len(pred_pose), len(gt_pose))
    if output_dims == 2:
        errors = np.linalg.norm(pred_pose[:count, :, :2] - gt_pose[:count, :, :2], axis=-1) * 1000.0
    else:
        errors = np.linalg.norm(pred_pose[:count] - gt_pose[:count], axis=-1) * 1000.0
    return np.mean(errors, axis=1).astype(float).tolist()


def compute_temporal_correlation(pred_pose, gt_pose):
    correlations = []
    correlations_by_axis = [[], [], []]
    count = min(len(pred_pose), len(gt_pose))
    pred_pose = np.asarray(pred_pose[:count], dtype=np.float64)
    gt_pose = np.asarray(gt_pose[:count], dtype=np.float64)

    for joint_idx in range(gt_pose.shape[1]):
        for coord_idx in range(gt_pose.shape[2]):
            gt_series = gt_pose[:, joint_idx, coord_idx]
            pred_series = pred_pose[:, joint_idx, coord_idx]
            if gt_series.std() <= 1e-8 or pred_series.std() <= 1e-8:
                continue
            corr = np.corrcoef(gt_series, pred_series)[0, 1]
            if np.isfinite(corr):
                correlations.append(float(corr))
                correlations_by_axis[coord_idx].append(float(corr))

    if not correlations:
        return {
            "mean": None,
            "median": None,
            "count": 0,
            "by_axis": {
                axis: {"mean": None, "median": None, "count": 0}
                for axis in ("x", "y", "z")
            },
        }

    by_axis = {}
    for axis, axis_correlations in zip(("x", "y", "z"), correlations_by_axis):
        by_axis[axis] = {
            "mean": float(np.mean(axis_correlations)) if axis_correlations else None,
            "median": float(np.median(axis_correlations))
            if axis_correlations
            else None,
            "count": int(len(axis_correlations)),
        }
    return {
        "mean": float(np.mean(correlations)),
        "median": float(np.median(correlations)),
        "count": int(len(correlations)),
        "by_axis": by_axis,
    }


def compute_motion_summary(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if len(pose) < 2:
        zero_axes = {axis: 0.0 for axis in ("x", "y", "z")}
        return {
            "mean_mm": 0.0,
            "median_mm": 0.0,
            "p95_mm": 0.0,
            "max_mm": 0.0,
            "axis_mean_abs_mm": zero_axes,
            "axis_p95_abs_mm": zero_axes,
        }
    delta_mm = np.diff(pose, axis=0) * 1000.0
    step_mm = np.linalg.norm(delta_mm, axis=-1)
    axis_mean_abs = np.mean(np.abs(delta_mm), axis=(0, 1))
    axis_p95_abs = np.percentile(np.abs(delta_mm), 95, axis=(0, 1))
    return {
        "mean_mm": float(np.mean(step_mm)),
        "median_mm": float(np.median(step_mm)),
        "p95_mm": float(np.percentile(step_mm, 95)),
        "max_mm": float(np.max(step_mm)),
        "axis_mean_abs_mm": {
            axis: float(value)
            for axis, value in zip(("x", "y", "z"), axis_mean_abs)
        },
        "axis_p95_abs_mm": {
            axis: float(value)
            for axis, value in zip(("x", "y", "z"), axis_p95_abs)
        },
    }


def compute_temporal_std_summary(pose):
    pose = np.asarray(pose, dtype=np.float64)
    per_joint_axis_std_mm = pose.std(axis=0) * 1000.0
    root_axis_std_mm = pose[:, 0, :].std(axis=0) * 1000.0
    return {
        "joint_mean_axis_std_mm": {
            axis: float(value)
            for axis, value in zip(
                ("x", "y", "z"), per_joint_axis_std_mm.mean(axis=0)
            )
        },
        "joint_median_axis_std_mm": {
            axis: float(value)
            for axis, value in zip(
                ("x", "y", "z"), np.median(per_joint_axis_std_mm, axis=0)
            )
        },
        "root_axis_std_mm": {
            axis: float(value)
            for axis, value in zip(("x", "y", "z"), root_axis_std_mm)
        },
    }


def safe_ratio(numerator, denominator, eps=1e-3):
    if abs(denominator) <= eps:
        return None
    return float(numerator / denominator)


def make_motion_ratios(pred_motion, gt_motion, pred_std, gt_std):
    axis_motion_ratio = {
        axis: safe_ratio(
            pred_motion["axis_mean_abs_mm"][axis],
            gt_motion["axis_mean_abs_mm"][axis],
        )
        for axis in ("x", "y", "z")
    }
    axis_temporal_std_ratio = {
        axis: safe_ratio(
            pred_std["joint_mean_axis_std_mm"][axis],
            gt_std["joint_mean_axis_std_mm"][axis],
        )
        for axis in ("x", "y", "z")
    }
    return {
        "mean_step_ratio": safe_ratio(pred_motion["mean_mm"], gt_motion["mean_mm"]),
        "axis_mean_abs_step_ratio": axis_motion_ratio,
        "axis_temporal_std_ratio": axis_temporal_std_ratio,
    }


def make_diagnostic_warnings(
    metrics, temporal_corr, ratios, gt_std, articulation_ratios=None
):
    warnings_out = []
    if metrics["mpjpe_gain_over_constant_mean_pose_mm"] <= 0:
        warnings_out.append(
            "Prediction MPJPE is not better than a constant sequence-mean pose."
        )
    if temporal_corr["mean"] is not None and temporal_corr["mean"] < 0.25:
        warnings_out.append(
            "Mean coordinate temporal correlation is below 0.25."
        )
    if ratios["mean_step_ratio"] is not None and ratios["mean_step_ratio"] < 0.25:
        warnings_out.append("Predicted mean frame-to-frame motion is below 25% of GT.")
    if (
        articulation_ratios is not None
        and articulation_ratios["mean_step_ratio"] is not None
        and articulation_ratios["mean_step_ratio"] < 0.25
    ):
        warnings_out.append(
            "Predicted root-centered articulation motion is below 25% of GT."
        )
    for axis in ("x", "y", "z"):
        gt_axis_std = gt_std["joint_mean_axis_std_mm"][axis]
        ratio = ratios["axis_temporal_std_ratio"][axis]
        if gt_axis_std >= 5.0 and ratio is not None and ratio < 0.25:
            warnings_out.append(
                f"Predicted {axis.upper()} temporal variation is below 25% of GT."
            )
    return warnings_out


def compute_constant_pose_mpjpe_mm(reference_pose, gt_pose):
    gt_pose = np.asarray(gt_pose, dtype=np.float64)
    pred_pose = np.broadcast_to(reference_pose, gt_pose.shape)
    errors_mm = np.linalg.norm(pred_pose - gt_pose, axis=-1) * 1000.0
    return float(np.mean(errors_mm))


def compute_demo_diagnostics(pred_pose, gt_pose, output_dims):
    count = min(len(pred_pose), len(gt_pose))
    pred_pose = np.asarray(pred_pose[:count], dtype=np.float64)
    gt_pose = np.asarray(gt_pose[:count], dtype=np.float64)

    if output_dims == 2:
        pred_for_metrics = pred_pose.copy()
        pred_for_metrics[:, :, 2] = gt_pose[:, :, 2]
    else:
        pred_for_metrics = pred_pose

    metrics = compute_3d_metrics(pred_for_metrics, gt_pose)
    root_centered_pred = pred_for_metrics - pred_for_metrics[:, 0:1, :]
    root_centered_gt = gt_pose - gt_pose[:, 0:1, :]
    root_centered_mpjpe = (
        np.linalg.norm(root_centered_pred - root_centered_gt, axis=-1).mean() * 1000.0
    )
    temporal_correlation = compute_temporal_correlation(pred_for_metrics, gt_pose)
    gt_motion = compute_motion_summary(gt_pose)
    pred_motion = compute_motion_summary(pred_for_metrics)
    gt_temporal_std = compute_temporal_std_summary(gt_pose)
    pred_temporal_std = compute_temporal_std_summary(pred_for_metrics)
    motion_ratios = make_motion_ratios(
        pred_motion, gt_motion, pred_temporal_std, gt_temporal_std
    )
    gt_root_motion = compute_motion_summary(gt_pose[:, 0:1, :])
    pred_root_motion = compute_motion_summary(pred_for_metrics[:, 0:1, :])
    gt_articulation_motion = compute_motion_summary(root_centered_gt)
    pred_articulation_motion = compute_motion_summary(root_centered_pred)
    gt_articulation_std = compute_temporal_std_summary(root_centered_gt)
    pred_articulation_std = compute_temporal_std_summary(root_centered_pred)
    articulation_motion_ratios = make_motion_ratios(
        pred_articulation_motion,
        gt_articulation_motion,
        pred_articulation_std,
        gt_articulation_std,
    )

    diagnostics = {
        "num_frames": int(count),
        "metric_mode": "xy_with_gt_z" if output_dims == 2 else "xyz",
        "mpjpe_mm": float(metrics["mpjpe_mm"]),
        "pa_mpjpe_mm": float(metrics["pa_mpjpe_mm"]),
        "pck_50mm": float(metrics["pck_50mm"]),
        "pck_100mm": float(metrics["pck_100mm"]),
        "root_centered_mpjpe_mm": float(root_centered_mpjpe),
        "root_mpjpe_mm": float(metrics["root_mpjpe_mm"]),
        "axis_mae_mm_by_name": metrics["axis_mae_mm_by_name"],
        "root_axis_mae_mm_by_name": metrics["root_axis_mae_mm_by_name"],
        "temporal_correlation": temporal_correlation,
        "gt_motion": gt_motion,
        "pred_motion": pred_motion,
        "gt_temporal_std": gt_temporal_std,
        "pred_temporal_std": pred_temporal_std,
        "motion_ratios": motion_ratios,
        "gt_root_motion": gt_root_motion,
        "pred_root_motion": pred_root_motion,
        "gt_articulation_motion": gt_articulation_motion,
        "pred_articulation_motion": pred_articulation_motion,
        "articulation_motion_ratios": articulation_motion_ratios,
        "constant_first_gt_mpjpe_mm": compute_constant_pose_mpjpe_mm(
            gt_pose[0:1], gt_pose
        ),
        "constant_sequence_mean_gt_mpjpe_mm": float(
            metrics["constant_mean_pose_mpjpe_mm"]
        ),
        "mpjpe_gain_over_constant_sequence_mean_mm": float(
            metrics["mpjpe_gain_over_constant_mean_pose_mm"]
        ),
        "pa_mpjpe_gain_over_constant_sequence_mean_mm": float(
            metrics["pa_mpjpe_gain_over_constant_mean_pose_mm"]
        ),
        "pred_coord_min": pred_for_metrics.reshape(-1, 3).min(axis=0).astype(float).tolist(),
        "pred_coord_max": pred_for_metrics.reshape(-1, 3).max(axis=0).astype(float).tolist(),
        "gt_coord_min": gt_pose.reshape(-1, 3).min(axis=0).astype(float).tolist(),
        "gt_coord_max": gt_pose.reshape(-1, 3).max(axis=0).astype(float).tolist(),
    }
    diagnostics["warnings"] = make_diagnostic_warnings(
        metrics,
        temporal_correlation,
        motion_ratios,
        gt_temporal_std,
        articulation_ratios=articulation_motion_ratios,
    )
    return diagnostics


def load_demo_payload(args):
    csi_path, gt_path, video_path = resolve_sequence_paths(args)
    device = resolve_device(args.device)

    status("Loading CSI frames...")
    csi_sequence = load_csi_sequence(
        csi_path,
        normalize=not args.no_csi_normalize,
        max_frames=args.max_frames,
    )

    status("Loading ground truth...")
    gt_pose = load_gt_sequence(gt_path, max_frames=args.max_frames)

    count = min(len(csi_sequence), len(gt_pose))
    csi_sequence = csi_sequence[:count]
    gt_pose = gt_pose[:count]

    status("Loading model...")
    model, checkpoint = load_model(args.checkpoint, device)
    pose_stats = get_pose_normalization(checkpoint)
    pose_stats_tensors = make_pose_stats_tensors(pose_stats, device)

    status(f"Running inference on {count} frames...")
    pred_pose, output_dims = run_inference(
        model,
        csi_sequence,
        device,
        args.batch_size,
        pose_stats_tensors,
    )
    pred_pose = pred_pose[:count]

    frame_names = make_frame_names(args.scene, args.subject, args.action, count)
    mpjpe_frames_mm = compute_mpjpe_frames_mm(pred_pose, gt_pose, output_dims)
    diagnostics = compute_demo_diagnostics(pred_pose, gt_pose, output_dims)

    return {
        "gt_frames": gt_pose.astype(float).tolist(),
        "pred_frames": pred_pose.astype(float).tolist(),
        "frame_names": frame_names,
        "mpjpe_frames_mm": mpjpe_frames_mm,
        "diagnostics": diagnostics,
        "paths": {
            "project_root": str(PROJECT_ROOT),
            "dataset_root": str(args.dataset_root),
            "csi_path": str(csi_path),
            "gt_path": str(gt_path),
            "video_path": str(video_path) if video_path is not None else "",
            "checkpoint": str(Path(args.checkpoint)),
            "device": str(device),
            "model_class": model.__class__.__name__,
            "model_config": model.get_model_config()
            if hasattr(model, "get_model_config")
            else {},
            "model_output_dims": int(output_dims),
            "metric_mode": "xy_mpjpe_mm" if output_dims == 2 else "xyz_mpjpe_mm",
        },
        "rgb_source": str(video_path) if video_path is not None else "",
        "pose_normalization": pose_stats,
    }


def status(text):
    with STATE_LOCK:
        STATE["status"] = text
    print(f"  {text}", flush=True)


def set_state(payload):
    with STATE_LOCK:
        STATE.update(payload)
        STATE["ready"] = True
        STATE["status"] = "ready"
        STATE["error"] = None


def set_error(exc):
    with STATE_LOCK:
        STATE["ready"] = False
        STATE["status"] = f"ERROR: {exc}"
        STATE["error"] = str(exc)


def get_state_snapshot():
    with STATE_LOCK:
        return dict(STATE)


def find_image_frame(directory, frame_idx):
    frame_num = frame_idx + 1
    patterns = [
        f"frame{frame_num:03d}.*",
        f"{frame_num:03d}.*",
        f"{frame_num}.*",
    ]
    for pattern in patterns:
        matches = sorted(Path(directory).glob(pattern), key=natural_key)
        for path in matches:
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                return path
    return None


def read_video_frame_bytes(video_file, frame_idx):
    import cv2

    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {video_file}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 0:
            frame_idx = max(0, min(frame_idx, frame_count - 1))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read video frame {frame_idx}.")
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("Could not encode video frame as JPEG.")
        return buffer.tobytes(), "image/jpeg"
    finally:
        capture.release()


def read_rgb_frame_bytes(rgb_source, frame_idx):
    if not rgb_source:
        return None, None
    source = Path(rgb_source)
    if source.is_dir():
        image_path = find_image_frame(source, frame_idx)
        if image_path is None:
            return None, None
        mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
        return image_path.read_bytes(), mime
    if source.is_file():
        if source.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            mime = mimetypes.guess_type(str(source))[0] or "image/png"
            return source.read_bytes(), mime
        return read_video_frame_bytes(source, frame_idx)
    return None, None


class DemoHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_json(self, payload, status_code=200):
        content = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def send_html(self):
        content = HTML_PATH.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_html()
            return

        snapshot = get_state_snapshot()
        if parsed.path == "/api/status":
            self.send_json(
                {
                    "ready": snapshot["ready"],
                    "status": snapshot["status"],
                    "error": snapshot["error"],
                }
            )
            return

        if parsed.path == "/api/config":
            self.send_json(
                {
                    "joint_names": MMFI_17_JOINT_NAMES,
                    "edges": MMFI_17_BONES,
                    "gt_joint_colors": GT_JOINT_COLORS,
                    "gt_edge_colors": GT_EDGE_COLORS,
                    "pred_joint_colors": PRED_JOINT_COLORS,
                    "pred_edge_colors": PRED_EDGE_COLORS,
                    "paths": snapshot["paths"],
                    "pose_normalization": snapshot["pose_normalization"],
                }
            )
            return

        if parsed.path == "/api/diagnostics":
            self.send_json(snapshot["diagnostics"])
            return

        if not snapshot["ready"]:
            self.send_json({"error": snapshot["status"]}, 503)
            return

        if parsed.path == "/api/data":
            self.send_json(
                {
                    "frames": snapshot["gt_frames"],
                    "frame_names": snapshot["frame_names"],
                    "num_frames": len(snapshot["gt_frames"]),
                }
            )
            return

        if parsed.path == "/api/predict":
            self.send_json(
                {
                    "frames": snapshot["pred_frames"],
                    "num_frames": len(snapshot["pred_frames"]),
                    "mpjpe_frames_mm": snapshot["mpjpe_frames_mm"],
                }
            )
            return

        if parsed.path == "/api/rgb":
            qs = parse_qs(parsed.query)
            frame_idx = int(qs.get("frame", ["0"])[0])
            content, mime = read_rgb_frame_bytes(snapshot["rgb_source"], frame_idx)
            if content is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
            return

        self.send_error(404)


def loader_thread(args):
    try:
        payload = load_demo_payload(args)
        set_state(payload)
        print("  Data ready.", flush=True)
    except Exception as exc:
        set_error(exc)
        print(f"  ERROR: {exc}", flush=True)
        import traceback

        traceback.print_exc()


def main():
    args = parse_args()
    csi_path, gt_path, video_path = resolve_sequence_paths(args)
    device = resolve_device(args.device)

    print("=" * 72)
    print("HPE-Li-3D browser inference demo")
    print("=" * 72)
    print(f"project_root={PROJECT_ROOT}")
    print(f"csi_path={csi_path}")
    print(f"gt_path={gt_path}")
    print(f"video_path={video_path if video_path is not None else 'None'}")
    print(f"checkpoint={args.checkpoint}")
    print(f"device={device}")

    if args.dry_run:
        payload = load_demo_payload(args)
        diagnostics = payload["diagnostics"]
        temporal_corr = diagnostics["temporal_correlation"]
        print(
            "loaded: "
            f"gt_frames={len(payload['gt_frames'])}, "
            f"pred_frames={len(payload['pred_frames'])}, "
            f"first_frame_mpjpe={payload['mpjpe_frames_mm'][0]:.1f}mm"
        )
        print(
            "diagnostics: "
            f"mpjpe={diagnostics['mpjpe_mm']:.1f}mm, "
            f"pa_mpjpe={diagnostics['pa_mpjpe_mm']:.1f}mm, "
            f"pck50={diagnostics['pck_50mm']:.1f}%, "
            f"pck100={diagnostics['pck_100mm']:.1f}%, "
            f"root_centered_mpjpe={diagnostics['root_centered_mpjpe_mm']:.1f}mm"
        )
        print(
            "motion: "
            f"gt_mean_step={diagnostics['gt_motion']['mean_mm']:.1f}mm, "
            f"pred_mean_step={diagnostics['pred_motion']['mean_mm']:.1f}mm, "
            f"step_ratio={diagnostics['motion_ratios']['mean_step_ratio']}, "
            f"articulation_ratio={diagnostics['articulation_motion_ratios']['mean_step_ratio']}, "
            f"coord_corr_mean={temporal_corr['mean']}"
        )
        print(
            "axis diagnostics: "
            f"mae_mm={diagnostics['axis_mae_mm_by_name']}, "
            f"temporal_std_ratio={diagnostics['motion_ratios']['axis_temporal_std_ratio']}, "
            f"axis_corr={temporal_corr['by_axis']}"
        )
        print(
            "constant baselines: "
            f"first_gt={diagnostics['constant_first_gt_mpjpe_mm']:.1f}mm, "
            f"sequence_mean_gt={diagnostics['constant_sequence_mean_gt_mpjpe_mm']:.1f}mm, "
            f"model_gain={diagnostics['mpjpe_gain_over_constant_sequence_mean_mm']:.1f}mm"
        )
        if diagnostics["warnings"]:
            print("warnings: " + " | ".join(diagnostics["warnings"]))
        return

    http.server.ThreadingHTTPServer.allow_reuse_address = True
    server = http.server.ThreadingHTTPServer((args.host, args.port), DemoHandler)
    url = f"http://{args.host}:{args.port}/index.html"

    print(f"server={url}")
    print("Loading runs in the background; keep this terminal open.")
    print("Press Ctrl+C to stop.")

    loader = threading.Thread(target=loader_thread, args=(args,), daemon=True)
    loader.start()

    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
