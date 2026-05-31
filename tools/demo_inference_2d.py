import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy.io as scio
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import DSKNetTransMMFI, OriginalHPE  # noqa: E402
from utils import compute_pck_pckh  # noqa: E402


JOINT_NAMES = [
    "Bot Torso",
    "L.Hip",
    "L.Knee",
    "L.Foot",
    "R.Hip",
    "R.Knee",
    "R.Foot",
    "Center Torso",
    "Upper Torso",
    "Neck Base",
    "Center Head",
    "R.Shoulder",
    "R.Elbow",
    "R.Hand",
    "L.Shoulder",
    "L.Elbow",
    "L.Hand",
]

SKELETON_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (9, 11),
    (11, 12),
    (12, 13),
    (9, 14),
    (14, 15),
    (15, 16),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a 2D HPE-Li checkpoint on MM-Fi WiFi CSI and export 2D skeleton previews."
    )
    parser.add_argument(
        "--checkpoint",
        default=os.getenv("HPE_LI_2D_CHECKPOINT"),
        help="Path to a 2D checkpoint. Defaults to known HPE_LI_CHECKPOINT locations.",
    )
    parser.add_argument(
        "--dataset-root",
        default=os.getenv(
            "MMFI_DATASET_ROOT", str(PROJECT_ROOT / "data" / "mmfi" / "dataset")
        ),
        help="MM-Fi dataset root containing E01/E02/E03/E04.",
    )
    parser.add_argument("--scene", default="E01")
    parser.add_argument("--subject", default="S01")
    parser.add_argument("--action", default="A01")
    parser.add_argument(
        "--frames",
        type=int,
        nargs="*",
        default=None,
        help="One-indexed frame numbers to render. Overrides start/num/stride.",
    )
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--model",
        default="auto",
        choices=["auto", "OriginalHPE", "DSKNetTransMMFI"],
        help="Only used when loading a state_dict checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "demo_2d"),
        help="Directory for PNG previews, summary JSON, and index HTML.",
    )
    parser.add_argument("--canvas-width", type=int, default=900)
    parser.add_argument("--canvas-height", type=int, default=900)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="GT joints with confidence <= threshold are hidden in GT drawing and ignored in mean error.",
    )
    parser.add_argument(
        "--metric-scale",
        type=float,
        default=1000.0,
        help="Scale raw coordinate error for display, matching the training log convention.",
    )
    parser.add_argument(
        "--make-video",
        action="store_true",
        help="Also write demo_2d.mp4 from rendered frames.",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def default_checkpoint_candidates():
    roots = []
    if os.getenv("HPE_LI_CHECKPOINT"):
        roots.append(Path(os.environ["HPE_LI_CHECKPOINT"]))
    roots.append(PROJECT_ROOT / "output")

    candidates = []
    for root in roots:
        candidates.extend(
            [
                root / "OriginalHPE" / "0.0" / "best.pt",
                root / "att_mmfi" / "best.pt",
            ]
        )
    return candidates


def resolve_checkpoint(path_arg):
    if path_arg:
        path = Path(path_arg).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    for path in default_checkpoint_candidates():
        if path.is_file():
            return path

    tried = "\n".join(str(p) for p in default_checkpoint_candidates())
    raise FileNotFoundError(
        "No checkpoint was provided and no default checkpoint exists. Tried:\n"
        + tried
    )


def instantiate_model(model_name):
    if model_name == "DSKNetTransMMFI":
        return DSKNetTransMMFI()
    return OriginalHPE()


def infer_model_name_from_state_dict(state_dict, fallback):
    if fallback != "auto":
        return fallback
    keys = list(state_dict.keys())
    if any(".tf." in key or key.endswith(".tf.encoder_norm.weight") for key in keys):
        return "DSKNetTransMMFI"
    return "OriginalHPE"


def load_model(checkpoint_path, device, model_name_arg):
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
        model_name = model.__class__.__name__
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model_name = checkpoint.get(
            "model_name", infer_model_name_from_state_dict(state_dict, model_name_arg)
        )
        model = instantiate_model(model_name)
        model.load_state_dict(state_dict)
    else:
        raise TypeError(f"Unsupported checkpoint object: {type(checkpoint)!r}")

    model.to(device)
    model.eval()
    return model, model_name


def load_csi_frame(path):
    data = scio.loadmat(path)["CSIamp"].astype(np.float32)
    data[np.isinf(data)] = np.nan
    for channel_idx in range(data.shape[-1]):
        channel = data[:, :, channel_idx]
        if np.isnan(channel).any():
            finite = channel[np.isfinite(channel)]
            fill_value = float(finite.mean()) if finite.size else 0.0
            channel[np.isnan(channel)] = fill_value
    min_value = float(np.min(data))
    max_value = float(np.max(data))
    denom = max_value - min_value
    if denom <= 1e-12:
        return np.zeros_like(data, dtype=np.float32)
    return ((data - min_value) / denom).astype(np.float32)


def make_frame_numbers(args, total_frames):
    if args.frames:
        frames = args.frames
    else:
        stop = args.start_frame + args.num_frames * args.stride
        frames = list(range(args.start_frame, stop, args.stride))
    frames = [frame for frame in frames if 1 <= frame <= total_frames]
    if not frames:
        raise ValueError("No valid frames were selected.")
    return frames


def load_action_batch(dataset_root, scene, subject, action, frame_numbers):
    action_dir = Path(dataset_root).expanduser() / scene / subject / action
    gt_path = action_dir / "ground_truth.npy"
    csi_dir = action_dir / "wifi-csi"
    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")
    if not csi_dir.is_dir():
        raise FileNotFoundError(f"WiFi CSI directory not found: {csi_dir}")

    gt_all = np.load(gt_path).astype(np.float32)
    csi_frames = []
    gt_frames = []
    for frame_number in frame_numbers:
        csi_path = csi_dir / f"frame{frame_number:03d}.mat"
        if not csi_path.is_file():
            raise FileNotFoundError(f"CSI frame not found: {csi_path}")
        csi_frames.append(load_csi_frame(csi_path))
        gt_frames.append(gt_all[frame_number - 1])

    return np.stack(csi_frames, axis=0), np.stack(gt_frames, axis=0), gt_all


def run_inference(model, csi_array, device, batch_size):
    predictions = []
    with torch.no_grad():
        for start in range(0, csi_array.shape[0], batch_size):
            batch = torch.from_numpy(csi_array[start : start + batch_size]).to(device)
            batch = batch.float()
            pred, _ = model(batch)
            predictions.append(pred.detach().cpu().numpy())
    return np.concatenate(predictions, axis=0).astype(np.float32)


def compute_demo_metrics(pred_xy, gt_pose, confidence_threshold, metric_scale):
    gt_xy = gt_pose[:, :, 0:2]
    if gt_pose.shape[-1] > 2:
        valid = gt_pose[:, :, 2] > confidence_threshold
    else:
        valid = np.ones(gt_xy.shape[:2], dtype=bool)

    raw_errors = np.linalg.norm(pred_xy - gt_xy, axis=2)
    mean_error = float(raw_errors[valid].mean()) if valid.any() else float("nan")
    per_frame = []
    for frame_idx in range(pred_xy.shape[0]):
        frame_valid = valid[frame_idx]
        frame_error = (
            float(raw_errors[frame_idx][frame_valid].mean())
            if frame_valid.any()
            else float("nan")
        )
        per_frame.append(
            {
                "mean_error": frame_error,
                "scaled_mean_error": frame_error * metric_scale,
            }
        )

    pred_for_pck = np.transpose(pred_xy, (0, 2, 1))
    gt_for_pck = np.transpose(gt_xy, (0, 2, 1))
    pck_50 = compute_pck_pckh(pred_for_pck, gt_for_pck, 0.5)[17]
    pck_20 = compute_pck_pckh(pred_for_pck, gt_for_pck, 0.2)[17]

    return {
        "mean_error": mean_error,
        "scaled_mean_error": mean_error * metric_scale,
        "pck_50": float(pck_50),
        "pck_20": float(pck_20),
        "per_frame": per_frame,
    }


def make_transform(points, width, height, pad=80):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    points = points[np.isfinite(points).all(axis=1)]
    if points.size == 0:
        min_xy = np.array([0.0, 0.0], dtype=np.float32)
        max_xy = np.array([1.0, 1.0], dtype=np.float32)
    else:
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)

    span = np.maximum(max_xy - min_xy, 1e-6)
    scale = min((width - 2 * pad) / span[0], (height - 2 * pad) / span[1])

    def transform(point):
        x = pad + (point[0] - min_xy[0]) * scale
        y = height - pad - (point[1] - min_xy[1]) * scale
        return int(round(x)), int(round(y))

    return transform


def draw_skeleton(canvas, points, color, transform, valid=None, thickness=3):
    if valid is None:
        valid = np.ones(len(points), dtype=bool)
    for a, b in SKELETON_EDGES:
        if valid[a] and valid[b]:
            cv2.line(canvas, transform(points[a]), transform(points[b]), color, thickness)
    for idx, point in enumerate(points):
        if valid[idx]:
            cv2.circle(canvas, transform(point), 6, color, -1)
            cv2.circle(canvas, transform(point), 7, (255, 255, 255), 1)


def render_frame(
    output_path,
    pred_xy,
    gt_pose,
    axis_points,
    title,
    frame_metric,
    width,
    height,
    confidence_threshold,
):
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    transform = make_transform(axis_points, width, height)

    for alpha in np.linspace(0.15, 0.85, 5):
        x = int(80 + alpha * (width - 160))
        y = int(80 + alpha * (height - 160))
        cv2.line(canvas, (x, 80), (x, height - 80), (230, 230, 230), 1)
        cv2.line(canvas, (80, y), (width - 80, y), (230, 230, 230), 1)

    gt_xy = gt_pose[:, 0:2]
    valid = (
        gt_pose[:, 2] > confidence_threshold
        if gt_pose.shape[-1] > 2
        else np.ones(gt_xy.shape[0], dtype=bool)
    )

    draw_skeleton(canvas, gt_xy, (70, 150, 70), transform, valid=valid, thickness=3)
    draw_skeleton(canvas, pred_xy, (70, 70, 210), transform, thickness=3)

    cv2.putText(
        canvas,
        title,
        (28, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (35, 35, 35),
        2,
        cv2.LINE_AA,
    )
    metric_text = "mean error: %.3f | scaled: %.1f" % (
        frame_metric["mean_error"],
        frame_metric["scaled_mean_error"],
    )
    cv2.putText(
        canvas,
        metric_text,
        (28, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (60, 60, 60),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "GT",
        (width - 135, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (70, 150, 70),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Pred",
        (width - 135, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (70, 70, 210),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(output_path), canvas)


def write_html(output_dir, frame_records):
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>HPE-Li 2D Demo</title>",
        "<style>body{font-family:Arial,sans-serif;background:#f6f7f9;color:#1c2430;margin:24px;}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:18px;}"
        ".card{background:#fff;border:1px solid #d8dee8;border-radius:8px;padding:12px;}"
        "img{width:100%;height:auto;display:block;}code{background:#eef2f7;padding:2px 4px;border-radius:4px;}</style>",
        "</head><body><h1>HPE-Li 2D Demo</h1><div class='grid'>",
    ]
    for record in frame_records:
        image_name = Path(record["image"]).name
        lines.append(
            "<div class='card'><img src='frames/%s'><p><code>%s</code><br>"
            "mean error %.3f, scaled %.1f</p></div>"
            % (
                image_name,
                record["label"],
                record["mean_error"],
                record["scaled_mean_error"],
            )
        )
    lines.append("</div></body></html>")
    (output_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")


def write_video(output_path, frame_paths, fps=8):
    if not frame_paths:
        return
    first = cv2.imread(str(frame_paths[0]))
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path))
        writer.write(image)
    writer.release()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    output_dir = Path(args.output_dir).expanduser()
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    model, model_name = load_model(checkpoint_path, device, args.model)

    gt_path = (
        Path(args.dataset_root).expanduser()
        / args.scene
        / args.subject
        / args.action
        / "ground_truth.npy"
    )
    gt_all = np.load(gt_path)
    frame_numbers = make_frame_numbers(args, total_frames=gt_all.shape[0])
    csi_array, gt_pose, _ = load_action_batch(
        args.dataset_root, args.scene, args.subject, args.action, frame_numbers
    )

    pred_xy = run_inference(model, csi_array, device, args.batch_size)
    metrics = compute_demo_metrics(
        pred_xy, gt_pose, args.confidence_threshold, args.metric_scale
    )

    axis_points = np.concatenate(
        [gt_pose[:, :, 0:2].reshape(-1, 2), pred_xy.reshape(-1, 2)], axis=0
    )
    frame_records = []
    frame_paths = []
    for idx, frame_number in enumerate(frame_numbers):
        frame_name = f"{args.scene}_{args.subject}_{args.action}_frame{frame_number:03d}.png"
        frame_path = frames_dir / frame_name
        label = f"{args.scene}/{args.subject}/{args.action} frame {frame_number:03d}"
        frame_metric = metrics["per_frame"][idx]
        render_frame(
            frame_path,
            pred_xy[idx],
            gt_pose[idx],
            axis_points,
            label,
            frame_metric,
            args.canvas_width,
            args.canvas_height,
            args.confidence_threshold,
        )
        frame_paths.append(frame_path)
        frame_records.append(
            {
                "frame": frame_number,
                "label": label,
                "image": str(frame_path),
                **frame_metric,
            }
        )

    if args.make_video:
        write_video(output_dir / "demo_2d.mp4", frame_paths)

    summary = {
        "checkpoint": str(checkpoint_path),
        "model_name": model_name,
        "dataset_root": str(Path(args.dataset_root).expanduser()),
        "scene": args.scene,
        "subject": args.subject,
        "action": args.action,
        "frames": frame_numbers,
        "device": str(device),
        "metric_scale": args.metric_scale,
        "mean_error": metrics["mean_error"],
        "scaled_mean_error": metrics["scaled_mean_error"],
        "pck_50": metrics["pck_50"],
        "pck_20": metrics["pck_20"],
        "frames_rendered": frame_records,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fd:
        json.dump(summary, fd, indent=2)
    write_html(output_dir, frame_records)

    print(f"checkpoint={checkpoint_path}")
    print(f"model={model_name} device={device}")
    print(
        "demo mean_error=%.6f scaled_mean_error=%.3f pck_50=%.3f pck_20=%.3f"
        % (
            summary["mean_error"],
            summary["scaled_mean_error"],
            summary["pck_50"],
            summary["pck_20"],
        )
    )
    print(f"wrote {output_dir / 'index.html'}")
    print(f"wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
