import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import HPEWiPoseModel  # noqa: E402
from utils import compute_pck_pckh_18  # noqa: E402
from wipose import WiPoseDataset  # noqa: E402


JOINT_NAMES = [
    "Nose",
    "Neck",
    "R.Shoulder",
    "R.Elbow",
    "R.Wrist",
    "L.Shoulder",
    "L.Elbow",
    "L.Wrist",
    "R.Hip",
    "R.Knee",
    "R.Ankle",
    "L.Hip",
    "L.Knee",
    "L.Ankle",
    "R.Eye",
    "L.Eye",
    "R.Ear",
    "L.Ear",
]

SKELETON_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (8, 11),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a WiPose HPE-Li checkpoint and export 2D skeleton previews."
    )
    parser.add_argument(
        "--checkpoint",
        default=os.getenv("HPE_LI_WIPOSE_CHECKPOINT"),
        required=os.getenv("HPE_LI_WIPOSE_CHECKPOINT") is None,
    )
    parser.add_argument(
        "--dataset-root",
        default=os.getenv("WIPOSE_DATASET_ROOT", str(PROJECT_ROOT / "data" / "wipose")),
        help="WiPose root containing Train/ and Test/.",
    )
    parser.add_argument(
        "--preprocessed-root",
        default=os.getenv("WIPOSE_PREPROCESSED_ROOT"),
        help="Optional directory containing Train.pt and Test.pt.",
    )
    parser.add_argument("--split", default="Test", choices=["Train", "Test"])
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "demo_wipose"),
    )
    parser.add_argument("--canvas-width", type=int, default=900)
    parser.add_argument("--canvas-height", type=int, default=900)
    parser.add_argument("--metric-scale", type=float, default=1000.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--make-video", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model = HPEWiPoseModel()
        model.load_state_dict(state_dict)
    else:
        raise TypeError(f"Unsupported checkpoint object: {type(checkpoint)!r}")

    model.to(device)
    model.eval()
    return model


def make_indices(args, dataset_len):
    if args.indices:
        indices = args.indices
    else:
        stop = args.start_index + args.num_samples * args.stride
        indices = list(range(args.start_index, stop, args.stride))
    indices = [idx for idx in indices if 0 <= idx < dataset_len]
    if not indices:
        raise ValueError("No valid sample indices were selected.")
    return indices


def run_inference(model, inputs, device, batch_size):
    preds = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            batch = inputs[start : start + batch_size].to(device).float()
            pred, _ = model(batch)
            preds.append(pred.detach().cpu())
    return torch.cat(preds, dim=0).numpy()


def compute_metrics(pred_xy, gt_pose, confidence_threshold, metric_scale):
    gt_xy = gt_pose[:, :, 0:2]
    confidence = gt_pose[:, :, 2] if gt_pose.shape[-1] > 2 else np.ones(gt_xy.shape[:2])
    valid = confidence > confidence_threshold
    errors = np.linalg.norm(pred_xy - gt_xy, axis=-1)
    valid_errors = errors[valid]
    mean_error = float(valid_errors.mean()) if valid_errors.size else float(errors.mean())

    pred_for_pck = np.transpose(pred_xy, (0, 2, 1))
    gt_for_pck = np.transpose(gt_xy, (0, 2, 1))
    pck_50 = float(compute_pck_pckh_18(pred_for_pck, gt_for_pck, 0.5)[18])
    pck_20 = float(compute_pck_pckh_18(pred_for_pck, gt_for_pck, 0.2)[18])

    per_sample = []
    for idx in range(pred_xy.shape[0]):
        sample_errors = errors[idx][valid[idx]]
        if sample_errors.size == 0:
            sample_errors = errors[idx]
        sample_mean = float(sample_errors.mean())
        per_sample.append(
            {
                "mean_error": sample_mean,
                "scaled_mean_error": sample_mean * metric_scale,
            }
        )

    return {
        "mean_error": mean_error,
        "scaled_mean_error": mean_error * metric_scale,
        "pck_50": pck_50,
        "pck_20": pck_20,
        "per_sample": per_sample,
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
    valid = gt_pose[:, 2] > confidence_threshold

    draw_skeleton(canvas, gt_xy, (70, 150, 70), transform, valid=valid)
    draw_skeleton(canvas, pred_xy, (70, 70, 210), transform)

    cv2.putText(canvas, title, (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (35, 35, 35), 2, cv2.LINE_AA)
    metric_text = "mean error: %.3f | scaled: %.1f" % (
        frame_metric["mean_error"],
        frame_metric["scaled_mean_error"],
    )
    cv2.putText(canvas, metric_text, (28, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 60), 2, cv2.LINE_AA)
    cv2.putText(canvas, "GT", (width - 135, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 150, 70), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Pred", (width - 135, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 70, 210), 2, cv2.LINE_AA)

    cv2.imwrite(str(output_path), canvas)


def write_html(output_dir, frame_records):
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>HPE-Li WiPose Demo</title>",
        "<style>body{font-family:Arial,sans-serif;background:#f6f7f9;color:#1c2430;margin:24px;}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:18px;}"
        ".card{background:#fff;border:1px solid #d8dee8;border-radius:8px;padding:12px;}"
        "img{width:100%;height:auto;display:block;}code{background:#eef2f7;padding:2px 4px;border-radius:4px;}</style>",
        "</head><body><h1>HPE-Li WiPose Demo</h1><div class='grid'>",
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
        writer.write(cv2.imread(str(frame_path)))
    writer.release()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(args.output_dir).expanduser()
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    dataset = WiPoseDataset(
        root_dir=args.dataset_root,
        split=args.split,
        preprocessed_root=args.preprocessed_root,
    )
    indices = make_indices(args, len(dataset))
    samples = [dataset[idx] for idx in indices]
    inputs = torch.stack([sample["input_wifi-csi"] for sample in samples], dim=0)
    gt_pose = torch.stack([sample["output"] for sample in samples], dim=0).numpy()

    model = load_model(checkpoint_path, device)
    pred_xy = run_inference(model, inputs, device, args.batch_size)
    metrics = compute_metrics(
        pred_xy, gt_pose, args.confidence_threshold, args.metric_scale
    )

    axis_points = np.concatenate(
        [gt_pose[:, :, 0:2].reshape(-1, 2), pred_xy.reshape(-1, 2)], axis=0
    )
    frame_records = []
    frame_paths = []
    for row, sample_index in enumerate(indices):
        frame_path = frames_dir / f"{args.split}_sample{sample_index:05d}.png"
        label = f"{args.split} sample {sample_index}"
        frame_metric = metrics["per_sample"][row]
        render_frame(
            frame_path,
            pred_xy[row],
            gt_pose[row],
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
                "sample_index": sample_index,
                "label": label,
                "image": str(frame_path),
                **frame_metric,
            }
        )

    if args.make_video:
        write_video(output_dir / "demo_wipose.mp4", frame_paths)

    summary = {
        "checkpoint": str(checkpoint_path),
        "dataset_root": str(Path(args.dataset_root).expanduser()),
        "preprocessed_root": str(Path(args.preprocessed_root).expanduser()) if args.preprocessed_root else None,
        "split": args.split,
        "indices": indices,
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
    print(f"model=HPEWiPoseModel device={device}")
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
