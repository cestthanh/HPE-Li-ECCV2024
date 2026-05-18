import math

import numpy as np


def _to_numpy(array):
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array, dtype=np.float64)


def _validate_pose_arrays(pred_xyz, gt_xyz):
    pred_xyz = _to_numpy(pred_xyz)
    gt_xyz = _to_numpy(gt_xyz)
    if pred_xyz.shape != gt_xyz.shape:
        raise ValueError(
            f"Shape mismatch: pred_xyz={pred_xyz.shape}, gt_xyz={gt_xyz.shape}"
        )
    if pred_xyz.ndim != 3 or pred_xyz.shape[-1] != 3:
        raise ValueError(
            "Expected pose arrays with shape (batch, joints, 3), "
            f"got {pred_xyz.shape}"
        )
    return pred_xyz, gt_xyz


def mpjpe_mm(pred_xyz, gt_xyz):
    pred_xyz, gt_xyz = _validate_pose_arrays(pred_xyz, gt_xyz)
    dist_mm = np.linalg.norm(pred_xyz - gt_xyz, axis=-1) * 1000.0
    return float(np.mean(dist_mm))


def pck_3d_mm(pred_xyz, gt_xyz, threshold_mm=50.0):
    pred_xyz, gt_xyz = _validate_pose_arrays(pred_xyz, gt_xyz)
    dist_mm = np.linalg.norm(pred_xyz - gt_xyz, axis=-1) * 1000.0
    return float(np.mean(dist_mm <= threshold_mm) * 100.0)


def _align_by_similarity_transform(pred, gt, eps=1e-8):
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        return None

    mu_gt = gt.mean(axis=0)
    mu_pred = pred.mean(axis=0)
    gt_centered = gt - mu_gt
    pred_centered = pred - mu_pred

    norm_gt = np.sqrt(np.sum(gt_centered**2))
    norm_pred = np.sqrt(np.sum(pred_centered**2))
    if norm_gt < eps or norm_pred < eps:
        return None

    gt_centered = gt_centered / norm_gt
    pred_centered = pred_centered / norm_pred

    try:
        u, s, vt = np.linalg.svd(gt_centered.T @ pred_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    v = vt.T
    rotation = v @ u.T
    if np.linalg.det(rotation) < 0:
        v[:, -1] *= -1
        s[-1] *= -1
        rotation = v @ u.T

    scale = s.sum() * norm_gt / norm_pred
    translation = mu_gt - scale * (mu_pred @ rotation)
    return scale * (pred @ rotation) + translation


def pa_mpjpe_mm(pred_xyz, gt_xyz, return_invalid_count=False):
    pred_xyz, gt_xyz = _validate_pose_arrays(pred_xyz, gt_xyz)
    errors = []
    invalid_count = 0

    for pred_frame, gt_frame in zip(pred_xyz, gt_xyz):
        aligned_pred = _align_by_similarity_transform(pred_frame, gt_frame)
        if aligned_pred is None:
            invalid_count += 1
            continue
        frame_error = np.linalg.norm(aligned_pred - gt_frame, axis=-1).mean()
        errors.append(frame_error * 1000.0)

    value = float(np.mean(errors)) if errors else math.nan
    if return_invalid_count:
        return value, invalid_count
    return value


def compute_3d_metrics(pred_xyz, gt_xyz):
    pa_mpjpe, invalid_count = pa_mpjpe_mm(
        pred_xyz, gt_xyz, return_invalid_count=True
    )
    return {
        "mpjpe_mm": mpjpe_mm(pred_xyz, gt_xyz),
        "pa_mpjpe_mm": pa_mpjpe,
        "pa_mpjpe_invalid_count": invalid_count,
        "pck_50mm": pck_3d_mm(pred_xyz, gt_xyz, threshold_mm=50.0),
        "pck_100mm": pck_3d_mm(pred_xyz, gt_xyz, threshold_mm=100.0),
    }
