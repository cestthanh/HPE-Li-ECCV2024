import math

import numpy as np


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
XYZ_AXIS_NAMES = ("x", "y", "z")

GRAPHPOSE_PCK_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)
GRAPHPOSE_MMFI_SCALE_JOINTS = (5, 12)


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


def per_joint_mpjpe_mm(pred_xyz, gt_xyz):
    pred_xyz, gt_xyz = _validate_pose_arrays(pred_xyz, gt_xyz)
    dist_mm = np.linalg.norm(pred_xyz - gt_xyz, axis=-1) * 1000.0
    return np.mean(dist_mm, axis=0).astype(float).tolist()


def pck_3d_mm(pred_xyz, gt_xyz, threshold_mm=50.0):
    pred_xyz, gt_xyz = _validate_pose_arrays(pred_xyz, gt_xyz)
    dist_mm = np.linalg.norm(pred_xyz - gt_xyz, axis=-1) * 1000.0
    return float(np.mean(dist_mm <= threshold_mm) * 100.0)


def graphpose_body_scale_mmfi(gt_xyz, scale_joints=GRAPHPOSE_MMFI_SCALE_JOINTS):
    """Body scale used by GraphPose-Fi for MMFi CSI PCK."""
    gt_xyz = _to_numpy(gt_xyz)
    joint_a, joint_b = scale_joints
    return np.linalg.norm(gt_xyz[:, joint_a, :] - gt_xyz[:, joint_b, :], axis=-1)


def graphpose_pck_mmfi(pred_xyz, gt_xyz, threshold, eps=1e-8):
    """GraphPose-Fi compatible PCK for MMFi 17-joint 3D pose.

    `threshold=0.5` corresponds to benchmark column `g_PCK@50`, meaning
    the joint error is at most `0.5 * body_scale`. It is not a 50 mm threshold.
    """
    pred_xyz, gt_xyz = _validate_pose_arrays(pred_xyz, gt_xyz)
    if pred_xyz.shape[1] <= max(GRAPHPOSE_MMFI_SCALE_JOINTS):
        raise ValueError(
            "GraphPose-style MMFi PCK expects at least "
            f"{max(GRAPHPOSE_MMFI_SCALE_JOINTS) + 1} joints, got {pred_xyz.shape[1]}"
        )

    dist = np.linalg.norm(pred_xyz - gt_xyz, axis=-1)
    scale = graphpose_body_scale_mmfi(gt_xyz)
    dist_norm = np.divide(
        dist,
        scale[:, None],
        out=np.full_like(dist, np.inf),
        where=scale[:, None] > eps,
    )
    return float(np.mean(dist_norm <= threshold) * 100.0)


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
    pred_xyz, gt_xyz = _validate_pose_arrays(pred_xyz, gt_xyz)
    per_joint = per_joint_mpjpe_mm(pred_xyz, gt_xyz)
    per_joint_by_name = {
        name: value for name, value in zip(MMFI_17_JOINT_NAMES, per_joint)
    }
    pa_mpjpe, invalid_count = pa_mpjpe_mm(
        pred_xyz, gt_xyz, return_invalid_count=True
    )
    axis_mae = np.mean(np.abs(pred_xyz - gt_xyz), axis=(0, 1)) * 1000.0
    root_axis_mae = np.mean(
        np.abs(pred_xyz[:, 0, :] - gt_xyz[:, 0, :]), axis=0
    ) * 1000.0
    root_mpjpe = (
        np.linalg.norm(pred_xyz[:, 0, :] - gt_xyz[:, 0, :], axis=-1).mean()
        * 1000.0
    )
    root_centered_pred = pred_xyz - pred_xyz[:, 0:1, :]
    root_centered_gt = gt_xyz - gt_xyz[:, 0:1, :]
    root_centered_mpjpe = (
        np.linalg.norm(root_centered_pred - root_centered_gt, axis=-1).mean()
        * 1000.0
    )

    pred_coord_std = pred_xyz.reshape(-1, 3).std(axis=0) * 1000.0
    gt_coord_std = gt_xyz.reshape(-1, 3).std(axis=0) * 1000.0
    coord_std_ratio = np.divide(
        pred_coord_std,
        gt_coord_std,
        out=np.full(3, np.nan, dtype=np.float64),
        where=gt_coord_std > 1e-8,
    )

    constant_mean_pose = np.broadcast_to(
        gt_xyz.mean(axis=0, keepdims=True), gt_xyz.shape
    )
    constant_pa_mpjpe, constant_pa_invalid_count = pa_mpjpe_mm(
        constant_mean_pose, gt_xyz, return_invalid_count=True
    )
    constant_mpjpe = mpjpe_mm(constant_mean_pose, gt_xyz)
    metrics = {
        "mpjpe_mm": mpjpe_mm(pred_xyz, gt_xyz),
        "pa_mpjpe_mm": pa_mpjpe,
        "pa_mpjpe_invalid_count": invalid_count,
        "per_joint_mpjpe_mm": per_joint,
        "per_joint_mpjpe_mm_by_name": per_joint_by_name,
        "pck_50mm": pck_3d_mm(pred_xyz, gt_xyz, threshold_mm=50.0),
        "pck_100mm": pck_3d_mm(pred_xyz, gt_xyz, threshold_mm=100.0),
        "axis_mae_mm": axis_mae.astype(float).tolist(),
        "axis_mae_mm_by_name": {
            name: float(value) for name, value in zip(XYZ_AXIS_NAMES, axis_mae)
        },
        "root_mpjpe_mm": float(root_mpjpe),
        "root_axis_mae_mm_by_name": {
            name: float(value)
            for name, value in zip(XYZ_AXIS_NAMES, root_axis_mae)
        },
        "root_centered_mpjpe_mm": float(root_centered_mpjpe),
        "pred_coord_std_mm_by_name": {
            name: float(value)
            for name, value in zip(XYZ_AXIS_NAMES, pred_coord_std)
        },
        "gt_coord_std_mm_by_name": {
            name: float(value) for name, value in zip(XYZ_AXIS_NAMES, gt_coord_std)
        },
        "coord_std_ratio_by_name": {
            name: float(value)
            for name, value in zip(XYZ_AXIS_NAMES, coord_std_ratio)
        },
        "constant_mean_pose_mpjpe_mm": float(constant_mpjpe),
        "constant_mean_pose_pa_mpjpe_mm": float(constant_pa_mpjpe),
        "constant_mean_pose_pa_invalid_count": int(constant_pa_invalid_count),
        "mpjpe_gain_over_constant_mean_pose_mm": float(
            constant_mpjpe - mpjpe_mm(pred_xyz, gt_xyz)
        ),
        "pa_mpjpe_gain_over_constant_mean_pose_mm": float(
            constant_pa_mpjpe - pa_mpjpe
        ),
        "constant_mean_pose_pck_50mm": pck_3d_mm(
            constant_mean_pose, gt_xyz, threshold_mm=50.0
        ),
        "constant_mean_pose_pck_100mm": pck_3d_mm(
            constant_mean_pose, gt_xyz, threshold_mm=100.0
        ),
        "g_PCK_scale_joints": list(GRAPHPOSE_MMFI_SCALE_JOINTS),
    }
    for threshold in GRAPHPOSE_PCK_THRESHOLDS:
        tag = int(round(threshold * 100))
        metrics[f"g_PCK@{tag}"] = graphpose_pck_mmfi(pred_xyz, gt_xyz, threshold)
        metrics[f"constant_mean_pose_g_PCK@{tag}"] = graphpose_pck_mmfi(
            constant_mean_pose, gt_xyz, threshold
        )
    return metrics
