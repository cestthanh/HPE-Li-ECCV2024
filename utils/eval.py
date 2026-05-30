#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def compute_pck_pckh(dt_kpts, gt_kpts, thr):
    """Compute MM-Fi 17-joint PCK using the original body-scale convention."""
    dt = np.array(dt_kpts)
    gt = np.array(gt_kpts)
    assert dt.shape[0] == gt.shape[0]
    kpts_num = gt.shape[2]

    scale = np.sqrt(np.sum(np.square(gt[:, :, 1] - gt[:, :, 11]), 1))
    dist = (
        np.sqrt(np.sum(np.square(dt - gt), 1))
        / np.tile(scale, (gt.shape[2], 1)).T
    )

    pck = np.zeros(gt.shape[2] + 1)
    for kpt_idx in range(kpts_num):
        pck[kpt_idx] = 100 * np.mean(dist[:, kpt_idx] <= thr)
    pck[17] = 100 * np.mean(dist <= thr)
    return pck


def compute_similarity_transform(X, Y, compute_optimal_scale=False):
    """A small NumPy port of MATLAB's `procrustes` function."""
    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.0).sum()
    ssY = (Y0**2.0).sum()

    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    X0 /= normX
    Y0 /= normY

    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    detT = np.linalg.det(T)
    if detT < 0:
        V[:, -1] *= -1
        s[-1] *= -1
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    c = muX - b * np.dot(muY, T)

    return d, Z, T, b, c


def calulate_error(predicted_keypoints, ground_truth_keypoints):
    """Compute MPJPE and PA-MPJPE for matching pose arrays."""
    predicted_keypoints = np.array(predicted_keypoints)
    ground_truth_keypoints = np.array(ground_truth_keypoints)
    assert predicted_keypoints.shape == ground_truth_keypoints.shape

    N = predicted_keypoints.shape[0]
    mpjpe = np.mean(
        np.sqrt(
            np.sum(
                np.square(predicted_keypoints - ground_truth_keypoints), axis=2
            )
        )
    )

    pampjpe = np.zeros(N)
    for n in range(N):
        frame_pred = predicted_keypoints[n]
        frame_gt = ground_truth_keypoints[n]
        _, _, T, b, c = compute_similarity_transform(
            frame_gt, frame_pred, compute_optimal_scale=True
        )
        frame_pred_transformed = (b * frame_pred @ T) + c
        pampjpe[n] = np.mean(
            np.sqrt(np.sum(np.square(frame_pred_transformed - frame_gt), axis=1))
        )

    return mpjpe, np.mean(pampjpe)

