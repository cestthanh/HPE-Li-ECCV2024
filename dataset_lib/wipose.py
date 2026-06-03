import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


WIPOSE_CSI_MEAN = torch.tensor(
    (15.9144, 15.9394, 12.1088, 27.6384, 26.1122, 21.0799, 14.1105, 13.8744, 13.8895),
    dtype=torch.float32,
).view(9, 1, 1)

WIPOSE_CSI_STD = torch.tensor(
    (9.8100, 10.2362, 8.0946, 11.2562, 12.9910, 10.1495, 8.0082, 7.4262, 9.5949),
    dtype=torch.float32,
).view(9, 1, 1)


def _load_mat(path):
    try:
        import mat73
    except ImportError as exc:
        raise ImportError(
            "WiPose .mat loading requires mat73. Install it in the training "
            "environment or run preprocessing to create Train.pt/Test.pt caches."
        ) from exc
    return mat73.loadmat(str(path))


def _canonicalize_csi(csi):
    array = np.asarray(csi)
    if array.shape == (9, 30, 5):
        return array.astype(np.float32)
    if array.ndim == 4:
        return array.transpose(3, 2, 1, 0).reshape((9, 30, 5)).astype(np.float32)
    raise ValueError(f"Unsupported WiPose CSI shape: {array.shape}")


def _canonicalize_pose(skeleton_points, num_joints):
    array = np.asarray(skeleton_points, dtype=np.float32)
    if array.shape == (num_joints, 3):
        return array
    if array.shape == (3, num_joints):
        return array.T
    if array.size == num_joints * 3:
        return array.reshape((3, num_joints)).T
    raise ValueError(
        f"Unsupported WiPose SkeletonPoints shape: {array.shape}; "
        f"expected ({num_joints}, 3) or (3, {num_joints})."
    )


def load_wipose_mat_sample(
    file_path,
    num_joints=18,
    pose_scale=1.0,
    normalize_csi=True,
):
    data_mat = _load_mat(file_path)
    csi_data = torch.tensor(_canonicalize_csi(data_mat["CSI"]), dtype=torch.float32)
    if normalize_csi:
        csi_data = (csi_data - WIPOSE_CSI_MEAN) / WIPOSE_CSI_STD

    pose = _canonicalize_pose(data_mat["SkeletonPoints"], num_joints)
    pose = torch.tensor(pose * float(pose_scale), dtype=torch.float32)
    return {
        "input_wifi-csi": csi_data,
        "output": pose,
        "file_path": str(file_path),
    }


def _load_preprocessed(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class WiPoseDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="Train",
        preprocessed_root=None,
        num_joints=18,
        pose_scale=1.0,
        normalize_csi=True,
    ):
        self.root_dir = Path(root_dir).expanduser()
        self.split = split
        self.preprocessed_root = (
            Path(preprocessed_root).expanduser()
            if preprocessed_root
            else (
                Path(os.getenv("WIPOSE_PREPROCESSED_ROOT")).expanduser()
                if os.getenv("WIPOSE_PREPROCESSED_ROOT")
                else None
            )
        )
        self.num_joints = int(num_joints)
        self.pose_scale = float(pose_scale)
        self.normalize_csi = bool(normalize_csi)
        self.inputs = None
        self.outputs = None

        cache_path = (
            self.preprocessed_root / f"{split}.pt"
            if self.preprocessed_root is not None
            else None
        )
        if cache_path is not None and cache_path.is_file():
            cache = _load_preprocessed(cache_path)
            self.inputs = cache["input_wifi-csi"].float()
            self.outputs = cache["output"].float()
            self.file_list = cache.get(
                "file_paths",
                [str(idx) for idx in range(self.inputs.shape[0])],
            )
            return

        split_dir = self.root_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"WiPose split directory not found: {split_dir}")
        self.file_list = sorted(
            path for path in split_dir.iterdir() if path.suffix.lower() == ".mat"
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.inputs is not None:
            return {
                "input_wifi-csi": self.inputs[idx],
                "output": self.outputs[idx],
                "file_path": str(self.file_list[idx]),
            }
        return load_wipose_mat_sample(
            self.file_list[idx],
            num_joints=self.num_joints,
            pose_scale=self.pose_scale,
            normalize_csi=self.normalize_csi,
        )


def make_dataset(dataset_root, config):
    train_cfg = dict(config.get("train_dataset", {}))
    eval_cfg = dict(config.get("val_dataset", {}))
    common = {
        "preprocessed_root": config.get("preprocessed_root")
        or os.getenv("WIPOSE_PREPROCESSED_ROOT"),
        "num_joints": int(config.get("num_joints", 18)),
        "pose_scale": float(config.get("pose_scale", 1.0)),
        "normalize_csi": bool(config.get("normalize_csi", True)),
    }
    train_dataset = WiPoseDataset(
        dataset_root,
        split=train_cfg.get("split", "Train"),
        **common,
    )
    eval_dataset = WiPoseDataset(
        dataset_root,
        split=eval_cfg.get("split", "Test"),
        **common,
    )
    return train_dataset, eval_dataset


def make_dataloader(
    dataset,
    is_training,
    generator,
    batch_size,
    num_workers=0,
    pin_memory=False,
    **_,
):
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        drop_last=is_training,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **loader_kwargs,
    )
