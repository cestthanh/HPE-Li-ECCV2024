import os
from pathlib import Path

import h5py
import mat73
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

mean = (
    15.9144,
    15.9394,
    12.1088,
    27.6384,
    26.1122,
    21.0799,
    14.1105,
    13.8744,
    13.8895,
)

std = (
    9.8100,
    10.2362,
    8.0946,
    11.2562,
    12.9910,
    10.1495,
    8.0082,
    7.4262,
    9.5949,
)

transform = transforms.Compose([
    transforms.Normalize(mean, std)  # Chuẩn hóa dữ liệu
])


def load_wipose_mat_sample(file_path):
    data_mat = mat73.loadmat(str(file_path))
    csi = data_mat["CSI"]
    csi_amp = np.array(csi).transpose(3, 2, 1, 0).reshape((9, 30, 5))
    csi_data = torch.tensor(csi_amp, dtype=torch.float32)
    csi_data = transform(csi_data)

    keypoints = torch.tensor(
        np.array(data_mat["SkeletonPoints"]).reshape((3, 18)).T,
        dtype=torch.float32,
    )
    xy_keypoints = keypoints[:, :2] * 0.001
    confidence_score = keypoints[:, 2:3]
    keypoints = torch.cat([xy_keypoints, confidence_score], dim=1)
    return {"input_wifi-csi": csi_data, "output": keypoints}


def load_preprocessed_wipose(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class WiPoseDataset(Dataset):
    def __init__(self, root_dir, split="Train", preprocessed_root=None):
        self.root_dir = root_dir
        self.split = split
        self.preprocessed_root = preprocessed_root or os.getenv(
            "WIPOSE_PREPROCESSED_ROOT"
        )
        self.inputs = None
        self.outputs = None

        cache_path = None
        if self.preprocessed_root:
            cache_path = Path(self.preprocessed_root) / f"{split}.pt"

        if cache_path and cache_path.is_file():
            cache = load_preprocessed_wipose(cache_path)
            self.inputs = cache["input_wifi-csi"]
            self.outputs = cache["output"]
            self.file_list = list(range(self.inputs.shape[0]))
        else:
            self.file_list = sorted(os.listdir(os.path.join(root_dir, split)))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.inputs is not None:
            return {
                "input_wifi-csi": self.inputs[idx],
                "output": self.outputs[idx],
            }

        file_path = os.path.join(self.root_dir, self.split, self.file_list[idx])
        return load_wipose_mat_sample(file_path)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    dataset = WiPoseDataset("/home/jackson-devworks/Desktop/HPE/Wi-Pose")
    loader = DataLoader(dataset, batch_size=1000, shuffle=False)

    # Tính mean và std
    mean = torch.zeros(9)
    std = torch.zeros(9)
    for data in tqdm(loader):
        images = data["input_wifi-csi"]
        mean += images.mean(dim=[0, 2, 3])
        std += images.std(dim=[0, 2, 3])
    mean /= len(loader)
    std /= len(loader)

    print(f"Mean: {mean}")
    print(f"Std: {std}")
