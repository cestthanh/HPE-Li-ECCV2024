from dataclasses import dataclass
from typing import Dict, Sequence, Tuple


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_joints: int
    pose_dims: int
    csi_shape: Tuple[int, int, int]
    joint_names: Tuple[str, ...]
    skeleton_edges: Tuple[Tuple[int, int], ...]
    pose_unit: str


MMFI_JOINT_NAMES = (
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
)

MMFI_SKELETON_EDGES = (
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
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
)

WIPOSE_JOINT_NAMES = (
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
)

WIPOSE_SKELETON_EDGES = (
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
)

DATASET_SPECS: Dict[str, DatasetSpec] = {
    "mmfi": DatasetSpec(
        name="mmfi",
        num_joints=17,
        pose_dims=3,
        csi_shape=(3, 114, 10),
        joint_names=MMFI_JOINT_NAMES,
        skeleton_edges=MMFI_SKELETON_EDGES,
        pose_unit="meter",
    ),
    "wipose": DatasetSpec(
        name="wipose",
        num_joints=18,
        pose_dims=3,
        csi_shape=(9, 30, 5),
        joint_names=WIPOSE_JOINT_NAMES,
        skeleton_edges=WIPOSE_SKELETON_EDGES,
        pose_unit="audit_required",
    ),
}


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    key = dataset_name.lower()
    if key not in DATASET_SPECS:
        choices = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unknown dataset_name={dataset_name!r}. Choices: {choices}")
    return DATASET_SPECS[key]


def joint_names_for(dataset_name: str) -> Sequence[str]:
    return get_dataset_spec(dataset_name).joint_names
