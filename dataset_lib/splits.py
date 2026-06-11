from collections import Counter
from math import ceil

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


def split_eval_dataset_by_sequence(dataset, test_size=0.5, random_state=41):
    """Split a frame dataset without placing one sequence in both subsets."""
    if not hasattr(dataset, "data_list"):
        raise TypeError("Sequence-level split requires a dataset with data_list.")

    sequence_groups = {}
    for frame_index, item in enumerate(dataset.data_list):
        gt_path = item.get("gt_path")
        action = item.get("action")
        if gt_path is None or action is None:
            raise ValueError("Each dataset item must contain gt_path and action.")

        group = sequence_groups.setdefault(
            gt_path,
            {
                "action": action,
                "indices": [],
            },
        )
        if group["action"] != action:
            raise ValueError(f"Sequence {gt_path} contains inconsistent action labels.")
        group["indices"].append(frame_index)

    sequence_paths = sorted(sequence_groups)
    if len(sequence_paths) < 2:
        raise ValueError("At least two sequences are required for val/test splitting.")

    action_labels = [sequence_groups[path]["action"] for path in sequence_paths]
    action_counts = Counter(action_labels)
    num_test = ceil(len(sequence_paths) * test_size)
    num_val = len(sequence_paths) - num_test
    can_stratify = (
        all(count >= 2 for count in action_counts.values())
        and num_test >= len(action_counts)
        and num_val >= len(action_counts)
    )
    stratify = action_labels if can_stratify else None

    val_paths, test_paths = train_test_split(
        sequence_paths,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    val_indices = sorted(
        index
        for path in val_paths
        for index in sequence_groups[path]["indices"]
    )
    test_indices = sorted(
        index
        for path in test_paths
        for index in sequence_groups[path]["indices"]
    )

    metadata = {
        "split_unit": "sequence",
        "sequence_key": "gt_path",
        "stratify_key": "action" if stratify is not None else None,
        "test_size": float(test_size),
        "random_state": int(random_state),
        "eval_num_sequences": len(sequence_paths),
        "val_num_sequences": len(val_paths),
        "test_num_sequences": len(test_paths),
        "val_num_frames": len(val_indices),
        "test_num_frames": len(test_indices),
        "sequence_overlap_count": len(set(val_paths) & set(test_paths)),
    }
    return Subset(dataset, val_indices), Subset(dataset, test_indices), metadata
