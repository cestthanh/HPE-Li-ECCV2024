import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from wipose.wipose_dataset import load_wipose_mat_sample  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess WiPose .mat files into split-level .pt tensor caches."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="WiPose root containing Train/ and Test/.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where Train.pt and Test.pt will be written.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["Train", "Test"],
        help="WiPose splits to preprocess.",
    )
    return parser.parse_args()


def preprocess_split(dataset_root, output_root, split):
    split_dir = dataset_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    files = sorted(path for path in split_dir.iterdir() if path.is_file())
    if not files:
        raise FileNotFoundError(f"No files found in {split_dir}")

    inputs = []
    outputs = []
    for path in tqdm(files, desc=f"preprocess {split}"):
        sample = load_wipose_mat_sample(path)
        inputs.append(sample["input_wifi-csi"])
        outputs.append(sample["output"])

    cache = {
        "input_wifi-csi": torch.stack(inputs, dim=0).contiguous(),
        "output": torch.stack(outputs, dim=0).contiguous(),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / f"{split}.pt"
    torch.save(cache, out_path)

    meta = {
        "split": split,
        "num_samples": len(files),
        "input_shape": list(cache["input_wifi-csi"].shape),
        "output_shape": list(cache["output"].shape),
        "source": str(split_dir),
        "cache": str(out_path),
    }
    with open(output_root / f"{split}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"wrote {out_path} samples={meta['num_samples']} "
        f"input_shape={meta['input_shape']} output_shape={meta['output_shape']}",
        flush=True,
    )


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser()
    output_root = Path(args.output_root).expanduser()

    for split in args.splits:
        preprocess_split(dataset_root, output_root, split)


if __name__ == "__main__":
    main()
