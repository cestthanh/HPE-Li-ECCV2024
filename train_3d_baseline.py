import argparse
import copy
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from tqdm import tqdm

from dataset_lib import make_dataloader, make_dataset
from model import OriginalHPE3D
from utils.eval_3d import compute_3d_metrics


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Phase B HPE-Li-3D baseline on MMFi."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "dataset_lib" / "config.yaml"),
        help="Path to the MMFi dataset config.",
    )
    parser.add_argument(
        "--dataset-root",
        default=os.getenv(
            "MMFI_DATASET_ROOT", str(PROJECT_ROOT / "data" / "mmfi" / "dataset")
        ),
        help="MMFi dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv(
            "HPE_LI_3D_OUTPUT",
            str(PROJECT_ROOT / "results" / "hpe_li_3d_baseline"),
        ),
        help="Directory used for logs, metrics, and checkpoints.",
    )
    parser.add_argument(
        "--epochs", type=int, default=int(os.getenv("HPE_LI_3D_EPOCHS", "20"))
    )
    parser.add_argument(
        "--split-to-use",
        default=None,
        choices=[
            "random_split",
            "cross_scene_split",
            "cross_subject_split",
            "manual_split",
        ],
        help="Override config split_to_use without editing dataset_lib/config.yaml.",
    )
    parser.add_argument(
        "--lr", type=float, default=float(os.getenv("HPE_LI_3D_LR", "0.001"))
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=float(os.getenv("HPE_LI_3D_WEIGHT_DECAY", "0.0001")),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--loss",
        default="smooth_l1",
        choices=["smooth_l1", "mse"],
        help="Pose regression loss.",
    )
    parser.add_argument("--smooth-l1-beta", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop if val_mpjpe_mm has no meaningful improvement for this many epochs.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1.0,
        help="Minimum val_mpjpe_mm improvement to reset early stopping patience.",
    )
    parser.add_argument(
        "--eval-max-batches",
        type=int,
        default=None,
        help="Limit validation/test batches for quick smoke tests.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limit train batches for quick smoke tests.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Override config train_loader.batch_size.",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=None,
        help="Override config val_loader.batch_size.",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=None,
        help="Override config test_loader.batch_size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override num_workers for train/val/test loaders.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name. Defaults to timestamp plus split/model.",
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip final test evaluation. Useful for short smoke tests.",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, "r") as fd:
        return yaml.load(fd, Loader=yaml.FullLoader)


def apply_loader_overrides(config, args):
    config = copy.deepcopy(config)
    if args.split_to_use is not None:
        config["split_to_use"] = args.split_to_use

    for key in ("train_loader", "val_loader", "test_loader"):
        config[key] = dict(config[key])

    if args.train_batch_size is not None:
        config["train_loader"]["batch_size"] = args.train_batch_size
    if args.val_batch_size is not None:
        config["val_loader"]["batch_size"] = args.val_batch_size
    if args.test_batch_size is not None:
        config["test_loader"]["batch_size"] = args.test_batch_size

    if args.num_workers is not None:
        for key in ("train_loader", "val_loader", "test_loader"):
            config[key]["num_workers"] = args.num_workers

    return config


def make_run_dir(output_dir, run_name, split_to_use):
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}_original_hpe_3d_{split_to_use}"
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


def make_loaders(dataset_root, config, seed):
    train_dataset, eval_dataset = make_dataset(dataset_root, config)
    generator = torch.Generator().manual_seed(seed)

    train_loader = make_dataloader(
        train_dataset,
        is_training=True,
        generator=generator,
        **config["train_loader"],
    )

    val_indices, test_indices = train_test_split(
        list(range(len(eval_dataset))), test_size=0.5, random_state=41
    )
    val_dataset = Subset(eval_dataset, val_indices)
    test_dataset = Subset(eval_dataset, test_indices)

    val_loader = make_dataloader(
        val_dataset,
        is_training=False,
        generator=generator,
        **config["val_loader"],
    )
    test_loader = make_dataloader(
        test_dataset,
        is_training=False,
        generator=generator,
        **config["test_loader"],
    )
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def make_criterion(args):
    if args.loss == "smooth_l1":
        return torch.nn.SmoothL1Loss(beta=args.smooth_l1_beta)
    return torch.nn.MSELoss()


def batch_to_device(batch, device):
    csi_data = batch["input_wifi-csi"].to(device).float()
    gt_pose = batch["output"][:, :, 0:3].to(device).float()
    return csi_data, gt_pose


def evaluate(model, loader, criterion, device, max_batches=None, desc="eval"):
    model.eval()
    losses = []
    pred_chunks = []
    gt_chunks = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
            if max_batches is not None and batch_idx >= max_batches:
                break

            csi_data, gt_pose = batch_to_device(batch, device)
            pred_pose, _ = model(csi_data)
            loss = criterion(pred_pose, gt_pose)
            losses.append(float(loss.item()))

            pred_chunks.append(pred_pose.detach().cpu().numpy())
            gt_chunks.append(gt_pose.detach().cpu().numpy())

    if not losses:
        raise RuntimeError(f"No batches were evaluated for {desc}.")

    pred_all = np.concatenate(pred_chunks, axis=0)
    gt_all = np.concatenate(gt_chunks, axis=0)
    metrics = compute_3d_metrics(pred_all, gt_all)
    metrics["loss"] = float(np.mean(losses))
    metrics["num_samples"] = int(pred_all.shape[0])
    metrics["num_batches"] = int(len(losses))
    return metrics


def train_one_epoch(model, loader, criterion, optimizer, device, args, epoch):
    model.train()
    losses = []

    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for batch_idx, batch in enumerate(progress):
        if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
            break

        csi_data, gt_pose = batch_to_device(batch, device)
        pred_pose, _ = model(csi_data)
        loss = criterion(pred_pose, gt_pose)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip is not None and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()

        loss_value = float(loss.item())
        losses.append(loss_value)
        if batch_idx % args.log_interval == 0:
            print(
                f"epoch={epoch}, batch={batch_idx}, "
                f"loss={loss_value:.6f}, lr={optimizer.param_groups[0]['lr']:.6f}",
                flush=True,
            )
        progress.set_postfix(loss=f"{loss_value:.4f}")

    return {"loss": float(np.mean(losses)), "num_batches": int(len(losses))}


def save_json(path, payload):
    with open(path, "w") as fd:
        json.dump(payload, fd, indent=2)


def save_checkpoint(path, model, optimizer, epoch, config, args, metrics):
    torch.save(
        {
            "model_name": "OriginalHPE3D",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    config = load_config(args.config)
    config = apply_loader_overrides(config, args)
    run_dir = make_run_dir(args.output_dir, args.run_name, config["split_to_use"])

    with open(run_dir / "config.yaml", "w") as fd:
        yaml.safe_dump(config, fd, sort_keys=False)
    save_json(run_dir / "args.json", vars(args))

    print(f"run_dir={run_dir}", flush=True)
    print(f"dataset_root={args.dataset_root}", flush=True)
    print(f"split_to_use={config['split_to_use']}", flush=True)
    print(f"device={device}", flush=True)

    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = make_loaders(
        args.dataset_root, config, args.seed
    )
    print(
        f"train_samples={len(train_dataset)}, val_samples={len(val_dataset)}, "
        f"test_samples={len(test_dataset)}",
        flush=True,
    )
    print(
        f"train_batches={len(train_loader)}, val_batches={len(val_loader)}, "
        f"test_batches={len(test_loader)}",
        flush=True,
    )

    model = OriginalHPE3D().to(device)
    criterion = make_criterion(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_mpjpe = float("inf")
    best_epoch = None
    epochs_without_meaningful_improvement = 0
    stopped_early = False
    history = []
    best_checkpoint_path = run_dir / "checkpoints" / "best.pt"
    last_checkpoint_path = run_dir / "checkpoints" / "last.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args, epoch
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            max_batches=args.eval_max_batches,
            desc=f"val epoch {epoch}",
        )

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_record)
        save_json(run_dir / "history.json", history)

        print(
            "epoch=%d train_loss=%.6f val_loss=%.6f "
            "val_mpjpe=%.3f val_pa_mpjpe=%.3f "
            "val_pck50=%.3f val_pck100=%.3f pa_invalid=%d"
            % (
                epoch,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["mpjpe_mm"],
                val_metrics["pa_mpjpe_mm"],
                val_metrics["pck_50mm"],
                val_metrics["pck_100mm"],
                val_metrics["pa_mpjpe_invalid_count"],
            ),
            flush=True,
        )

        save_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            epoch,
            config,
            args,
            {"train": train_metrics, "val": val_metrics},
        )

        current_val_mpjpe = val_metrics["mpjpe_mm"]
        meaningful_improvement = (
            current_val_mpjpe < best_val_mpjpe - args.early_stopping_min_delta
        )

        if current_val_mpjpe < best_val_mpjpe:
            previous_best = best_val_mpjpe
            best_val_mpjpe = current_val_mpjpe
            best_epoch = epoch
            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                epoch,
                config,
                args,
                {"train": train_metrics, "val": val_metrics},
            )
            print(
                f"saved best checkpoint at epoch={epoch} "
                f"val_mpjpe={best_val_mpjpe:.3f}",
                flush=True,
            )
            if meaningful_improvement or previous_best == float("inf"):
                epochs_without_meaningful_improvement = 0
            else:
                epochs_without_meaningful_improvement += 1
        else:
            epochs_without_meaningful_improvement += 1

        if (
            args.early_stopping_patience is not None
            and args.early_stopping_patience > 0
        ):
            if epochs_without_meaningful_improvement >= args.early_stopping_patience:
                print(
                    "early stopping triggered at epoch=%d "
                    "best_epoch=%s best_val_mpjpe=%.3f "
                    "epochs_without_meaningful_improvement=%d"
                    % (
                        epoch,
                        best_epoch,
                        best_val_mpjpe,
                        epochs_without_meaningful_improvement,
                    ),
                    flush=True,
                )
                stopped_early = True
                break

    final_payload = {
        "best_epoch": best_epoch,
        "best_val_mpjpe_mm": best_val_mpjpe,
        "stopped_early": stopped_early,
        "history": history,
    }

    if not args.no_test:
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            max_batches=args.eval_max_batches,
            desc="test best",
        )
        final_payload["test"] = test_metrics
        print(
            "test_loss=%.6f test_mpjpe=%.3f test_pa_mpjpe=%.3f "
            "test_pck50=%.3f test_pck100=%.3f pa_invalid=%d"
            % (
                test_metrics["loss"],
                test_metrics["mpjpe_mm"],
                test_metrics["pa_mpjpe_mm"],
                test_metrics["pck_50mm"],
                test_metrics["pck_100mm"],
                test_metrics["pa_mpjpe_invalid_count"],
            ),
            flush=True,
        )

    save_json(run_dir / "final_metrics.json", final_payload)
    print(f"finished run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
