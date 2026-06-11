import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_lib.splits import split_eval_dataset_by_sequence  # noqa: E402
from train_phase_c_dsknet3d import (  # noqa: E402
    denormalize_pose,
    make_criterion,
    make_training_metadata,
    normalize_pose,
)
from tools.evaluate_phase_c_dsknet3d_checkpoint import (  # noqa: E402
    get_eval_partition_unit,
)
from tools.diagnose_phase_c_action_loader import compute_gt_summary  # noqa: E402
from utils.eval_3d import (  # noqa: E402
    MMFI_BODY_SCALE_JOINT_NAMES,
    MMFI_BODY_SCALE_JOINTS,
    compute_3d_metrics,
    graphpose_body_scale_mmfi,
    graphpose_pck_mmfi,
    mpjpe_mm,
)


class PhaseCDemo2Tests(unittest.TestCase):
    def test_p1s1_config_uses_official_random_split_ratio(self):
        config_path = (
            PROJECT_ROOT / "dataset_lib" / "config_phase_c_demo_2_p1s1.yaml"
        )
        with open(config_path, "r") as fd:
            config = yaml.safe_load(fd)

        self.assertEqual(config["protocol"], "protocol1")
        self.assertEqual(config["split_to_use"], "random_split")
        self.assertEqual(config["random_split"]["ratio"], 0.8)
        self.assertEqual(config["random_split"]["random_seed"], 0)

    def test_g_pck_uses_gt_rhip_to_lshoulder_scale(self):
        gt = np.zeros((1, 17, 3), dtype=np.float64)
        gt[0, 1] = [0.0, 0.0, 0.0]
        gt[0, 11] = [2.0, 0.0, 0.0]

        # Make the legacy (5, 12) distance intentionally different.
        gt[0, 5] = [0.0, 0.0, 0.0]
        gt[0, 12] = [100.0, 0.0, 0.0]

        pred = gt + np.array([0.75, 0.0, 0.0])
        scale = graphpose_body_scale_mmfi(gt)

        self.assertEqual(MMFI_BODY_SCALE_JOINTS, (1, 11))
        self.assertEqual(MMFI_BODY_SCALE_JOINT_NAMES, ("R.Hip", "L.Shoulder"))
        np.testing.assert_allclose(scale, [2.0])
        self.assertEqual(graphpose_pck_mmfi(pred, gt, threshold=0.5), 100.0)
        self.assertEqual(graphpose_pck_mmfi(pred, gt, threshold=0.25), 0.0)

        metrics = compute_3d_metrics(pred, gt)
        self.assertEqual(metrics["g_PCK_scale_joints"], [1, 11])
        self.assertEqual(
            metrics["g_PCK_scale_joint_names"], ["R.Hip", "L.Shoulder"]
        )
        self.assertEqual(metrics["g_PCK_scale_source"], "ground_truth")

    def test_eval_partition_keeps_sequences_disjoint_and_stratifies_actions(self):
        class FakeFrameDataset:
            def __init__(self):
                self.data_list = []
                for action in ("A01", "A02"):
                    for sequence_index in range(4):
                        gt_path = f"{action}/sequence_{sequence_index}/ground_truth.npy"
                        for frame_index in range(3):
                            self.data_list.append(
                                {
                                    "action": action,
                                    "gt_path": gt_path,
                                    "idx": frame_index,
                                }
                            )

            def __len__(self):
                return len(self.data_list)

            def __getitem__(self, index):
                return self.data_list[index]

        dataset = FakeFrameDataset()
        val_dataset, test_dataset, metadata = split_eval_dataset_by_sequence(
            dataset, test_size=0.5, random_state=41
        )

        val_paths = {
            dataset.data_list[index]["gt_path"] for index in val_dataset.indices
        }
        test_paths = {
            dataset.data_list[index]["gt_path"] for index in test_dataset.indices
        }
        val_actions = {
            dataset.data_list[index]["action"] for index in val_dataset.indices
        }
        test_actions = {
            dataset.data_list[index]["action"] for index in test_dataset.indices
        }

        self.assertFalse(val_paths & test_paths)
        self.assertEqual(val_actions, {"A01", "A02"})
        self.assertEqual(test_actions, {"A01", "A02"})
        self.assertEqual(metadata["split_unit"], "sequence")
        self.assertEqual(metadata["stratify_key"], "action")
        self.assertEqual(metadata["val_num_sequences"], 4)
        self.assertEqual(metadata["test_num_sequences"], 4)
        self.assertEqual(metadata["sequence_overlap_count"], 0)

    def test_evaluator_auto_mode_preserves_legacy_checkpoint_partition(self):
        legacy_checkpoint = {"metrics": {}}
        new_checkpoint = {
            "eval_split_metadata": {
                "split_unit": "sequence",
            }
        }

        self.assertEqual(
            get_eval_partition_unit(legacy_checkpoint, "auto"), "frame"
        )
        self.assertEqual(
            get_eval_partition_unit(new_checkpoint, "auto"), "sequence"
        )
        self.assertEqual(
            get_eval_partition_unit(legacy_checkpoint, "sequence"), "sequence"
        )

    def test_gt_summary_does_not_report_flattened_temporal_std(self):
        static_pose = np.zeros((4, 17, 3), dtype=np.float64)
        static_pose[:, :, 2] = 3.0

        summary = compute_gt_summary(static_pose)
        short_summary = compute_gt_summary(static_pose[:1])

        self.assertNotIn("gt_temporal_std_mm", summary)
        self.assertNotIn("gt_temporal_std_mm", short_summary)
        self.assertEqual(summary["gt_mean_step_mm"], 0.0)
        self.assertEqual(summary["oracle_sequence_mean_mpjpe_mm"], 0.0)

    def test_normalized_mse_is_denormalized_before_mpjpe(self):
        args = SimpleNamespace(loss="mse", normalize_pose=True)
        criterion = make_criterion(args)
        metadata = make_training_metadata(args)
        pose_stats = {
            "mean": torch.tensor([1.0, 2.0, 3.0]).view(1, 1, 3),
            "std": torch.tensor([0.5, 2.0, 4.0]).view(1, 1, 3),
        }
        gt_meters = torch.tensor(
            [[[1.5, 4.0, 7.0], [0.5, 0.0, -1.0]]], dtype=torch.float32
        )

        gt_normalized = normalize_pose(gt_meters, pose_stats)
        normalized_delta = torch.tensor([1.0, -0.5, 0.25]).view(1, 1, 3)
        pred_normalized = gt_normalized + normalized_delta
        loss = criterion(pred_normalized, gt_normalized)
        pred_meters = denormalize_pose(pred_normalized, pose_stats)

        self.assertIsInstance(criterion, torch.nn.MSELoss)
        self.assertAlmostEqual(loss.item(), 0.4375)
        torch.testing.assert_close(
            pred_meters,
            gt_meters + torch.tensor([0.5, -1.0, 1.0]).view(1, 1, 3),
        )
        self.assertAlmostEqual(mpjpe_mm(pred_meters, gt_meters), 1500.0)
        self.assertEqual(metadata["pose_target_space"], "normalized_xyz")
        self.assertEqual(metadata["checkpoint_selection_metric"], "val_mpjpe_mm")
        self.assertEqual(metadata["checkpoint_selection_mode"], "min")


if __name__ == "__main__":
    unittest.main()
