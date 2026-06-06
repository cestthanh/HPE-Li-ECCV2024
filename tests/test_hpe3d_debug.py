import unittest
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import OriginalHPE3D, get_hpe3d_model_config
from tools.demo_inference_3d import (
    compute_demo_diagnostics,
    load_csi_sequence,
    load_sequence_with_dataset_loader,
)


class HPE3DCheckpointCompatibilityTests(unittest.TestCase):
    def test_default_model_uses_corrected_sk_attention(self):
        model = OriginalHPE3D()
        config = model.get_model_config()

        self.assertEqual(config["sk_layout"], "stack")
        self.assertTrue(config["sk_use_min_l"])
        self.assertEqual(
            tuple(model.skunit1.conv2_sk[0].fc[0].weight.shape),
            (32, 64, 1),
        )
        self.assertEqual(
            tuple(model.skunit2.conv2_sk[0].fc[0].weight.shape),
            (32, 128, 1),
        )

        model.eval()
        with torch.no_grad():
            output, _ = model(torch.rand(1, 3, 114, 10))
        self.assertEqual(tuple(output.shape), (1, 17, 3))

    def test_legacy_checkpoint_config_is_inferred_and_reproduced(self):
        legacy_config = {
            "sk_m": 4,
            "sk_g": 1,
            "sk_r": 4,
            "sk_l": 32,
            "sk_use_min_l": False,
            "sk_layout": "legacy_view",
        }
        original = OriginalHPE3D(**legacy_config).eval()
        state_dict = original.state_dict()
        inferred = get_hpe3d_model_config({"model_state_dict": state_dict})
        restored = OriginalHPE3D(**inferred).eval()
        restored.load_state_dict(state_dict)

        self.assertEqual(inferred, legacy_config)
        sample = torch.rand(2, 3, 114, 10)
        with torch.no_grad():
            expected, _ = original(sample)
            actual, _ = restored(sample)
        torch.testing.assert_close(actual, expected)

    def test_new_checkpoint_model_config_round_trip(self):
        original = OriginalHPE3D().eval()
        checkpoint = {
            "model_config": original.get_model_config(),
            "model_state_dict": original.state_dict(),
        }
        model_config = get_hpe3d_model_config(checkpoint)
        restored = OriginalHPE3D(**model_config).eval()
        restored.load_state_dict(checkpoint["model_state_dict"])

        self.assertEqual(model_config, original.get_model_config())
        sample = torch.rand(1, 3, 114, 10)
        with torch.no_grad():
            expected, _ = original(sample)
            actual, _ = restored(sample)
        torch.testing.assert_close(actual, expected)


class HPE3DDiagnosticTests(unittest.TestCase):
    def test_constant_prediction_is_flagged_when_gt_moves_in_z(self):
        rng = np.random.default_rng(0)
        base_pose = rng.normal(size=(17, 3)).astype(np.float32) * 0.1
        gt = np.stack([base_pose.copy() for _ in range(5)], axis=0)
        gt[:, :, 2] += np.linspace(0.0, 0.4, len(gt), dtype=np.float32)[:, None]
        pred = np.broadcast_to(gt.mean(axis=0, keepdims=True), gt.shape).copy()

        diagnostics = compute_demo_diagnostics(pred, gt, output_dims=3)

        self.assertAlmostEqual(
            diagnostics["motion_ratios"]["axis_temporal_std_ratio"]["z"], 0.0
        )
        self.assertAlmostEqual(diagnostics["motion_ratios"]["mean_step_ratio"], 0.0)
        self.assertIsNone(
            diagnostics["articulation_motion_ratios"]["mean_step_ratio"]
        )
        self.assertAlmostEqual(
            diagnostics["mpjpe_gain_over_constant_sequence_mean_mm"], 0.0, places=5
        )
        warning_text = " ".join(diagnostics["warnings"])
        self.assertIn("constant sequence-mean pose", warning_text)
        self.assertIn("Predicted Z temporal variation", warning_text)


class HPE3DDemoLoaderTests(unittest.TestCase):
    def test_dataset_sequence_loader_matches_direct_csi_and_gt_loading(self):
        try:
            import scipy.io as scio
        except Exception as exc:
            self.skipTest(f"MMFi loader dependencies unavailable: {exc}")

        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            sequence_root = dataset_root / "E01" / "S05" / "A06"
            csi_root = sequence_root / "wifi-csi"
            csi_root.mkdir(parents=True)

            gt = rng.normal(size=(297, 17, 3)).astype(np.float32)
            np.save(sequence_root / "ground_truth.npy", gt)

            for frame_idx in range(297):
                frame = rng.normal(size=(3, 114, 10)).astype(np.float32)
                if frame_idx == 0:
                    frame[0, 0, 0] = np.nan
                    frame[1, 1, 1] = np.inf
                scio.savemat(
                    csi_root / f"frame{frame_idx + 1:03d}.mat",
                    {"CSIamp": frame},
                )

            args = SimpleNamespace(
                dataset_root=str(dataset_root),
                scene="E01",
                subject="S05",
                action="A06",
                max_frames=2,
            )

            dataset_csi, dataset_gt, loader_info = load_sequence_with_dataset_loader(args)
            direct_csi = load_csi_sequence(csi_root, normalize=True, max_frames=2)

            self.assertEqual(loader_info["sequence_loader"], "dataset")
            self.assertEqual(dataset_csi.shape, (2, 3, 114, 10))
            self.assertEqual(dataset_gt.shape, (2, 17, 3))
            np.testing.assert_allclose(dataset_csi, direct_csi, atol=1e-7)
            np.testing.assert_allclose(dataset_gt, gt[:2], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
