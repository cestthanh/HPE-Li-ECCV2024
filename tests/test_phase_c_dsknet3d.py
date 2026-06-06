import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.dsknet_trans_mmfi_3d import (  # noqa: E402
    DSKNetTransMMFI3D,
    get_dsknet_trans_mmfi_3d_model_config,
)


class PhaseCDSKNetTransMMFI3DTests(unittest.TestCase):
    def test_forward_shape_matches_3d_pose(self):
        model = DSKNetTransMMFI3D().eval()
        sample = torch.rand(2, 3, 114, 10)

        with torch.no_grad():
            output, elapsed = model(sample)

        self.assertEqual(tuple(output.shape), (2, 17, 3))
        self.assertGreaterEqual(elapsed, 0.0)

    def test_feature_shape_before_regression_matches_author_port(self):
        model = DSKNetTransMMFI3D().eval()
        sample = torch.rand(2, 3, 114, 10)

        with torch.no_grad():
            features = model.forward_features(sample)

        self.assertEqual(tuple(features.shape), (2, 256, 14, 1))
        self.assertEqual(features.reshape(features.size(0), -1).shape[1], 3584)

    def test_checkpoint_config_round_trip(self):
        model = DSKNetTransMMFI3D().eval()
        checkpoint = {
            "model_name": "DSKNetTransMMFI3D",
            "model_config": model.get_model_config(),
            "model_state_dict": model.state_dict(),
        }
        model_config = get_dsknet_trans_mmfi_3d_model_config(checkpoint)
        restored = DSKNetTransMMFI3D(**model_config).eval()
        restored.load_state_dict(checkpoint["model_state_dict"])

        sample = torch.rand(2, 3, 114, 10)
        with torch.no_grad():
            expected, _ = model(sample)
            actual, _ = restored(sample)

        self.assertEqual(model_config, model.get_model_config())
        torch.testing.assert_close(actual, expected)

    def test_single_training_step_has_finite_loss_and_gradients(self):
        torch.manual_seed(0)
        model = DSKNetTransMMFI3D().train()
        sample = torch.rand(4, 3, 114, 10)
        target = torch.rand(4, 17, 3)
        criterion = torch.nn.SmoothL1Loss(beta=0.05)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        output, _ = model(sample)
        loss = criterion(output, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(loss).item())
        finite_gradients = [
            torch.isfinite(param.grad).all().item()
            for param in model.parameters()
            if param.grad is not None
        ]
        self.assertTrue(all(finite_gradients))


if __name__ == "__main__":
    unittest.main()
