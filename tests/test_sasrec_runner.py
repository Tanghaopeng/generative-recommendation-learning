"""Protect temporal boundaries, negative exclusions, and full-catalog ranks."""
import sys
from pathlib import Path
import unittest
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_sasrec import OverfitMonitor, checkpoint, initialize_model, metrics_from_ranks, restore_model, restore_training_state, sample_negatives, target_ranks, training_arrays, training_keys


class ProtocolTests(unittest.TestCase):
    def test_overfit_requires_sustained_validation_degradation(self):
        monitor = OverfitMonitor(0.04, 0.45, patience=2)
        self.assertFalse(monitor.observe(0.039, 0.44)["pause"])
        warning = monitor.observe(0.038, 0.43)
        self.assertTrue(warning["pause"])
        self.assertEqual(warning["consecutive_warnings"], 2)
        self.assertFalse(monitor.observe(0.041, 0.42)["pause"])
        self.assertEqual(monitor.best_metric, 0.041)
        self.assertEqual(monitor.streak, 0)

    def test_loss_drop_or_plateau_alone_is_not_overfitting(self):
        monitor = OverfitMonitor(0.04, 0.45, patience=2)
        self.assertFalse(monitor.observe(0.04, 0.44)["pause"])
        self.assertFalse(monitor.observe(0.039, 0.46)["pause"])
        self.assertFalse(monitor.observe(0.039, 0.43)["pause"])
        self.assertFalse(monitor.observe(0.04, 0.42)["pause"])
        self.assertEqual(monitor.streak, 0)

    def test_checkpoint_restores_adam_and_random_streams(self):
        config = dict(device="cpu", maxlen=4, hidden_units=8, num_blocks=2,
                      num_heads=2, dropout_rate=0.2, norm_first=False)
        torch.manual_seed(2026)
        model = initialize_model(1, 8, config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        rng = np.random.default_rng(2026)
        seq = np.array([[1, 2, 3, 4]])
        def step(current_model, current_optimizer):
            current_model.train()
            current_optimizer.zero_grad()
            loss = current_model.log2feats(seq).square().sum()
            loss.backward()
            current_optimizer.step()
        step(model, optimizer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pth"
            checkpoint(path, model, config, 1, 8, 1, None, "test", optimizer, rng)
            expected_random = rng.integers(1000, size=10)
            step(model, optimizer)
            restored, saved = restore_model(path)
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.001)
            self.assertTrue(restore_training_state(saved, restored_optimizer, rng))
            np.testing.assert_array_equal(rng.integers(1000, size=10), expected_random)
            step(restored, restored_optimizer)
            for expected, actual in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_shift_and_truncation(self):
        users, seq, pos = training_arrays({1: [1, 2, 3, 4, 5], 2: [6, 7, 8]}, 3)
        np.testing.assert_array_equal(seq, [[2, 3, 4], [0, 6, 7]])
        np.testing.assert_array_equal(pos, [[3, 4, 5], [0, 7, 8]])
        train = {1: [1, 2, 3, 4, 5], 2: [6, 7, 8]}
        neg = sample_negatives(users, pos, 8, training_keys(train, 8), np.random.default_rng(2026))
        for row, uid in enumerate(users):
            self.assertFalse(set(neg[row][pos[row] != 0]) & set(train[int(uid)]))
        self.assertEqual(neg[1, 0], 0)

    def test_rank_history_filter_and_ties(self):
        scores = torch.tensor([[99., 8., 8., 10., 8., 7.]])
        ranks = target_ranks(scores, [4], [[3]])
        self.assertEqual(ranks.tolist(), [3])  # items 1 and 2 precede target 4
        self.assertEqual(metrics_from_ranks(np.array([1, 11]))["Recall@10"], 0.5)
        with self.assertRaises(ValueError):
            target_ranks(torch.ones(1, 6), [4], [[4]])

    def test_future_input_cannot_change_earlier_features(self):
        torch.manual_seed(2026)
        config = dict(device="cpu", maxlen=4, hidden_units=8, num_blocks=2,
                      num_heads=2, dropout_rate=0., norm_first=False)
        model = initialize_model(1, 8, config).eval()
        with torch.inference_mode():
            original = model.log2feats(np.array([[1, 2, 3, 4]]))
            changed = model.log2feats(np.array([[1, 2, 3, 8]]))
        torch.testing.assert_close(original[:, :3], changed[:, :3], rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
