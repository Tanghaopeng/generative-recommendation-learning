"""Check causal alignment and candidate batching without loading pretrained weights."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_qwen_auc import score_candidates


class PositionModel:
    device = torch.device('cpu')

    def __call__(self, input_ids, attention_mask, use_cache, logits_to_keep):
        positions = torch.arange(1, input_ids.size(1) + 1).float()
        logits = positions[None, :, None] * torch.arange(8).float()[None, None, :] / 10
        logits = logits.expand(input_ids.size(0), -1, -1)
        return SimpleNamespace(logits=logits[:, -logits_to_keep:, :])


class ScoringTests(unittest.TestCase):
    def test_causal_positions_and_batch_boundaries(self):
        prompt, codes = [1, 2, 3], [[4, 5, 6, 7], [7, 6, 5, 4], [1, 2, 3, 4]]
        actual = score_candidates(PositionModel(), prompt, codes, batch_size=2)
        positions = torch.arange(len(prompt), len(prompt) + 4).float()
        expected_logits = positions[:, None] * torch.arange(8).float()[None, :] / 10
        expected = [torch.log_softmax(expected_logits, -1).gather(1, torch.tensor(code)[:, None]).sum().item()
                    for code in codes]
        torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected))
        torch.testing.assert_close(torch.tensor(actual), torch.tensor(score_candidates(PositionModel(), prompt, codes, 1)))


if __name__ == '__main__':
    unittest.main()
