import sys
from pathlib import Path
import unittest

import torch
from transformers import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from qwen_eval_fast import BulkCatalogConstraint, prepare_request, retrieve_group, predict_histories, evaluate_retrieval_fast
from qwen_sft import CatalogConstraint, retrieve, evaluate_retrieval
from test_qwen_sft import fixture, model_for


class FastEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_constraint_matches_reference_for_every_depth_and_user(self):
        tok, catalog, _ = fixture()
        index = CatalogConstraint(tok, catalog, 0, [])
        codes = list(index.leaves)
        histories = [[1, 2, 3], list(range(1, 33))]
        for depth in range(4):
            prefixes = [codes[i][:depth] for i in (0, 10, 20, 0, 10, 20)]
            ids = torch.tensor([[0, 0] + list(prefix) for prefix in prefixes])
            scores = torch.randn(6, len(tok))
            actual = BulkCatalogConstraint(index, 2, histories, 3)(ids, scores)
            expected = []
            for user in range(2):
                ref = CatalogConstraint(tok, catalog, 2, histories[user])
                expected.append(ref(ids[user * 3:(user + 1) * 3], scores[user * 3:(user + 1) * 3]))
            torch.testing.assert_close(actual, torch.cat(expected), rtol=0, atol=0)

    def test_single_and_batched_candidate_order_and_history_exclusion(self):
        tok, catalog, _ = fixture()
        set_seed(2026)
        model = model_for(tok).eval()
        histories = [[1, 2], [3, 4], list(range(1, 25)), [7, 8], list(range(1, 33))]
        expected = [retrieve(model, tok, catalog, history, beam=40) for history in histories]
        self.assertEqual(predict_histories(model, tok, catalog, histories, 1, beam=40), expected)
        self.assertEqual(predict_histories(model, tok, catalog, histories, 4, beam=40), expected)
        self.assertEqual(expected[-1], [])

    def test_rejects_mixed_lengths_and_bad_beam_mapping(self):
        tok, catalog, _ = fixture()
        model = model_for(tok).eval()
        index = CatalogConstraint(tok, catalog, 0, [])
        requests = [prepare_request(tok, catalog, history) for history in ([1], [1, 2])]
        with self.assertRaisesRegex(ValueError, 'equal-length'):
            retrieve_group(model, tok, requests, index)
        with self.assertRaisesRegex(ValueError, 'mapping'):
            BulkCatalogConstraint(index, 0, [[]], 20)(torch.zeros(1, 0, dtype=torch.long),
                                                     torch.zeros(1, len(tok)))

    def test_metrics_and_model_mode_match_original(self):
        tok, catalog, rows = fixture()
        set_seed(2026)
        model = model_for(tok).train()
        model.config.use_cache = False
        for split in ('valid', 'test'):
            expected = evaluate_retrieval(model, tok, catalog, rows, split, beam=40)
            actual = evaluate_retrieval_fast(model, tok, catalog, rows, split, beam=40)
            for key in ('Recall@10', 'NDCG@10', 'Recall@20', 'NDCG@20', 'shortfall_users', 'users'):
                self.assertEqual(actual[key], expected[key])
            self.assertTrue(model.training)
            self.assertFalse(model.config.use_cache)


if __name__ == '__main__':
    unittest.main()
