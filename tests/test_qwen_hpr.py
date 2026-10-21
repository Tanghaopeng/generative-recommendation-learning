"""CPU checks with real tiny Qwen layers; no 1.5B model or reported experiment gains."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from test_qwen_sft import fixture, model_for
from qwen_sft import CatalogConstraint, sha256, write_json
from qwen_eval_fast import prepare_request, predict_histories
from qwen_hpr import (HPRConfig, HPRScorer, HPR_PROTOCOL, PrefixNegativeSampler,
                      cumulative_paths, contrastive_loss, encode_context_batch, load_scorer,
                      verify_sft_identity)
from qwen_hpr_decode import select_expansion, predict_hpr
from train_qwen_hpr import fit_heads
from evaluate_qwen_hpr import retrieval_metrics, check_test_selection
from freeze_qwen_hpr_selection import choose_validation


class HPRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shared_true_prefixes_are_positives_not_negatives(self):
        torch.manual_seed(1)
        scorer = HPRScorer(HPRConfig(8, 4))
        table = torch.randn(3, 4, 8)
        contexts = torch.randn(4, 8)
        targets = torch.tensor([[1, 2, 3]] * 4)
        sampler = PrefixNegativeSampler(targets.numpy())
        loss, log = contrastive_loss(scorer, contexts, targets, table, sampler, 8, np.random.default_rng(1))
        self.assertAlmostEqual(float(loss.detach()), 0, places=6)
        self.assertEqual(log['layer1_distinct_prefixes'], 1)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in scorer.parameters()))

    def test_source_record_prevents_training_hpr_on_wrong_sid_model(self):
        record = dict(weight_sha256='gradient_sft', data_manifest_sha256='gradient_sid')
        verify_sft_identity(record, 'gradient_sft', 'gradient_sid')
        for weight, data in [('ema_sft', 'gradient_sid'), ('gradient_sft', 'ema_sid')]:
            with self.assertRaisesRegex(ValueError, 'SFT source record'):
                verify_sft_identity(record, weight, data)

    def test_prefix_negatives_are_valid_unique_and_exclude_true_prefix(self):
        catalog = np.array([[0, 1, 1], [0, 2, 1], [0, 3, 1], [1, 1, 2]])
        sampler = PrefixNegativeSampler(catalog)
        codes, mask = sampler.sample(catalog[:1], 1, 2, np.random.default_rng(1))
        self.assertTrue(mask.all())
        self.assertEqual(set(map(tuple, codes[0])), {(0, 2), (0, 3)})
        for layer in range(3):
            codes, mask = sampler.sample(catalog, layer, 8, np.random.default_rng(1))
            valid = set(map(tuple, catalog[:, :layer + 1]))
            for i, target in enumerate(catalog[:, :layer + 1]):
                chosen = list(map(tuple, codes[i][mask[i]]))
                self.assertEqual(len(chosen), len(set(chosen)))
                self.assertNotIn(tuple(target), chosen)
                self.assertTrue(set(chosen) <= valid)

    def test_path_uses_cumulative_qwen_embeddings_and_three_layers_only(self):
        table = torch.arange(3 * 4 * 8).reshape(3, 4, 8).float()
        code = torch.tensor([[1, 2, 3]])
        paths = cumulative_paths(code, table)
        torch.testing.assert_close(paths[0, 0], table[0, 1])
        torch.testing.assert_close(paths[0, 2], table[0, 1] + table[1, 2] + table[2, 3])
        with self.assertRaises(ValueError):
            cumulative_paths(torch.tensor([[1, 2, 3, 0]]), table)

    def test_context_is_prompt_end_and_padding_does_not_read_future_answers(self):
        tok, catalog, _ = fixture()
        torch.manual_seed(2)
        model = model_for(tok).eval().requires_grad_(False)
        requests = [prepare_request(tok, catalog, h) for h in ([1], [1, 2, 3])]
        actual = encode_context_batch(model, tok, requests)
        for i, request in enumerate(requests):
            ids = torch.tensor([request['prompt']])
            with torch.no_grad():
                expected = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, -1]
            torch.testing.assert_close(actual[i], expected, rtol=1e-5, atol=1e-6)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_path_reranking_can_rescue_before_beam_pruning(self):
        # Candidate 20 loses with beam=20; a larger scoring pool lets HPR rescue it.
        lm = -torch.arange(25, dtype=torch.float32) * .01
        path = torch.zeros(25)
        path[20] = 10
        kept, _ = select_expansion(lm, path, 20, 25, .25)
        self.assertIn(20, kept.tolist())
        narrow, _ = select_expansion(lm, path, 20, 20, .25)
        self.assertNotIn(20, narrow.tolist())
        baseline, _ = select_expansion(lm, path, 20, 25, 0)
        self.assertEqual(baseline.tolist(), list(range(20)))

    def test_zero_strength_matches_existing_sft_and_details_same_actual_batches(self):
        tok, catalog, _ = fixture()
        torch.manual_seed(3)
        model = model_for(tok).eval().requires_grad_(False)
        scorer = HPRScorer(HPRConfig(16, 4)).eval()
        histories = [[1], [3, 4], list(range(1, 33))]
        expected = predict_histories(model, tok, catalog, histories, batch_size=2, beam=40)
        details = {}
        actual = predict_hpr(model, tok, catalog, histories, scorer, 2, 40, 400, 0,
                             detail_sink=lambda user, entries: details.update({user: entries}))
        self.assertEqual(actual, expected)
        self.assertEqual([[x['item_id'] for x in details[i]] for i in range(3)], expected)

    def test_cached_beam_matches_exhaustive_full_paths_with_hpr(self):
        tok, catalog, _ = fixture()
        torch.manual_seed(4)
        model = model_for(tok).eval().requires_grad_(False)
        scorer = HPRScorer(HPRConfig(16, 4)).eval()
        histories = [[1, 2], [4, 5], list(range(1, 33))]
        actual = predict_hpr(model, tok, catalog, histories, scorer, 2, 40, 400, .25, return_details=True)
        index = CatalogConstraint(tok, catalog, 0, [])
        for user, history in enumerate(histories[:2]):
            prompt = prepare_request(tok, catalog, history)['prompt']
            context = encode_context_batch(model, tok, [dict(prompt=prompt)])
            expected = []
            for code, item in index.leaves.items():
                if item in history:
                    continue
                ids = torch.tensor([prompt + list(code)])
                with torch.no_grad():
                    logits = model(input_ids=ids).logits[0, len(prompt) - 1:len(prompt) + 3].float().log_softmax(-1)
                    lm = float(logits[torch.arange(4), torch.tensor(code)].sum())
                    path = model.get_input_embeddings()(torch.tensor(code[:3])).float().sum(0, keepdim=True)
                    score = lm + .25 * float(scorer.paired_scores(context, path, 2))
                expected.append((item, lm, score))
            expected.sort(key=lambda x: (-x[2], x[0]))
            self.assertEqual([x['item_id'] for x in actual[user]], [x[0] for x in expected[:20]])
            raw = {item: lm for item, lm, _ in expected}
            for entry in actual[user]:
                self.assertAlmostEqual(entry['lm_log_probability'], raw[entry['item_id']], places=4)
        self.assertEqual(actual[-1], [])
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_real_head_optimizer_saving_resume_and_qwen_stays_frozen(self):
        tok, catalog, _ = fixture()
        torch.manual_seed(5)
        model = model_for(tok).eval().requires_grad_(False)
        original = {k: v.clone() for k, v in model.state_dict().items()}
        requests = [prepare_request(tok, catalog, [i]) for i in range(1, 17)]
        contexts = encode_context_batch(model, tok, requests).numpy()
        targets = np.array([catalog[str(i + 1)]['sid'][:3] for i in range(1, 17)])
        token_ids = [[tok.encode(f'<sid_{layer}_{i}>', add_special_tokens=False)[0]
                      for i in range(8)] for layer in range(3)]
        table = model.get_input_embeddings().weight[torch.tensor(token_ids)].detach().numpy()
        with tempfile.TemporaryDirectory() as folder:
            cache, output = Path(folder) / 'cache', Path(folder) / 'heads'
            cache.mkdir()
            arrays = dict(train_contexts=contexts, train_targets=targets, valid_contexts=contexts[:8],
                          valid_targets=targets[:8], sid_embeddings=table,
                          catalog_sids=np.array([v['sid'][:3] for v in catalog.values()]))
            for name, value in arrays.items():
                np.save(cache / (name + '.npy'), value, allow_pickle=False)
            manifest = dict(status='completed', protocol=HPR_PROTOCOL, train_rows=16, valid_rows=8,
                            hidden_size=16, checkpoint_weight_sha256='toy', prepared_manifest_sha256='toy_data',
                            checkpoint_metadata_sha256={'config.json': 'toy'},
                            code_sha256={'qwen_eval_fast.py': sha256(ROOT / 'scripts/qwen_eval_fast.py')},
                            source_sft_code_sha256=sha256(ROOT / 'scripts/qwen_sft.py'),
                            max_history=20, max_length=512, diagnostic_only=True,
                            files={f.name: sha256(f) for f in cache.glob('*.npy')})
            write_json(cache / 'manifest.json', manifest)
            config = dict(seed=1, projection_dim=4, initial_temperature=.07, batch_size=8,
                          epochs=2, learning_rate=.01, weight_decay=0., max_grad_norm=1.,
                          catalog_negatives=4, early_stopping_patience=2)
            result = fit_heads(cache, output, config)
            self.assertFalse(result['qwen_loaded'])
            history = json.loads((output / 'history.json').read_text())
            self.assertGreater(history[0]['max_gradient_norm'], 0)
            self.assertTrue(all(np.isfinite(row['valid_contrastive_loss']) for row in history))
            self.assertLess(history[-1]['valid_contrastive_loss'], history[0]['valid_contrastive_loss'])
            head, metadata = load_scorer(output / 'selected')
            self.assertEqual(metadata['provenance']['checkpoint_weight_sha256'], 'toy')
            self.assertEqual(sum(p.numel() for p in head.parameters()), 3 * (2 * 16 * 4 + 1))
            again = fit_heads(cache, output, config, resume=True)
            self.assertEqual(again['completed_epochs'], result['completed_epochs'])
            with self.assertRaisesRegex(ValueError, 'contract'):
                fit_heads(cache, output, dict(config, learning_rate=.02), resume=True)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)

    def test_metrics_and_selection_guards(self):
        result = retrieval_metrics([1, 2], [[1], [3, 2]])
        self.assertEqual(result['Recall@10'], 1.)
        self.assertAlmostEqual(result['NDCG@10'], (1 + 1 / np.log2(3)) / 2)
        contract = dict(protocol=HPR_PROTOCOL, checkpoint_weight_sha256='sft',
                        checkpoint_metadata_sha256={'tokenizer.json': 'frozen'},
                        prepared_manifest_sha256='data', hpr_weight_sha256='head',
                        beam=20, max_history=20, max_length=512, evaluation_execution={'batch_size': 8},
                        code_sha256={'decode': 'fixed'}, strength=.25)
        report = dict(result, contract=contract, split='valid', diagnostic_head=False, subset_seed=2027)
        self.assertEqual(choose_validation([report]), report)
        with self.assertRaises(ValueError):
            choose_validation([dict(report, split='test')])
        altered = copy.deepcopy(report)
        altered['contract']['prepared_manifest_sha256'] = 'different_sid'
        with self.assertRaises(ValueError):
            choose_validation([report, altered])
        selection = dict(retrieval_validation_completed=True, contract=contract)
        check_test_selection(selection, contract)
        for wrong in (dict(contract, strength=.5), dict(contract, hpr_weight_sha256='different')):
            with self.assertRaises(ValueError):
                check_test_selection(selection, wrong)
        with self.assertRaises(ValueError):
            check_test_selection(selection, contract, diagnostic_head=True)


if __name__ == '__main__':
    unittest.main()
