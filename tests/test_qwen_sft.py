"""No downloaded 1.5B model is loaded: real causal Qwen2 layers in a tiny fixture."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM, TrainingArguments, set_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from qwen_sft import (positions, validate_rows, extend_tokenizer, encode_example, CompletionCollator,
                      SFTDataset, ValidationDataset, RankingTrainer, CatalogConstraint, retrieve,
                      evaluate_retrieval, sid_text)
from qwen_sft import prompt_ids


def fixture():
    base = Tokenizer(WordLevel({'[UNK]': 0, '<|im_end|>': 1, '<|im_start|>': 2,
                                'system': 3, 'user': 4, 'assistant': 5, 'Title': 6}, unk_token='[UNK]'))
    base.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=base, unk_token='[UNK]', eos_token='<|im_end|>',
                                        additional_special_tokens=['<|im_start|>'])
    tokenizer.chat_template = "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    extend_tokenizer(tokenizer, 8)
    catalog = {str(i): {'sid': [i % 2, (i // 2) % 2, (i // 4) % 2, i // 8], 'title': f'Title {i}'} for i in range(1, 33)}
    rows = [{'train': [1, 2, 3, 4], 'valid': [5], 'test': [6]},
            {'train': [7, 8, 9], 'valid': [10], 'test': [11]}]
    return tokenizer, catalog, rows


def model_for(tokenizer):
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tokenizer), hidden_size=16,
                                      intermediate_size=32, num_hidden_layers=1,
                                      num_attention_heads=2, num_key_value_heads=1,
                                      max_position_embeddings=512, tie_word_embeddings=True,
                                      eos_token_id=tokenizer.eos_token_id,
                                      pad_token_id=tokenizer.pad_token_id,
                                      attention_dropout=0.0))


class QwenSFTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_positions_and_no_heldout_targets(self):
        tok, catalog, rows = fixture()
        validate_rows(rows, 32)
        values = positions(rows, 2)
        self.assertEqual(values.tolist(), [[0, 1, 2], [0, 1, 3], [1, 0, 1], [1, 0, 2]])
        from tiger_model import training_examples
        np.testing.assert_array_equal(values, training_examples(rows, 2))
        for user, start, end in values:
            self.assertNotIn(rows[user]['train'][end], rows[user]['valid'] + rows[user]['test'])
        leaked = [dict(rows[0], test=[1])]
        with self.assertRaises(ValueError):
            validate_rows(leaked, 32)

    def test_atomic_sid_mask_padding_and_causal_shift(self):
        tok, catalog, rows = fixture()
        sample = encode_example(tok, catalog, [1, 2], 3)
        answer = tok.encode(sid_text(catalog['3']['sid']), add_special_tokens=False) + [tok.eos_token_id]
        self.assertEqual(sample['labels'][-5:], answer)
        self.assertTrue(all(i == -100 for i in sample['labels'][:-5]))
        batch = CompletionCollator(tok.pad_token_id)([sample, encode_example(tok, catalog, [1], 2)])
        self.assertTrue(torch.all(batch['labels'][batch['attention_mask'] == 0] == -100))
        # Genuine EOS is supervised even though padding uses the same token ID.
        self.assertEqual(int(batch['labels'][0, len(sample['labels']) - 1]), tok.eos_token_id)
        model = model_for(tok)
        out = model(**batch)
        manual = F.cross_entropy(out.logits[:, :-1].reshape(-1, len(tok)), batch['labels'][:, 1:].reshape(-1))
        torch.testing.assert_close(out.loss, manual)
        before = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
        catalog_before = json.dumps(catalog, sort_keys=True)
        out.loss.backward()
        self.assertTrue(all(p.requires_grad and p.grad is not None for p in model.parameters()))
        target_id = answer[0]
        self.assertGreater(float(model.get_input_embeddings().weight.grad[target_id].abs().sum()), 0)
        torch.optim.AdamW(model.parameters(), lr=0.01).step()
        self.assertFalse(torch.equal(before, model.model.layers[0].self_attn.q_proj.weight))
        self.assertEqual(catalog_before, json.dumps(catalog, sort_keys=True))

    def test_four_task_recipe(self):
        tok, catalog, rows = fixture()
        examples = positions(rows)
        ds = SFTDataset(rows, catalog, examples, tok, recipe='q2')
        self.assertEqual(len(ds), 2 * len(examples) + 2 * len(catalog))
        for index in [0, len(examples), len(examples) + 1, len(ds) - 1]:
            sample = ds[index]
            self.assertGreater(sum(i != -100 for i in sample['labels']), 1)
            self.assertLessEqual(len(sample['input_ids']), 512)

    def test_selection_rejects_test_partial_and_mismatched_protocol(self):
        from freeze_qwen_selection import select_full_validation
        report = dict(split='valid', diagnostic_only=False, users=86713, beam=20, max_length=512,
                      prepared_manifest_sha256='fixed_data', checkpoint_weight_sha256='a', **{'NDCG@10': .1})
        better = dict(report, checkpoint_weight_sha256='b', **{'NDCG@10': .2})
        self.assertEqual(select_full_validation([report, better])['checkpoint_weight_sha256'], 'b')
        for invalid in [dict(better, split='test'), dict(better, users=64),
                        dict(better, beam=50), dict(better, prepared_manifest_sha256='different')]:
            with self.assertRaises(ValueError):
                select_full_validation([report, invalid])

    def test_constrained_decoding_matches_exhaustive_small_catalog(self):
        tok, catalog, rows = fixture()
        set_seed(2026)
        model = model_for(tok).eval()
        history = list(range(1, 25))  # Also excludes observed products outside the last-20 window.
        result = retrieve(model, tok, catalog, history, beam=40)
        self.assertEqual(set(result), set(range(25, 33)))
        self.assertEqual(len(result), len(set(result)))
        # A wide beam covers all branches here, so it must match exact conditional log-probability ranking.
        prompt = prompt_ids(tok, 'next_sid', history, 0, catalog)
        exact = []
        for item in range(25, 33):
            code = tok.encode(sid_text(catalog[str(item)]['sid']), add_special_tokens=False)
            with torch.inference_mode():
                logits = model(torch.tensor([prompt + code[:-1]])).logits[0, len(prompt)-1:]
                score = float(F.log_softmax(logits, -1).gather(1, torch.tensor(code)[:, None]).sum())
            exact.append((item, score))
        self.assertEqual(result, [item for item, _ in sorted(exact, key=lambda pair: (-pair[1], pair[0]))])
        index = CatalogConstraint(tok, catalog, 0, history)
        for code, item in index.leaves.items():
            if item in history:
                scores = index(torch.tensor([code[:3]]), torch.zeros(1, len(tok)))
                self.assertTrue(torch.isneginf(scores[0, code[-1]]))
        metrics = evaluate_retrieval(model, tok, catalog, rows, beam=40)
        self.assertEqual(metrics['users'], 2)
        self.assertFalse(model.training)
        self.assertTrue(0 <= metrics['NDCG@10'] <= 1)

    def test_trainer_full_optimizer_resume_and_ranking_selection(self):
        tok, catalog, rows = fixture()
        data = SFTDataset(rows, catalog, positions(rows), tok)
        valid = ValidationDataset(rows, catalog, tok)
        with tempfile.TemporaryDirectory() as directory:
            def trainer(output, callback=None):
                set_seed(2026)
                args = TrainingArguments(output_dir=str(output), use_cpu=True, max_steps=2,
                                         per_device_train_batch_size=1, gradient_accumulation_steps=2,
                                         learning_rate=0.01, save_strategy='steps', save_steps=1,
                                         eval_strategy='steps', eval_steps=1, logging_strategy='no',
                                         load_best_model_at_end=True, metric_for_best_model='ndcg10',
                                         greater_is_better=True, report_to=[], disable_tqdm=True,
                                         seed=2026, data_seed=2026, remove_unused_columns=False)
                return RankingTrainer(model=model_for(tok), args=args, train_dataset=data,
                                      eval_dataset=valid, data_collator=CompletionCollator(tok.pad_token_id),
                                      processing_class=tok, callbacks=callback,
                                      ranking_evaluator=lambda model: {'NDCG@10': 0.25, 'Recall@10': 0.5, 'shortfall_users': 0})
            first = trainer(Path(directory) / 'a')
            first.train()
            checkpoint = Path(directory) / 'a/checkpoint-1'
            for name in ['optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json']:
                self.assertTrue((checkpoint / name).is_file(), name)
            self.assertEqual(first.state.best_metric, 0.25)
            resumed = trainer(Path(directory) / 'b')
            resumed.train(resume_from_checkpoint=str(checkpoint))
            self.assertEqual(resumed.state.global_step, 2)
            # load_best_model_at_end uses checkpoint-1 for this equal-metric fixture.
            for name, value in first.model.state_dict().items():
                torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)
            # Compare the latest weights too: best-model reload alone would hide a broken resumed update.
            from safetensors.torch import load_file
            uninterrupted = load_file(str(Path(directory) / 'a/checkpoint-2/model.safetensors'))
            restarted = load_file(str(Path(directory) / 'b/checkpoint-2/model.safetensors'))
            for name, value in uninterrupted.items():
                torch.testing.assert_close(value, restarted[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
