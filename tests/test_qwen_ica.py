"""Real tiny-Qwen checks: answer isolation, causal generation, gradients and reload."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from test_qwen_sft import fixture, model_for
from qwen_sft import CompletionCollator, encode_example
from qwen_eval_fast import prepare_request, retrieve_group
from qwen_sft import CatalogConstraint
from qwen_ica import QwenICAForCausalLM, load_ica_tokenizer


SETTINGS = dict(enabled=True, attention_dim=8, heads=2, bottleneck=8)


class ICATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make(self, directory):
        tok, catalog, rows = fixture()
        base = model_for(tok).eval()
        base.save_pretrained(directory)
        tok.save_pretrained(directory)
        model = QwenICAForCausalLM.from_sft(directory, tok, 8, SETTINGS).eval()
        return tok, catalog, base, model

    def test_identity_logits_beams_and_source_weights(self):
        with tempfile.TemporaryDirectory() as d:
            tok, cat, base, model = self.make(d)
            batch = CompletionCollator(tok.pad_token_id)([encode_example(tok, cat, [1, 2], 3)])
            with torch.no_grad():
                torch.testing.assert_close(base(**batch).logits, model(**batch).logits, rtol=0, atol=0)
            for name, value in base.state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
            requests = [prepare_request(tok, cat, [1, 2]), prepare_request(tok, cat, [3, 4])]
            index = CatalogConstraint(tok, cat, 0, [])
            self.assertEqual(retrieve_group(base, tok, requests, index), retrieve_group(model, tok, requests, index))

    def test_only_history_is_pooled_never_teacher_forced_answer(self):
        with tempfile.TemporaryDirectory() as d:
            tok, cat, _, model = self.make(d)
            with torch.no_grad():
                model.model.ica.mlp_out.weight.normal_(std=0.1)
            samples = [encode_example(tok, cat, [1, 2], target) for target in (3, 17)]
            batch = CompletionCollator(tok.pad_token_id)(samples)
            ids, mask = batch['input_ids'], batch['attention_mask']
            groups = model.model.history_groups(ids, mask)
            self.assertEqual(len(groups), 4)  # two historical products per sample, NOT three
            answer = int((batch['labels'][0] != -100).nonzero()[0])
            enhanced = model.model.contextual_embeddings(ids, mask)
            original = model.get_input_embeddings()(ids)
            torch.testing.assert_close(enhanced[0, :answer], enhanced[1, :answer], rtol=0, atol=0)
            torch.testing.assert_close(enhanced[:, answer:], original[:, answer:], rtol=0, atol=0)
            changed = (enhanced - original).abs().sum(-1) > 0
            self.assertEqual(int(changed.sum()), 16)  # only 4*2*2 history token positions
            logits = model(**batch).logits
            torch.testing.assert_close(logits[0, :answer], logits[1, :answer], rtol=0, atol=0)

    def test_padding_and_malformed_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            tok, cat, _, model = self.make(d)
            batch = CompletionCollator(tok.pad_token_id)([
                encode_example(tok, cat, [1], 2), encode_example(tok, cat, [3, 4, 5], 6)])
            self.assertEqual(len(model.model.history_groups(batch['input_ids'], batch['attention_mask'])), 4)
            no_boundary = torch.tensor([tok.encode(''.join(f'<sid_{i}_0>' for i in range(4)), add_special_tokens=False) * 3])
            with self.assertRaises(ValueError):
                model(input_ids=no_boundary)
            with self.assertRaises(ValueError):
                model(inputs_embeds=model.get_input_embeddings()(batch['input_ids']))

    def test_nonzero_ica_cached_matches_full_forward_and_beam(self):
        with tempfile.TemporaryDirectory() as d:
            tok, cat, _, model = self.make(d)
            with torch.no_grad():
                model.model.ica.mlp_out.weight.normal_(std=0.1)
            ids = torch.tensor([prepare_request(tok, cat, [1, 2])['prompt']])
            cache = model(input_ids=ids, use_cache=True).past_key_values
            answer = tok.encode(''.join(f'<sid_{i}_0>' for i in range(4)), add_special_tokens=False)
            for token in answer:
                new = torch.tensor([[token]])
                ids = torch.cat([ids, new], 1)
                cached = model(input_ids=new, past_key_values=cache, use_cache=True)
                cache = cached.past_key_values
                full = model(input_ids=ids, use_cache=False)
                torch.testing.assert_close(cached.logits[:, -1], full.logits[:, -1], rtol=1e-5, atol=1e-6)
            request = [prepare_request(tok, cat, [1, 2])]
            values = retrieve_group(model, tok, request, CatalogConstraint(tok, cat, 0, []))
            self.assertEqual(len(values[0]), 20)
            self.assertTrue(set(values[0]).issubset(set(map(int, cat)) - {1, 2}))

    def test_gradients_and_complete_package_reload(self):
        with tempfile.TemporaryDirectory() as d:
            tok, cat, base, model = self.make(d)
            model.train()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            batch = CompletionCollator(tok.pad_token_id)([encode_example(tok, cat, [1, 2], 3), encode_example(tok, cat, [4], 5)])
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            before = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
            for _ in range(3):
                optimizer.zero_grad()
                model(**batch, use_cache=False).loss.backward()
                optimizer.step()
            for name, p in model.model.ica.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
                self.assertGreater(float(p.grad.abs().sum()), 0, name)
            self.assertFalse(torch.equal(before, model.model.layers[0].self_attn.q_proj.weight))
            model.eval()
            expected = model(**batch, use_cache=False).logits.detach()
            target = Path(d) / 'selected'
            model.save_pretrained(target)
            tok.save_pretrained(target)
            loaded = AutoModelForCausalLM.from_pretrained(target, local_files_only=True).eval()
            self.assertIsInstance(loaded, QwenICAForCausalLM)
            torch.testing.assert_close(expected, loaded(**batch, use_cache=False).logits, rtol=0, atol=0)
            self.assertEqual(tok.get_vocab(), load_ica_tokenizer(target).get_vocab())

    def test_trainer_optimizer_resume_preserves_ica(self):
        with tempfile.TemporaryDirectory() as d:
            tok, cat, _, model = self.make(Path(d) / 'source')
            data = [encode_example(tok, cat, [1, 2], 3), encode_example(tok, cat, [4], 5)] * 2
            def trainer(model, output, steps):
                args = TrainingArguments(output_dir=str(output), per_device_train_batch_size=2,
                    max_steps=steps, save_steps=1, report_to=[], use_cpu=True, learning_rate=0.01,
                    lr_scheduler_type='constant', remove_unused_columns=False, disable_tqdm=True)
                return Trainer(model=model, args=args, train_dataset=data, data_collator=CompletionCollator(tok.pad_token_id))
            initial = copy.deepcopy(model)
            uninterrupted = copy.deepcopy(model)
            first = trainer(model, Path(d) / 'run', 1)
            first.train()
            checkpoint = Path(d) / 'run/checkpoint-1'
            self.assertTrue((checkpoint / 'optimizer.pt').is_file())
            second = trainer(initial, Path(d) / 'run', 2)
            second.train(resume_from_checkpoint=str(checkpoint))
            reference = trainer(uninterrupted, Path(d) / 'reference', 2)
            reference.train()
            self.assertEqual(second.state.global_step, 2)
            self.assertGreater(float(second.model.model.ica.mlp_out.weight.abs().sum()), 0)
            for key, value in reference.model.state_dict().items():
                torch.testing.assert_close(value, second.model.state_dict()[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
