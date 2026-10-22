"""Prove joint gradient flow, prompt isolation, full checkpoint and generation."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from transformers import AutoModelForCausalLM, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from test_qwen_sft import fixture, model_for
from qwen_sft import CompletionCollator, encode_example, CatalogConstraint
from qwen_eval_fast import prepare_request, predict_histories
from qwen_hpr_decode import predict_hpr
from qwen_ica_hpr_joint import QwenICAHPRForCausalLM
from train_qwen_ica_hpr_joint import JointTrainer

ICA = dict(enabled=True, attention_dim=8, heads=2, bottleneck=8)
JOINT = dict(loss_weight=0.05, projection_dim=4, catalog_negatives=3, negative_seed=2026)


class JointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make(self, directory):
        tok, catalog, _ = fixture()
        base = model_for(tok).eval()
        base.save_pretrained(directory)
        tok.save_pretrained(directory)
        model = QwenICAHPRForCausalLM.from_sft(directory, tok, 8, ICA, JOINT)
        model.set_catalog(catalog)
        samples = [encode_example(tok, catalog, [1, 2], 3), encode_example(tok, catalog, [4], 5)]
        batch = CompletionCollator(tok.pad_token_id)(samples)
        return tok, catalog, base, model, batch

    def test_hpr_only_gradient_reaches_qwen_ica_and_sid_embeddings(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, _, model, batch = self.make(d)
            model.train()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            output = model(**batch, use_cache=False)
            output.hpr_loss.backward()
            for module in (model.model.ica, model.model.layers[0], model.hpr, model.get_input_embeddings()):
                norm = sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
                self.assertGreater(norm, 0)
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
            torch.testing.assert_close(output.loss, output.ce_loss + 0.05 * output.hpr_loss)

    def test_ce_remains_original_at_identity_and_cannot_train_hpr_alone(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, base, model, batch = self.make(d)
            model.eval()
            output = model(**batch, use_cache=False)
            torch.testing.assert_close(output.ce_loss, base(**batch, use_cache=False).loss, rtol=0, atol=0)
            output.ce_loss.backward()
            self.assertTrue(all(p.grad is None for p in model.hpr.parameters()))

    def test_prompt_context_never_reads_answer_in_same_forward(self):
        with tempfile.TemporaryDirectory() as d:
            tok, catalog, _, model, _ = self.make(d)
            model.eval()
            with torch.no_grad():
                model.model.ica.mlp_out.weight.normal_(std=0.1)
            samples = [encode_example(tok, catalog, [1, 2], target) for target in (3, 17)]
            batch = CompletionCollator(tok.pad_token_id)(samples)
            output = model(**batch, use_cache=False)
            torch.testing.assert_close(output.prompt_context[0], output.prompt_context[1], rtol=0, atol=0)
            ids = torch.tensor([prepare_request(tok, catalog, [1, 2])['prompt']])
            expected = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, -1]
            torch.testing.assert_close(expected, output.prompt_context[0], rtol=1e-5, atol=1e-6)

    def test_complete_mixed_precision_reload_and_beam_paths(self):
        with tempfile.TemporaryDirectory() as d:
            tok, catalog, _, model, batch = self.make(Path(d) / 'base')
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            for _ in range(2):
                optimizer.zero_grad()
                model(**batch, use_cache=False).loss.backward()
                optimizer.step()
            # Match real package precision: BF16 Qwen/ICA and FP32 HPR.
            model.model.to(dtype=torch.bfloat16)
            model.lm_head.to(dtype=torch.bfloat16)
            model.eval()
            target = Path(d) / 'selected'
            model.save_pretrained(target)
            tok.save_pretrained(target)
            loaded = AutoModelForCausalLM.from_pretrained(target, local_files_only=True, dtype=torch.bfloat16).eval()
            self.assertIsInstance(loaded, QwenICAHPRForCausalLM)
            for name, parameter in model.hpr.named_parameters():
                value = dict(loaded.hpr.named_parameters())[name]
                self.assertEqual(value.dtype, torch.float32)
                torch.testing.assert_close(value, parameter, rtol=0, atol=0)
            histories = [[1, 2], [4]]
            before = predict_hpr(model, tok, catalog, histories, model.hpr, batch_size=1, pool_size=32)
            after = predict_hpr(loaded, tok, catalog, histories, loaded.hpr, batch_size=1, pool_size=32)
            self.assertEqual(before, after)
            zero = predict_hpr(loaded, tok, catalog, histories, loaded.hpr, batch_size=1, pool_size=32, strength=0)
            self.assertEqual(zero, predict_histories(loaded, tok, catalog, histories, batch_size=1))

    def test_trainer_joint_optimizer_resume_and_loss_accumulation_flag(self):
        with tempfile.TemporaryDirectory() as d:
            tok, catalog, _, model, _ = self.make(Path(d) / 'base')
            data = [encode_example(tok, catalog, [1, 2], 3), encode_example(tok, catalog, [4], 5)] * 2
            initial, reference = copy.deepcopy(model), copy.deepcopy(model)

            def trainer(model, output, steps):
                args = TrainingArguments(output_dir=str(output), max_steps=steps, per_device_train_batch_size=2,
                    learning_rate=0.001, lr_scheduler_type='constant', weight_decay=0.01, save_steps=1,
                    report_to=[], use_cpu=True, remove_unused_columns=False, disable_tqdm=True)
                return JointTrainer(model=model, args=args, train_dataset=data,
                    data_collator=CompletionCollator(tok.pad_token_id), hpr_learning_rate=0.01)

            first = trainer(model, Path(d) / 'run', 1)
            self.assertFalse(first.model_accepts_loss_kwargs)
            first.train()
            second = trainer(initial, Path(d) / 'run', 2)
            second.train(resume_from_checkpoint=str(Path(d) / 'run/checkpoint-1'))
            full = trainer(reference, Path(d) / 'reference', 2)
            full.train()
            for name, value in second.model.state_dict().items():
                torch.testing.assert_close(value, full.model.state_dict()[name], rtol=0, atol=0)
            self.assertEqual({g['initial_lr'] for g in second.optimizer.param_groups}, {0.001, 0.01})


if __name__ == '__main__':
    unittest.main()
