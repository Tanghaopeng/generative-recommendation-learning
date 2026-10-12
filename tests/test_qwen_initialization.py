"""Exercise Q0 initialization/export/SFT loading on real tiny Qwen2 models only."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from qwen_initialization import check_initialization, initialize_sid_rows
from qwen_sft import sha256, write_json, positions


def base_fixture(tied=True):
    base = Tokenizer(WordLevel({'[UNK]': 0, '<|im_end|>': 1, '<|im_start|>': 2,
                                'system': 3, 'user': 4, 'assistant': 5, 'Title': 6}, unk_token='[UNK]'))
    base.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=base, unk_token='[UNK]',
                                        eos_token='<|im_end|>', additional_special_tokens=['<|im_start|>'])
    tokenizer.chat_template = "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    # Three pre-existing padded rows: exercise the real Qwen vocab/tokenizer mismatch.
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tokenizer)+3, hidden_size=16,
                                        intermediate_size=32, num_hidden_layers=1,
                                        num_attention_heads=2, num_key_value_heads=1,
                                        max_position_embeddings=512, tie_word_embeddings=tied,
                                        eos_token_id=1, attention_dropout=0.0)).to(torch.bfloat16)
    return model, tokenizer


def prepared_fixture(directory):
    directory.mkdir()
    rows = [{'user_id': 'tiny', 'train': [1, 2], 'valid': [3], 'test': [4]}]
    catalog = {str(i): {'sid': [i % 2, (i // 2) % 2, (i // 4) % 2, 0], 'title': f'Title {i}'}
               for i in range(1, 7)}
    (directory / 'histories.jsonl').write_text(json.dumps(rows[0])+'\n')
    write_json(directory / 'catalog.json', catalog)
    np.save(directory / 'train_examples.npy', positions(rows))
    write_json(directory / 'manifest.json', dict(status='completed', max_history=20, tokens_per_namespace=2,
               files={name: sha256(directory / name) for name in ('histories.jsonl', 'catalog.json', 'train_examples.npy')}))


class QwenInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_seed_reproducibility_and_preservation_of_original_parameters(self):
        for tied in (True, False):
            with self.subTest(tied=tied):
                original, tok = base_fixture(tied)
                a, b, c = copy.deepcopy(original), copy.deepcopy(original), copy.deepcopy(original)
                info = initialize_sid_rows(a, copy.deepcopy(tok), 2, 2026)
                torch.manual_seed(999)  # SID draws must be independent of unrelated global RNG use.
                initialize_sid_rows(b, copy.deepcopy(tok), 2, 2026)
                initialize_sid_rows(c, copy.deepcopy(tok), 2, 2027)
                self.assertEqual(info['reused_padding_rows'], 3)
                self.assertEqual(info['sid_rows_initialized'], 8)
                self.assertEqual(info['input_output_embeddings_tied'], tied)
                for name, before in original.state_dict().items():
                    after = a.state_dict()[name]
                    if name in ('model.embed_tokens.weight', 'lm_head.weight'):
                        torch.testing.assert_close(before[:len(tok)], after[:len(tok)], rtol=0, atol=0)
                        self.assertFalse(torch.equal(after[len(tok):], c.state_dict()[name][len(tok):]))
                        self.assertFalse(torch.equal(before[len(tok):], after[len(tok):len(tok)+3]))
                    else:
                        torch.testing.assert_close(before, after, rtol=0, atol=0)
                    torch.testing.assert_close(after, b.state_dict()[name], rtol=0, atol=0)

    def test_export_reload_and_sft_dry_run_share_the_same_q0(self):
        import export_qwen_q0
        import qwen_initialization
        import train_qwen_sft
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, tok = base_fixture()
            base.save_pretrained(root / 'base')
            tok.save_pretrained(root / 'base')
            # Qwen's tokenizer has these BPE source files; the tiny WordLevel
            # tokenizer instead loads tokenizer.json, but exercise metadata binding.
            write_json(root / 'base/vocab.json', tok.get_vocab())
            (root / 'base/merges.txt').write_text('# tiny tokenizer fixture\n')
            prepared_fixture(root / 'data')
            expected = sha256(root / 'base/model.safetensors')
            with patch.object(export_qwen_q0, 'MODEL_SHA256', expected), patch.object(qwen_initialization, 'MODEL_SHA256', expected):
                argv = ['export', '--model', str(root / 'base'), '--data', str(root / 'data'), '--output', str(root / 'q0')]
                with patch.object(sys, 'argv', argv):
                    export_qwen_q0.main()
                record = check_initialization(root / 'q0', root / 'data', 2026)
                self.assertEqual(record['optimizer_steps'], 0)
                loaded = Qwen2ForCausalLM.from_pretrained(root / 'q0', dtype=torch.bfloat16)
                reference = copy.deepcopy(base)
                initialize_sid_rows(reference, copy.deepcopy(tok), 2, 2026)
                for name, value in reference.state_dict().items():
                    torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)
                config = dict(seed=2026, max_history=20, recipe='q1', max_length=512,
                              train_limit=0, validation_users=0, pilot=True, max_steps=1)
                write_json(root / 'config.json', config)
                args = ['sft', '--config', str(root / 'config.json'), '--model', str(root / 'base'),
                        '--data', str(root / 'data'), '--initialization', str(root / 'q0'),
                        '--output', str(root / 'dry_run'), '--dry-run']
                with patch.object(sys, 'argv', args):
                    train_qwen_sft.main()
                dry = json.loads((root / 'dry_run/dry_run.json').read_text())
                self.assertEqual(dry['initialization']['checkpoint_weight_sha256'], record['checkpoint_weight_sha256'])
                self.assertFalse(dry['actual_gpu_training'])
                with self.assertRaisesRegex(ValueError, 'seed'):
                    check_initialization(root / 'q0', root / 'data', 2027)
                # Detect tokenizer corruption before using the otherwise unchanged weights.
                (root / 'q0/tokenizer_config.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    check_initialization(root / 'q0', root / 'data', 2026)

    def test_export_rejects_already_extended_tokenizer(self):
        model, tok = base_fixture()
        initialize_sid_rows(model, tok, 2, 2026)
        with self.assertRaisesRegex(ValueError, 'original tokenizer'):
            initialize_sid_rows(model, tok, 2, 2026)


if __name__ == '__main__':
    unittest.main()
