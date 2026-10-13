"""Validate frozen training-subset selection evidence without model weights."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from freeze_qwen_training_selection import freeze_training_selection
from qwen_sft import sha256, write_json


class TrainingSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'run'
        self.data = self.root / 'data'
        self.data.mkdir()
        write_json(self.data / 'manifest.json', {'frozen': True})
        self.best = self.run / 'checkpoint-3000'
        self.best.mkdir(parents=True)
        (self.best / 'model.safetensors').write_bytes(b'synthetic hash fixture')
        selected = self.run / 'selected'
        selected.mkdir()
        (selected / 'model.safetensors').write_bytes(b'synthetic hash fixture')
        config = {'pilot': False, 'validation_users': 5000, 'beam': 20, 'max_length': 512}
        contract = {'config': config, 'prepared_manifest_sha256': sha256(self.data / 'manifest.json'),
                    'code_sha256': {'qwen_sft.py': sha256(ROOT / 'scripts/qwen_sft.py')}}
        write_json(self.run / 'contract.json', contract)
        write_json(self.run / 'run.json', dict(contract,
            status='sft_completed_requires_full_validation_before_test',
            test_evaluated=False, best_checkpoint=str(self.best), validation_users=5000))
        write_json(self.run / 'trainer_state.json', {
            'best_model_checkpoint': str(self.best), 'best_metric': .1,
            'log_history': [{'step': 2500, 'eval_ndcg10': .09, 'eval_recall10': .11},
                            {'step': 3000, 'eval_ndcg10': .1, 'eval_recall10': .12}]})

    def test_fixed_subset_and_export_fingerprints(self):
        result = freeze_training_selection(self.run, self.data)
        self.assertTrue(result['validation_completed'])
        self.assertFalse(result['full_validation_completed'])
        self.assertEqual(result['validation_scope'], 'fixed_training_validation_subset')
        self.assertEqual(result['validation_users'], 5000)
        self.assertEqual(result['selected_step'], 3000)
        self.assertEqual(len(result['selection_evidence_sha256']), 3)

    def test_changed_export_rejected(self):
        (self.run / 'selected/model.safetensors').write_bytes(b'changed weights')
        with self.assertRaisesRegex(ValueError, 'Exported weights differ'):
            freeze_training_selection(self.run, self.data)

    def test_changed_data_rejected(self):
        write_json(self.data / 'manifest.json', {'frozen': False})
        with self.assertRaisesRegex(ValueError, 'Prepared data differs'):
            freeze_training_selection(self.run, self.data)

    def test_nonbest_checkpoint_rejected(self):
        path = self.run / 'trainer_state.json'
        state = json.loads(path.read_text())
        state['log_history'][0]['eval_ndcg10'] = .2
        write_json(path, state)
        with self.assertRaisesRegex(ValueError, 'not validation-best'):
            freeze_training_selection(self.run, self.data)


if __name__ == '__main__':
    unittest.main()
