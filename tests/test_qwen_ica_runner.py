"""Verify the actual compressed model backup and quota failure behavior."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from collections import namedtuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_qwen_ica_experiment import backup, verify_archive, fingerprint


class ICABackupTests(unittest.TestCase):
    def setup_package(self, base):
        root, project, cloud = [Path(base) / name for name in ('run', 'project', 'cloud')]
        (root / 'formal/selected').mkdir(parents=True)
        (root / 'formal/selected/model.safetensors').write_bytes(b'fixture weight' * 10)
        (root / 'checks').mkdir()
        (root / 'checks/test.json').write_text('{"Recall@10": 0.1}')
        (project / 'scripts').mkdir(parents=True)
        for name in ('qwen_ica.py', 'train_qwen_ica.py', 'evaluate_qwen_ica.py', 'run_qwen_ica_experiment.py',
                     'qwen_sft.py', 'qwen_eval_fast.py', 'qwen_hpr.py', 'qwen_hpr_decode.py', 'evaluate_qwen_hpr.py'):
            (project / 'scripts' / name).write_text('# fixture')
        (project / 'experiments').mkdir()
        (project / 'experiments/qwen_ica_full.json').write_text('{}')
        (project / 'requirements-qwen-sft.txt').write_text('torch\ntransformers\n')
        return root, project, cloud

    def test_verified_cloud_archive_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root, project, cloud = self.setup_package(directory)
            self.assertTrue(backup(root, project, cloud))
            self.assertEqual(fingerprint(root / 'ica_selected_and_results.tar.gz'),
                             fingerprint(cloud / 'ica_selected_and_results.tar.gz'))
            with self.assertRaises(ValueError):
                backup(root, project, cloud)

    def test_insufficient_cloud_space_keeps_local_model_and_cloud_originals(self):
        with tempfile.TemporaryDirectory() as directory:
            root, project, cloud = self.setup_package(directory)
            cloud.mkdir()
            original = cloud / 'previous_model.bin'
            original.write_bytes(b'preserve me')
            usage = namedtuple('usage', 'total used free')(1000, 999, 1)
            with patch('run_qwen_ica_experiment.shutil.disk_usage', return_value=usage):
                self.assertFalse(backup(root, project, cloud))
            self.assertEqual(original.read_bytes(), b'preserve me')
            self.assertTrue((root / 'ica_selected_and_results.tar.gz').exists())
            self.assertTrue((cloud / 'test.json').exists())
            self.assertFalse((cloud / 'ica_selected_and_results.tar.gz').exists())


if __name__ == '__main__':
    unittest.main()
