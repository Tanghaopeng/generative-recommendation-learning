from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_ema_sft import copy_verified
from check_ema_sft_inputs import digest, compare_training_config


class TrainingConfigTests(unittest.TestCase):
    def setUp(self):
        self.baseline = dict(micro_batch_size=2, gradient_accumulation_steps=64,
                             learning_rate=5e-5, epochs=1)

    def test_supported_batches_preserve_effective_batch(self):
        for batch, accum in [(4, 32), (8, 16), (16, 8)]:
            with self.subTest(batch=batch):
                candidate = dict(self.baseline, micro_batch_size=batch, gradient_accumulation_steps=accum)
                differences = compare_training_config(candidate, self.baseline)
                self.assertEqual(set(differences), {'micro_batch_size', 'gradient_accumulation_steps'})
        self.assertEqual(compare_training_config(self.baseline, self.baseline), {})

    def test_rejects_other_training_changes(self):
        with self.assertRaisesRegex(ValueError, 'outside batch'):
            compare_training_config(dict(self.baseline, learning_rate=1e-4), self.baseline)
        with self.assertRaisesRegex(ValueError, 'effective batch'):
            compare_training_config(dict(self.baseline, micro_batch_size=4), self.baseline)


class VerifiedCopyTests(unittest.TestCase):
    def test_copy_readback_and_repeat(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';target=root/'sub/target'
            source.write_bytes(b'persisted model bytes')
            checksum=digest(source)
            self.assertEqual(copy_verified(source,target,checksum),checksum)
            self.assertEqual(target.read_bytes(),source.read_bytes())
            self.assertEqual(copy_verified(source,target,checksum),checksum)

    def test_corrupt_existing_destination_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';target=root/'target'
            source.write_bytes(b'correct');target.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'existing different'):
                copy_verified(source,target)
            self.assertEqual(target.read_bytes(),b'corrupt')

    def test_wrong_source_checksum_never_commits_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';target=root/'target'
            source.write_bytes(b'corrupt transfer')
            with self.assertRaisesRegex(ValueError,'checksum failed'):
                copy_verified(source,target,'0'*64)
            self.assertFalse(target.exists())
            self.assertTrue((root/'target.partial').exists())


if __name__=='__main__':
    unittest.main()
