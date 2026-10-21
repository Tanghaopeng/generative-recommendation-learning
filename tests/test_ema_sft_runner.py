from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_ema_sft import copy_verified
from check_ema_sft_inputs import digest


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
