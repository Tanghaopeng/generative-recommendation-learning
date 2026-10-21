import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from qwen_eval_runtime import (BATCHED_EXECUTION, contract_digest, validate_eval_migration,
                               project_validation_state)


class EvaluationMigrationTests(unittest.TestCase):
    def setUp(self):
        self.old = dict(config={'micro_batch_size': 16}, prepared_manifest_sha256='data',
                        initialization={'seed': 2026}, code_sha256={
                            'qwen_sft.py': 'core', 'train_qwen_sft.py': 'old'})
        self.new = copy.deepcopy(self.old)
        self.new['evaluation_execution'] = dict(BATCHED_EXECUTION)
        self.new['code_sha256'].update({'train_qwen_sft.py': 'new', 'qwen_eval_fast.py': 'fast',
                                       'qwen_eval_runtime.py': 'runtime'})
        self.record = dict(source_contract_sha256=contract_digest(self.old),
                           target_code_sha256=self.new['code_sha256'])

    def test_only_evaluation_migration_is_allowed(self):
        validate_eval_migration(self.old, self.new, self.record)
        changed = copy.deepcopy(self.new)
        changed['config']['micro_batch_size'] = 8
        with self.assertRaisesRegex(ValueError, 'Training configuration'):
            validate_eval_migration(self.old, changed, self.record)
        changed = copy.deepcopy(self.new)
        changed['code_sha256']['qwen_sft.py'] = 'changed'
        record = dict(self.record, target_code_sha256=changed['code_sha256'])
        with self.assertRaisesRegex(ValueError, 'Core training'):
            validate_eval_migration(self.old, changed, record)

    def test_validation_state_keeps_progress_and_excludes_old_scores(self):
        state = dict(global_step=2500, epoch=.8, best_metric=.99, best_model_checkpoint='old',
                     log_history=[dict(step=2000, eval_ndcg10=.99, eval_recall10=.99, epoch=.6),
                                  dict(step=2500, loss=1.2)])
        def report(ndcg):
            return dict(split='valid', users=5000, beam=20, max_length=512,
                        prepared_manifest_sha256='data', evaluation_execution=BATCHED_EXECUTION,
                        **{'NDCG@10': ndcg, 'Recall@10': .2}, shortfall_users=0, seconds=1)
        reports = [dict(step=2000, result=report(.1)), dict(step=2500, result=report(.11))]
        result = project_validation_state(state, reports, '/new', 'data')
        self.assertEqual(result['global_step'], 2500)
        self.assertEqual(result['best_metric'], .11)
        self.assertEqual(result['best_global_step'], 2500)
        self.assertTrue(result['best_model_checkpoint'].endswith('checkpoint-2500'))
        self.assertEqual([r['eval_ndcg10'] for r in result['log_history'] if 'eval_ndcg10' in r], [.1, .11])
        self.assertEqual(state['best_metric'], .99)
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            project_validation_state(state, reports, '/new', 'wrong_data')

    def test_frozen_selection_rejects_mixed_execution_metrics(self):
        from freeze_qwen_training_selection import freeze_training_selection
        from qwen_sft import sha256, write_json
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run_dir, data = root / 'run', root / 'data'
            (run_dir / 'selected').mkdir(parents=True)
            best = run_dir / 'checkpoint-2000'
            best.mkdir()
            data.mkdir()
            (data / 'manifest.json').write_text('{}')
            (best / 'model.safetensors').write_bytes(b'chosen weights')
            (run_dir / 'selected/model.safetensors').write_bytes(b'chosen weights')
            scripts = Path(__file__).resolve().parents[1] / 'scripts'
            contract = dict(config=dict(pilot=False, validation_users=5000, beam=20, max_length=512),
                prepared_manifest_sha256=sha256(data / 'manifest.json'), evaluation_execution=BATCHED_EXECUTION,
                code_sha256={n: sha256(scripts / n) for n in ('qwen_sft.py', 'qwen_eval_fast.py', 'qwen_eval_runtime.py')})
            run = dict(contract, status='sft_completed_requires_full_validation_before_test',
                       test_evaluated=False, best_checkpoint=str(best), validation_users=5000)
            state = dict(best_model_checkpoint=str(best), best_metric=.1, log_history=[dict(step=2000,
                eval_ndcg10=.1, eval_recall10=.2, eval_execution_backend=BATCHED_EXECUTION['backend'],
                eval_execution_batch_size=8)])
            write_json(run_dir / 'contract.json', contract)
            write_json(run_dir / 'run.json', run)
            write_json(run_dir / 'trainer_state.json', state)
            self.assertEqual(freeze_training_selection(run_dir, data)['evaluation_execution'], BATCHED_EXECUTION)
            state['log_history'].append(dict(step=2500, eval_ndcg10=.09, eval_recall10=.1))
            write_json(run_dir / 'trainer_state.json', state)
            with self.assertRaisesRegex(ValueError, 'Mixed single-user'):
                freeze_training_selection(run_dir, data)


if __name__ == '__main__':
    unittest.main()
