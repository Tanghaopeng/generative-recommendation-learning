"""Freeze the checkpoint chosen during SFT on a fixed validation subset, before test."""
import argparse
import math
from pathlib import Path

from qwen_sft import SID_DECODING_PROTOCOL, read_json, write_json, sha256
from qwen_eval_runtime import REFERENCE_EXECUTION


def freeze_training_selection(run_dir, data):
    run = read_json(run_dir / 'run.json')
    state = read_json(run_dir / 'trainer_state.json')
    contract = read_json(run_dir / 'contract.json')
    if run['status'] != 'sft_completed_requires_full_validation_before_test':
        raise ValueError('Training must finish before selection is frozen')
    if run.get('test_evaluated') or run['config']['pilot']:
        raise ValueError('Requires a formal training run without test-based selection')
    for key, value in contract.items():
        if run.get(key) != value:
            raise ValueError('Training contract mismatch: ' + key)
    if sha256(data / 'manifest.json') != run['prepared_manifest_sha256']:
        raise ValueError('Prepared data differs from training')
    if run['code_sha256']['qwen_sft.py'] != sha256(Path(__file__).with_name('qwen_sft.py')):
        raise ValueError('Decoding implementation differs from training validation')
    best = Path(state['best_model_checkpoint'])
    if str(best) != run['best_checkpoint'] or best.parent.resolve() != run_dir.resolve():
        raise ValueError('Best checkpoint evidence mismatch')
    step = int(best.name.removeprefix('checkpoint-'))
    matching = [row for row in state['log_history']
                if row.get('step') == step and 'eval_ndcg10' in row]
    if not matching or not math.isclose(matching[0]['eval_ndcg10'], state['best_metric'], abs_tol=1e-12):
        raise ValueError('Missing best-checkpoint validation evidence')
    evaluations = [row['eval_ndcg10'] for row in state['log_history'] if 'eval_ndcg10' in row]
    execution = run.get('evaluation_execution', REFERENCE_EXECUTION)
    if execution != REFERENCE_EXECUTION:
        if any(row.get('eval_execution_backend') != execution['backend'] or
               row.get('eval_execution_batch_size') != execution['batch_size']
               for row in state['log_history'] if 'eval_ndcg10' in row):
            raise ValueError('Mixed single-user and batch validation metrics cannot select a model')
        for name in ('qwen_eval_fast.py', 'qwen_eval_runtime.py'):
            if run['code_sha256'][name] != sha256(Path(__file__).with_name(name)):
                raise ValueError('Batched evaluator implementation differs from validation')
    if not math.isclose(max(evaluations), state['best_metric'], abs_tol=1e-12):
        raise ValueError('Chosen checkpoint is not validation-best')
    weight_hash = sha256(run_dir / 'selected/model.safetensors')
    if weight_hash != sha256(best / 'model.safetensors'):
        raise ValueError('Exported weights differ from the validation-selected checkpoint')
    if run['validation_users'] != run['config']['validation_users'] or run['validation_users'] <= 0:
        raise ValueError('Invalid validation subset')
    return dict(checkpoint_weight_sha256=weight_hash,
                prepared_manifest_sha256=run['prepared_manifest_sha256'],
                beam=run['config']['beam'], max_length=run['config']['max_length'],
                decoding_protocol=SID_DECODING_PROTOCOL,
                evaluation_execution=execution,
                validation_completed=True, full_validation_completed=False,
                validation_scope='fixed_training_validation_subset',
                validation_users=run['validation_users'],
                selection_metric='training validation subset NDCG@10',
                selected_step=step, best_checkpoint=str(best),
                **{'NDCG@10': matching[0]['eval_ndcg10'], 'Recall@10': matching[0]['eval_recall10']},
                selection_evidence_sha256={name: sha256(run_dir / name) for name in
                                           ('run.json', 'contract.json', 'trainer_state.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve the existing frozen selection record')
    result = freeze_training_selection(args.run, args.data)
    write_json(args.output, result)
    print('FROZEN_TRAINING_SELECTION', result)


if __name__ == '__main__':
    main()
