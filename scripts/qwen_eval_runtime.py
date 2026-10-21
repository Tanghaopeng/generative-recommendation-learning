"""Explicit evaluation execution settings and guarded checkpoint migration."""
import hashlib
import json
import math
from pathlib import Path

REFERENCE_EXECUTION = dict(backend='reference_v1', batch_size=1, precision='bf16')
BATCHED_EXECUTION = dict(backend='equal_length_bucket_v1', batch_size=8, precision='bf16')


def load_execution(path=None):
    if path is None:
        return dict(REFERENCE_EXECUTION)
    with open(path, encoding='utf-8') as stream:
        value = json.load(stream)
    if value not in (REFERENCE_EXECUTION, BATCHED_EXECUTION):
        raise ValueError('Unsupported evaluation execution settings')
    return value


def contract_digest(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_eval_migration(old, new, record):
    if record['source_contract_sha256'] != contract_digest(old):
        raise ValueError('Migration source contract fingerprint mismatch')
    if record['target_code_sha256'] != new['code_sha256']:
        raise ValueError('Migration target code fingerprint mismatch')
    if old.get('evaluation_execution', REFERENCE_EXECUTION) != REFERENCE_EXECUTION:
        raise ValueError('Migration requires the original single-user execution')
    if new['evaluation_execution'] != BATCHED_EXECUTION:
        raise ValueError('Migration target must be BF16 batch8')
    ignored = {'evaluation_execution', 'code_sha256'}
    if {k: v for k, v in old.items() if k not in ignored} != {k: v for k, v in new.items() if k not in ignored}:
        raise ValueError('Training configuration, data or initialization changed during evaluation migration')
    added = {'qwen_eval_fast.py', 'qwen_eval_runtime.py'}
    if set(new['code_sha256']) != set(old['code_sha256']) | added:
        raise ValueError('Unexpected source file change in evaluation migration')
    for name, checksum in old['code_sha256'].items():
        if name != 'train_qwen_sft.py' and new['code_sha256'][name] != checksum:
            raise ValueError('Core training/data/decoding source changed: ' + name)


def project_validation_state(state, reports, output, prepared_manifest_sha256):
    """Retain optimizer progress, replace selection metrics with batch8 reruns."""
    if not reports:
        raise ValueError('Batched candidate validation reports are required')
    projected = json.loads(json.dumps(state))
    epochs = {row['step']: row.get('epoch', 0) for row in state['log_history'] if 'eval_ndcg10' in row}
    projected['log_history'] = [
        {('legacy_single_' + key if key.startswith('eval_') else key): value for key, value in row.items()}
        for row in state['log_history']]
    steps = set()
    for entry in sorted(reports, key=lambda entry: entry['step']):
        step, report = entry['step'], entry['result']
        if (step in steps or not 0 < step <= state['global_step'] or report['split'] != 'valid'
            or report['users'] != 5000 or report['beam'] != 20 or report['max_length'] != 512
            or report['prepared_manifest_sha256'] != prepared_manifest_sha256
            or report.get('evaluation_execution') != BATCHED_EXECUTION):
            raise ValueError('Incompatible batched candidate report')
        if any(not math.isfinite(report[k]) or not 0 <= report[k] <= 1 for k in ('NDCG@10', 'Recall@10')):
            raise ValueError('Invalid candidate metrics')
        steps.add(step)
        projected['log_history'].append(dict(
            step=step, epoch=epochs.get(step, 0), eval_ndcg10=report['NDCG@10'],
            eval_recall10=report['Recall@10'], eval_shortfalls=report['shortfall_users'],
            eval_runtime=report['seconds'], eval_execution_backend=BATCHED_EXECUTION['backend'],
            eval_execution_batch_size=8))
    best = max(sorted(reports, key=lambda entry: entry['step']), key=lambda entry: entry['result']['NDCG@10'])
    projected.update(best_metric=best['result']['NDCG@10'], best_global_step=best['step'],
                     best_model_checkpoint=str(Path(output) / f"checkpoint-{best['step']}"))
    callback = projected.get('stateful_callbacks', {}).get('EarlyStoppingCallback', {})
    if isinstance(callback, dict):
        callback.get('attributes', {}).update(early_stopping_patience_counter=0)
    return projected
