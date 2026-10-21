"""Select HPR decoding settings from validation only; never read test labels."""
import argparse
from pathlib import Path

from qwen_sft import read_json, write_json, sha256


def choose_validation(reports):
    if not reports:
        raise ValueError('Validation reports required')
    comparison = None
    for report in reports:
        if report['split'] != 'valid' or report.get('diagnostic_head', True) or report['users'] < 1:
            raise ValueError('Requires retrieval validation of a non-pilot HPR head')
        if any(not 0 <= report[key] <= 1 for key in ('Recall@10', 'NDCG@10', 'Recall@20', 'NDCG@20')):
            raise ValueError('Invalid validation metrics')
        c = report['contract']
        comparable = {k: c[k] for k in ('protocol', 'checkpoint_weight_sha256', 'checkpoint_metadata_sha256', 'prepared_manifest_sha256',
                                        'beam', 'max_history', 'max_length', 'evaluation_execution', 'code_sha256')}
        comparable.update(users=report['users'], subset_seed=report['subset_seed'])
        if comparison is not None and comparison != comparable:
            raise ValueError('Compare the same validation users, frozen generator, SID data and execution')
        comparison = comparable
    return max(reports, key=lambda r: r['NDCG@10'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve the prior frozen choice')
    reports = [read_json(path) for path in args.reports]
    chosen = choose_validation(reports)
    record = dict(retrieval_validation_completed=True, contract=chosen['contract'],
        validation_users=chosen['users'], subset_seed=chosen['subset_seed'],
        selection_metric='validation NDCG@10', **{'NDCG@10': chosen['NDCG@10']},
        evidence_sha256={str(p): sha256(p) for p in args.reports})
    write_json(args.output, record)
    print('Frozen HPR choice:', chosen['contract']['strength'], flush=True)


if __name__ == '__main__':
    main()
