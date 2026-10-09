"""Freeze the best FULL-validation report before test; no model or test data loaded."""
import argparse
from pathlib import Path

from qwen_sft import read_json, write_json, sha256


def select_full_validation(reports):
    if not reports:
        raise ValueError('At least one full validation report is required')
    protocol = None
    for report in reports:
        if report['split'] != 'valid' or report['diagnostic_only'] or report['users'] != 86713:
            raise ValueError('Select only from complete Office Products validation reports')
        contract = (report['prepared_manifest_sha256'], report['beam'], report['users'], report['max_length'])
        if protocol is not None and protocol != contract:
            raise ValueError('Reports use different data, beam or prompt length')
        protocol = contract
    return max(reports, key=lambda report: report['NDCG@10'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Selection record already exists; do not overwrite a frozen decision')
    reports = [read_json(path) for path in args.reports]
    # Stable tie policy: the first supplied report wins ties. Test results never enter this function.
    selected = select_full_validation(reports)
    record = {key: selected[key] for key in
              ('checkpoint_weight_sha256', 'prepared_manifest_sha256', 'beam', 'max_length', 'NDCG@10', 'Recall@10')}
    record.update(full_validation_completed=True, selection_metric='full validation NDCG@10',
                  validation_reports={str(path): sha256(path) for path in args.reports})
    write_json(args.output, record)
    print('Frozen checkpoint weight hash:', record['checkpoint_weight_sha256'])


if __name__ == '__main__':
    main()
