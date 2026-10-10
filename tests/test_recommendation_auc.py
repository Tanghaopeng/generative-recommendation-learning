"""Metric direction, ties, cross-user score offsets and fixed candidate sampling."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from recommendation_auc import binary_auc, candidate_ids, grouped_auc


class AUCTests(unittest.TestCase):
    def test_perfect_reversed_and_tied(self):
        self.assertEqual(binary_auc([1, 0, 0], [3, 2, 1]), 1)
        self.assertEqual(binary_auc([1, 0, 0], [0, 2, 1]), 0)
        self.assertEqual(binary_auc([1, 0, 0], [1, 1, 1]), .5)

    def test_partial_ties_and_multiple_positives(self):
        self.assertEqual(binary_auc([1, 0, 0], [1, 1, 0]), .75)
        self.assertEqual(binary_auc([1, 1, 0, 0], [4, 2, 3, 1]), .75)

    def test_pooled_auc_differs_from_gauc(self):
        result = grouped_auc([{'labels': [1, 0], 'scores': [2, 1]},
                              {'labels': [1, 0], 'scores': [102, 101]}])
        self.assertEqual(result['GAUC'], 1)
        self.assertEqual(result['AUC'], .75)

    def test_gauc_weights_and_single_class_groups(self):
        result = grouped_auc([{'labels': [1, 0], 'scores': [2, 1]},
                              {'labels': [1, 0, 0, 0], 'scores': [0, 1, 2, 3]},
                              {'labels': [0], 'scores': [0]}])
        self.assertAlmostEqual(result['GAUC'], 1 / 3)
        self.assertEqual(result['skipped_single_class_groups'], 1)

    def test_reject_malformed(self):
        for labels, scores in [([1], [1]), ([1, 0], [1, float('nan')]), ([1, 2], [1, 2])]:
            with self.assertRaises(ValueError):
                binary_auc(labels, scores)

    def test_sampling_reproducible_unique_and_observed_excluded(self):
        row = {'user_id': 17, 'train': [1, 2, 3], 'valid': [4], 'test': [5]}
        for split in ('valid', 'test'):
            candidates = candidate_ids(row, 150, split, 100, 2026)
            self.assertEqual(candidates, candidate_ids(row, 150, split, 100, 2026))
            self.assertEqual(candidates[0], row[split][0])
            self.assertEqual(len(set(candidates)), 101)
            observed = row['train'] + (row['valid'] if split == 'test' else [])
            self.assertFalse(set(candidates) & set(observed))
        with self.assertRaises(ValueError):
            candidate_ids(row, 5, 'test', 100)


if __name__ == '__main__':
    unittest.main()
