from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from sid_usage import codebook_usage, semantic_usage


class CodebookPerplexityTests(unittest.TestCase):
    def test_known_uniform_and_concentrated_distributions(self):
        for counts,expected in [([25,25,25,25],4),([50,50,0,0],2),([100,0,0,0],1)]:
            self.assertAlmostEqual(codebook_usage(np.array(counts))['effective_codes'],expected)
        r=codebook_usage(np.array([97,1,1,1]))
        self.assertEqual(r['active_codes'],4)
        self.assertAlmostEqual(r['effective_codes'],1.1825824177485913,places=12)

    def test_matches_one_hot_mean_formula(self):
        ids=np.random.default_rng(5).integers(0,8,size=57)
        p=np.eye(8)[ids].mean(0)
        oracle=np.exp(-np.sum(p*np.log(p+1e-10)))
        actual=codebook_usage(np.bincount(ids,minlength=8))
        self.assertAlmostEqual(actual['effective_codes'],oracle,places=7)
        self.assertEqual(actual['codebook_perplexity'],actual['effective_codes'])

    def test_permutation_and_count_scale_invariance(self):
        c=np.array([1,8,0,3,2]); first=codebook_usage(c)
        for other in (c[::-1],c*5):
            self.assertAlmostEqual(first['effective_codes'],codebook_usage(other)['effective_codes'])

    def test_three_layers_and_invalid_data(self):
        codes=np.array([[0,0,0],[1,0,1],[0,1,0],[1,1,1]])
        rows=semantic_usage(codes,4)
        self.assertEqual([r['effective_codes'] for r in rows],[2,2,2])
        for bad in (np.zeros((4,4),dtype=int),np.full((4,3),-1),np.full((4,3),4)):
            with self.assertRaises(ValueError):
                semantic_usage(bad,4)
        for bad in ([0,0],[-1,2],[1.5,2]):
            with self.assertRaises(ValueError):
                codebook_usage(np.asarray(bad))


if __name__=='__main__':
    unittest.main()
