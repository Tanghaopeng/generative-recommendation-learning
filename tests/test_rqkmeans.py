import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_rqkmeans import assign, encode, fit_residual
from build_semantic_ids import disambiguate


class ResidualKMeansTests(unittest.TestCase):
    def test_assign_matches_explicit_distance(self):
        rng=np.random.default_rng(7)
        x=rng.normal(size=(17,5)); c=rng.normal(size=(4,5))
        expected=((x[:,None,:]-c[None,:,:])**2).sum(-1).argmin(1)
        np.testing.assert_array_equal(assign(x,c,batch_size=3),expected)

    def test_sequential_residual_not_independent_clustering(self):
        x=np.array([[5.,0.],[2.,0.]],dtype=np.float32)
        books=np.array([[[0.,0.],[4.,0.]],[[0.,0.],[1.,0.]],[[0.,0.],[.5,0.]]],dtype=np.float32)
        codes,residual,_=encode(x,books)
        np.testing.assert_array_equal(codes[0],[1,1,0])
        reconstructed=sum(books[d,codes[:,d]] for d in range(3))
        np.testing.assert_allclose(x,reconstructed+residual)
        full,_=disambiguate(np.vstack([codes,codes]))
        self.assertEqual(len(np.unique(full,axis=0)),4)

    def test_fit_repeatable_and_input_unchanged(self):
        x=np.random.default_rng(3).normal(size=(80,4)).astype(np.float32)
        original=x.copy()
        a,_=fit_residual(x,layers=3,size=4,seed=8,threads=1,n_init=1,max_iter=3,batch_size=16)
        b,_=fit_residual(x,layers=3,size=4,seed=8,threads=1,n_init=1,max_iter=3,batch_size=16)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(x,original)


if __name__=='__main__':
    unittest.main()
