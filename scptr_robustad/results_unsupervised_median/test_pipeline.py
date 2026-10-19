"""Mathematical checks for the new hierarchy/median/imputation implementation."""
import itertools
import unittest
import numpy as np
from scipy import sparse
from hierarchy import adjacent_tree,median_em,neighbor_weights,diffusion_weights,rf_distances

def laminar(intervals):
    return all(not(a<c<b<d or c<a<d<b)
               for (a,b),(c,d) in itertools.combinations(intervals,2))

class PipelineTests(unittest.TestCase):
    def test_hierarchies_are_laminar(self):
        rng=np.random.default_rng(3)
        for n in (5,20,70):
            a=rng.random((n,n)); a+=a.T
            t=adjacent_tree(a,max_span=n)
            self.assertTrue(laminar(t))
            self.assertEqual(len(t),n-2) # root excluded
            self.assertTrue(all(0<=i<j<=n for i,j in t))

    def test_majority_is_exact_rf_median(self):
        features=[(0,2),(1,3),(2,4),(0,3),(1,4)]
        x=np.array([[1,0,1,0,0],[1,0,0,1,0],[0,1,0,0,1]],dtype=float)
        self.assertTrue(all(laminar([t for t,b in zip(features,row) if b]) for row in x))
        median=x.sum(axis=0)>len(x)/2
        self.assertTrue(laminar([t for t,b in zip(features,median) if b]))
        cost=np.abs(x-median).sum()
        for bits in itertools.product((0,1),repeat=len(features)):
            if laminar([t for t,b in zip(features,bits) if b]):
                self.assertLessEqual(cost,np.abs(x-np.array(bits)).sum())

    def test_em_monotonic_and_nearest_median(self):
        rng=np.random.default_rng(2)
        x=sparse.csr_matrix(rng.random((60,40))>0.5)
        y,m,h=median_em(x.astype(float),seed=3)
        self.assertTrue(np.all(np.diff(h)<=1e-8))
        d=rf_distances(x.astype(float),m)
        np.testing.assert_allclose(d[np.arange(len(y)),y],d.min(axis=1))
        for c in range(5):
            if (y==c).sum():
                np.testing.assert_array_equal(m[c],np.asarray(x[y==c].sum(axis=0)).ravel()>(y==c).sum()/2)

    def test_weights_ignore_order_and_normalize(self):
        rng=np.random.default_rng(4)
        x=rng.normal(size=(30,5)); perm=rng.permutation(30)
        w=neighbor_weights(x,7,0.2)
        np.testing.assert_allclose(w.sum(axis=1),1)
        np.testing.assert_allclose(np.diag(w),0.2)
        np.testing.assert_allclose(neighbor_weights(x[perm],7,0.2),w[perm][:,perm])
        self.assertTrue(np.all(w>=0))

    def test_diffusion_is_stochastic_and_order_invariant(self):
        rng=np.random.default_rng(5)
        x=rng.normal(size=(30,5)); perm=rng.permutation(30)
        w=diffusion_weights(x,7,.05,5)
        np.testing.assert_allclose(w.sum(axis=1),1)
        self.assertTrue(np.all(w>=0))
        np.testing.assert_allclose(diffusion_weights(x[perm],7,.05,5),w[perm][:,perm])

if __name__=='__main__': unittest.main()
