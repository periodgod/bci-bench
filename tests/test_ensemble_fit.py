import unittest
import numpy as np
from wyu_bci_bench.ensemble_fit import fit_simplex_weights


class EnsembleFitTests(unittest.TestCase):
    def test_identical_experts_stay_uniform(self):
        p = np.array([[.8,.2],[.3,.7]])
        w = fit_simplex_weights(np.stack([p,p,p],axis=1),[0,1])
        np.testing.assert_allclose(w,np.ones(3)/3,atol=1e-8)

    def test_good_expert_and_weighted_conflict(self):
        p = np.array([[[.9,.1],[.1,.9]],[[.1,.9],[.9,.1]]])
        self.assertGreater(fit_simplex_weights(p,[0,0],sample_weight=[9,1])[0],.8)
        self.assertLess(fit_simplex_weights(p,[0,0],sample_weight=[1,9])[0],.2)
        np.testing.assert_allclose(fit_simplex_weights(p,[0,1]),[1,0],atol=1e-7)

    def test_bad_probabilities_rejected(self):
        with self.assertRaises(ValueError):
            fit_simplex_weights(np.array([[[1.,1.]]]),[0])


if __name__ == '__main__':
    unittest.main()
