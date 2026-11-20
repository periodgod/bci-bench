import unittest
import numpy as np
import torch
from wyu_bci_bench.foundation.labram import prepare_input,channel_ids
from wyu_bci_bench.foundation.token_adaptation import pool_tokens


class TokenAdaptationTests(unittest.TestCase):
    def test_labram_units_and_channel_identity(self):
        x=np.full((2,3,480),100.,np.float32)
        p=prepare_input(x,120,['C3','Cz','C4'])
        self.assertEqual(p.shape,(2,3,4,200))
        self.assertAlmostEqual(float(p[...,20:-20].mean()),1.,places=3)
        self.assertEqual(channel_ids(['C3','Cz','C4']),[0,40,42,44])
        with self.assertRaises(ValueError): channel_ids(['C3','C3'])

    def test_time_pool_preserves_position(self):
        e=torch.arange(16.).reshape(1,16,1,1).expand(2,16,4,512)
        q=pool_tokens(e,'eegpt','time').reshape(2,4,512)
        torch.testing.assert_close(q[0,:,0],torch.tensor([1.5,5.5,9.5,13.5]))
        l=torch.zeros(1,9,200); l[:,1:]=torch.tensor([1,2,3,4,1,2,3,4]).reshape(1,8,1)
        torch.testing.assert_close(pool_tokens(l,'labram','time').reshape(1,4,200)[0,:,0],torch.arange(1.,5.))


if __name__=='__main__': unittest.main()
