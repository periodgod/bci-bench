import unittest
import torch
from wyu_bci_bench.ssl import MaskedEEGAutoencoder
from wyu_bci_bench.mae_downstream import MaskedEEGClassifier


class MAEDownstreamTests(unittest.TestCase):
    def test_same_encoder_and_frozen_gradients(self):
        torch.manual_seed(1)
        ae=MaskedEEGAutoencoder(3,patch_size=12,hidden=12,heads=3,depth=1)
        clf=MaskedEEGClassifier(ae,4,frozen=True).train()
        x=torch.randn(2,3,48)
        expected=ae.encode_tokens(x).mean(1).flatten(1)
        torch.testing.assert_close(clf.encode(x),expected)
        clf(x).sum().backward()
        self.assertFalse(ae.training)
        self.assertTrue(all(p.grad is None for p in ae.parameters()))
        self.assertIsNotNone(clf.head.weight.grad)

    def test_masked_positions_preserved_and_signal_hidden(self):
        ae=MaskedEEGAutoencoder(2,patch_size=4,hidden=8,heads=2,depth=1).eval()
        seen=[]
        hook=ae.encoder.register_forward_pre_hook(lambda m,args:seen.append(args[0].detach()))
        x=torch.randn(1,2,16)
        first=ae(x,generator=torch.Generator().manual_seed(3),return_details=True)
        altered=x.unfold(-1,4,4).clone();altered[first['mask']]+=100
        ae(altered.reshape_as(x),generator=torch.Generator().manual_seed(3))
        hook.remove()
        torch.testing.assert_close(seen[0],seen[1])
        self.assertGreater(float(seen[0][0,first['mask'].flatten()].std(dim=0).sum()),0.)


if __name__=='__main__':unittest.main()
