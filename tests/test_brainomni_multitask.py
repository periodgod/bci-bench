import copy
import unittest
import torch
from wyu_bci_bench.foundation.brainomni import BrainOmniClassifier
from wyu_bci_bench.foundation.brainomni_multitask import BrainOmniMultiTask
from test_brainomni_adaptation import encoder


class MultiTaskTests(unittest.TestCase):
    def test_shared_a_is_tied_b_is_separate_and_both_receive_gradients(self):
        torch.manual_seed(11)
        base=BrainOmniClassifier(encoder(),strategy='lora',rank=2)
        shared=BrainOmniMultiTask(copy.deepcopy(base),share_a=True)
        separate=BrainOmniMultiTask(copy.deepcopy(base),share_a=False)
        self.assertLess(sum(p.numel() for p in shared.parameters()),sum(p.numel() for p in separate.parameters()))
        x=torch.randn(4,3,4,8);tasks=torch.tensor([1,0,1,0]);y=torch.tensor([0,1,1,0])
        for m in (shared,separate):
            m.eval()
            p=m(x,tasks)
            for t in (0,1):
                m.activate(t)
                expected=m.heads[t](m.backbone(x[tasks==t]).mean(2).flatten(1))
                torch.testing.assert_close(p[tasks==t],expected)
            torch.nn.functional.cross_entropy(p,y).backward()
            names=[n for n,v in m.named_parameters() if v.requires_grad and v.grad is None]
            self.assertEqual(names,[])
            for module in m.backbone.modules():
                if hasattr(module,'lora_A') and 'task1' in module.lora_A:
                    self.assertEqual(module.lora_A['default'].weight is module.lora_A['task1'].weight,m.share_a)
                    self.assertIsNot(module.lora_B['default'].weight,module.lora_B['task1'].weight)


if __name__=='__main__':unittest.main()
