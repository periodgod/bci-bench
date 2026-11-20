import copy
import unittest
import numpy as np
import torch
from wyu_bci_bench.foundation.brainomni import BrainOmniClassifier
from wyu_bci_bench.foundation.brainomni_personal import (personal_partition,
    PersonalFiLM, add_personal_lora, personal_state, load_personal_state)
from test_brainomni_adaptation import encoder


class PersonalTests(unittest.TestCase):
    def test_calibration_gap_test_are_disjoint_and_chronological(self):
        subjects=np.repeat(['a','b'],20); sessions=np.repeat('s',40)
        y=np.tile(np.repeat([0,1],10),2)
        cal,gap,test=personal_partition(subjects,sessions,y,'a',k=3,gap=1)
        self.assertEqual((cal.sum(),gap.sum(),test.sum()),(6,2,12))
        np.testing.assert_array_equal(np.flatnonzero(cal),[0,1,2,10,11,12])
        np.testing.assert_array_equal(cal.astype(int)+gap+test,subjects=='a')
        self.assertFalse(np.any(cal&test))

    def test_personal_branches_start_from_population_and_global_stays_frozen(self):
        torch.manual_seed(14)
        population=BrainOmniClassifier(encoder(),strategy='lora',rank=2).eval()
        x=torch.randn(4,3,4,8)
        reference=population(x)
        for model in [PersonalFiLM(copy.deepcopy(population)),add_personal_lora(copy.deepcopy(population),rank=2)]:
            model.eval()
            torch.testing.assert_close(reference,model(x),atol=1e-6,rtol=1e-5)
            for name,p in model.named_parameters():
                if 'lora_' in name and '.default.' in name:self.assertFalse(p.requires_grad)
            original={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
            loss=model(x).square().mean();loss.backward()
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01);opt.step()
            for n,p in model.named_parameters():
                if n in original:torch.testing.assert_close(p,original[n],atol=0,rtol=0)
            state=personal_state(model)
            expected=model(x).detach()
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:p.add_(1)
            load_personal_state(model,state)
            torch.testing.assert_close(expected,model(x))


if __name__=='__main__':unittest.main()
