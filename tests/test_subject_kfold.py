import unittest
import numpy as np
from wyu_bci_bench.splits import subject_kfold


class SubjectKFoldTests(unittest.TestCase):
    def test_87_people_have_single_out_of_fold_assignment_and_inner_validation(self):
        subjects = np.repeat([f'sub-{i:02d}' for i in range(1, 88)], 3)
        folds = subject_kfold(subjects, 5, seed=19)
        self.assertEqual(len(folds), 5)
        np.testing.assert_array_equal(np.stack([s.test for s in folds]).sum(0), 1)
        for split in folds:
            split.check_disjoint(subjects)
            self.assertIn(len(set(subjects[split.test])), [17, 18])
            self.assertTrue(np.all(split.train | split.test | split.valid))
            self.assertEqual(len(set(subjects[split.valid])), 14)

    def test_row_permutation_does_not_change_participant_assignment(self):
        subjects = np.repeat([str(i) for i in range(20)], np.arange(1, 21))
        perm = np.random.default_rng(0).permutation(len(subjects))
        first = subject_kfold(subjects, seed=31)
        second = subject_kfold(subjects[perm], seed=31)
        for a, b in zip(first, second):
            for field in ['train', 'valid', 'test']:
                self.assertEqual(set(subjects[getattr(a, field)]),
                                 set(subjects[perm][getattr(b, field)]))

    def test_seed_changes_assignment_and_invalid_designs_raise(self):
        subjects = np.array([str(i) for i in range(10)])
        self.assertNotEqual(subject_kfold(subjects, seed=1)[0].info,
                            subject_kfold(subjects, seed=2)[0].info)
        for n_splits, fraction in [(1, .2), (11, .2), (5, 0), (5, 1)]:
            with self.assertRaises(ValueError):
                subject_kfold(subjects, n_splits, valid_fraction=fraction)
        with self.assertRaises(ValueError):
            subject_kfold(np.array(['one', 'two']), 2)
