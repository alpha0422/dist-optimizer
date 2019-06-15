import os
import time
import unittest

import torch
import apex
import dist_opt

class AttentionScoreTest(unittest.TestCase):
    def setUp(self):
        torch.cuda.manual_seed(1234)

    def gen_test_inputs(self):
        options = {'device': 'cuda:0', 'dtype': torch.float16, 'requires_grad': True}

    def print_max_diff_elem(self, ref, tst):
        ref, tst = ref.flatten(), tst.flatten()
        diff = (ref - tst).abs().max()
        idx = (ref - tst).abs().argmax()
        print("Max diff: idx: {}, diff: {:.6f}, ref: {:.6f}, tst: {:.6f}".format(
            idx, diff, ref[idx], tst[idx]))

    def test_attn_score_function(self):
        pass

    def test_attn_score_perf(self):
        pass

if __name__ == '__main__':
    script_path = os.path.dirname(os.path.realpath(__file__))
    unittest.main()

