import unittest
import os
import random
import time
import itertools

import torch
import apex

class TestFusedAdam(unittest.TestCase):
    def setUp(self, max_abs_diff=1e-3, max_rel_diff=1, iters=7):
        self.max_abs_diff = max_abs_diff
        self.max_rel_diff = max_rel_diff
        self.iters = iters
        torch.cuda.manual_seed(9876)

    def tearDown(self):
        pass

    def gen_param_optim(self, tensors, ref_adam_option, tst_adam_option=None):
        ref_param = []
        tst_param = []
        for tensor in tensors:
            ref_param.append(torch.nn.Parameter(tensor.clone()))
            tst_param.append(torch.nn.Parameter(tensor.clone()))

        ref_optim = torch.optim.Adam(ref_param, **ref_adam_option)
        if tst_adam_option:
            tst_optim = apex.optimizers.FusedAdam(tst_param, **tst_adam_option)
        else:
            tst_optim = apex.optimizers.FusedAdam(tst_param, **ref_adam_option)
       
        return (ref_param, tst_param, ref_optim, tst_optim)

    def gen_grad(self, ref_param, tst_param):
        for p_ref, p_tst in zip(ref_param, tst_param):
            p_ref.grad = torch.rand_like(p_ref)
            p_tst.grad = p_ref.grad

    def gen_mixed_grad(self, ref_param, tst_param, scale=1.0):
        half_grads = []
        for p_ref, p_tst in zip(ref_param, tst_param):
            half_grads.append(torch.rand_like(p_ref).half())
            p_ref.grad = half_grads[-1].float() / scale
        return half_grads

    def get_max_diff(self, ref_param, tst_param):
        max_abs_diff = max_rel_diff = 0
        for p_ref, p_tst in zip(ref_param, tst_param):
            max_abs_diff_p = (p_ref - p_tst.type(p_ref.type())).abs().max().item()
            max_rel_diff_p = ((p_ref - p_tst.type(p_ref.type())) / p_ref).abs().max().item()

            if max_abs_diff_p > max_abs_diff:  max_abs_diff = max_abs_diff_p
            if max_rel_diff_p > max_rel_diff:  max_rel_diff = max_rel_diff_p

        return max_abs_diff, max_rel_diff

    def test_fp16_output(self):
        iters = 1000
        sizes = [[2**i * 512] for i in range(21)]
        adam_option = {'lr':5e-4, 'betas':(0.9, 0.999), 'eps':1e-08,
            'weight_decay':0, 'amsgrad':False}
        ranks = [1, 8, 16, 24, 48, 384]

        for rank, nelem in itertools.product(ranks, sizes):
            elems = [i // rank for i in nelem]
            tensor = torch.rand(elems, dtype=torch.float, device='cuda')
            ref_param, tst_param, ref_optim, tst_optim = \
                self.gen_param_optim([tensor], adam_option)

            fp16_param = torch.nn.Parameter(tensor.clone().half())
            half_grads = self.gen_mixed_grad(ref_param, tst_param)

            # Warm up
            tst_optim.step(grads=half_grads, output_params=[fp16_param])

            torch.cuda.synchronize()
            ts = time.time()
            for i in range(iters):
                tst_optim.step(grads=half_grads, output_params=[fp16_param])
            torch.cuda.synchronize()
            td = time.time()

            # bytes in fp16 gradient
            # time in micro seconds
            # bandwidth in 7 fp32 memory accesses
            print("{} {} adam_perf {:.1f} {:.2f}".format(rank, nelem[0]*2,
                (td-ts)/iters*1e6, elems[0]*4*7*iters/(td-ts)/1e9))

if __name__ == '__main__':
    script_path = os.path.dirname(os.path.realpath(__file__))
    unittest.main()
