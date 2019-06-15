import os
import time
import unittest
import argparse

import torch
import apex
import dist_optimizer

def parse_args():
    parser = argparse.ArgumentParser()

    distributed = parser.add_argument_group('distributed setup')
    distributed.add_argument('--rank', default=0, type=int,
        help='global rank of the process, do not set!')
    distributed.add_argument("--local_rank", default=0, type=int,
        help='local rank of the process, do not set!')

    args, unknownargs = parser.parse_known_args()
    return args

class DistributedOptimizerTest(unittest.TestCase):
    def setUp(self):
        torch.cuda.manual_seed(1234)
        args = parse_args()

        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        assert torch.distributed.is_initialized()

        self.world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()

        self.barrier()

    def barrier(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(torch.cuda.FloatTensor(1))
            torch.cuda.synchronize()

    def gen_test_inputs(self, sizes, ref_optim_option, tst_optim_option):
        tensors = []
        for size in sizes:
            tensors.append(torch.randn(size, dtype=torch.float, device='cuda'))

        ref_param = []
        tst_param = []
        for tensor in tensors:
            ref_param.append(torch.nn.Parameter(tensor.clone()))
            tst_param.append(torch.nn.Parameter(tensor.clone().half()))
        with torch.no_grad():
            ref_param = [torch.cat([p.view(-1) for p in ref_param])]

        ref_optim = apex.optimizers.FusedAdam(ref_param, **ref_optim_option)
        tst_optim = dist_optimizer.DistributedOptimizer(tst_param,
            apex.optimizers.FusedAdam, grad_clip=5.0, align=64, **tst_optim_option)

        return (ref_param, tst_param, ref_optim, tst_optim)

    def gen_mixed_grad(self, tst_param, scale=1.0):
        tst_grads = []
        for p_tst in tst_param:
            grad = torch.rand_like(p_tst).half()
            tst_grads.append(grad)
            p_tst.grad = grad
        ref_grads = [torch.cat([g.view(-1) for g in tst_grads])]
        return ref_grads, tst_grads

    def print_max_diff_elem(self, ref, tst):
        ref, tst = ref.flatten(), tst.flatten()
        diff = (ref - tst).abs().max()
        idx = (ref - tst).abs().argmax()
        print("Max diff: idx: {}, diff: {:.6f}, ref: {:.6f}, tst: {:.6f}".format(
            idx, diff, ref[idx], tst[idx]))

    def test_attn_score_function(self):
        pass

    def test_attn_score_perf(self):
        # MLPerf GNMT has 160671297 parameters
        iters = 1000
        sizes = [[4096, 1024], [4096, 1024], [4096], [4096], [4096, 1024], [4096, 1024], [4096], [4096], [4096, 2048], [4096, 1024], [4096], [4096], [4096, 1024], [4096, 1024], [4096], [4096], [4096, 1024], [4096, 1024], [4096], [4096], [32320, 1024], [4096, 1024], [4096, 1024], [4096], [4096], [1024], [1], [1024], [1024, 1024], [1024, 1024], [4096, 2048], [4096, 1024], [4096], [4096], [4096, 2048], [4096, 1024], [4096], [4096], [4096, 2048], [4096, 1024], [4096], [4096], [32320, 1024], [32320]]
        adam_option = {'lr':5e-4, 'betas':(0.9, 0.999), 'eps':1e-08,
            'weight_decay':0, 'amsgrad':False}
        scale = 4

        ref_param, tst_param, ref_optim, tst_optim = \
            self.gen_test_inputs(sizes, adam_option, adam_option)
        ref_grads, tst_grads = self.gen_mixed_grad(tst_param)

        # Warm up
        torch.distributed.all_reduce(ref_grads[0], async_op=False)
        ref_optim.step(grads=ref_grads, scale=scale)
        tst_optim.step(grads=ref_grads, scale=scale)

        self.barrier()
        ts = time.time()
        for i in range(iters):
            torch.distributed.all_reduce(ref_grads[0], async_op=False)
            ref_optim.step(grads=ref_grads, scale=scale)
        td = time.time()
        print("{}:{} Ref time {:.2f} s elapsed for {} iterations, norm {:.4f}".format(
            self.world_size, self.rank, td - ts, iters, ref_param[0].norm()))

        self.barrier()
        ts = time.time()
        for i in range(iters):
            tst_optim.step(grads=ref_grads, scale=scale)
        td = time.time()
        print("{}:{} Opt time {:.2f} s elapsed for {} iterations, norm {:.4f}".format(
            self.world_size, self.rank, td - ts, iters, tst_param[0].norm()))

if __name__ == '__main__':
    #script_path = os.path.dirname(os.path.realpath(__file__))
    #unittest.main()

    test = DistributedOptimizerTest()
    test.setUp()
    test.test_attn_score_perf()

