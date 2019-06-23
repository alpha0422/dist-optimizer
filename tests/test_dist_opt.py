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

class FullyDistributedOptimizerTest(unittest.TestCase):
    def setUp(self):
        torch.cuda.manual_seed(1234)
        args = parse_args()

        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        assert torch.distributed.is_initialized()

        self.world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()
        self.local_rank = args.local_rank
        self.device_count = torch.cuda.device_count()

        # MLPerf GNMT has 160671297 parameters
        self.gnmt_sizes = [[4096, 1024], [4096, 1024], [4096], [4096], [4096, 1024], [4096, 1024], [4096], [4096], [4096, 2048], [4096, 1024], [4096], [4096], [4096, 1024], [4096, 1024], [4096], [4096], [4096, 1024], [4096, 1024], [4096], [4096], [32320, 1024], [4096, 1024], [4096, 1024], [4096], [4096], [1024], [1], [1024], [1024, 1024], [1024, 1024], [4096, 2048], [4096, 1024], [4096], [4096], [4096, 2048], [4096, 1024], [4096], [4096], [4096, 2048], [4096, 1024], [4096], [4096], [32320, 1024], [32320]]

        self.barrier()

    def barrier(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def gen_test_inputs(self, sizes, ref_optim_class, tst_optim_class,
        ref_optim_option, tst_optim_option, random=True):
        tensors = []
        for size in sizes:
            if random:
                tensors.append(torch.randn(size, dtype=torch.float, device='cuda'))
            else:
                tensors.append(torch.zeros(size, dtype=torch.float, device='cuda'))

        ref_param = []
        tst_param = []
        for tensor in tensors:
            ref_param.append(torch.nn.Parameter(tensor.clone()))
            tst_param.append(torch.nn.Parameter(tensor.clone().half()))
        with torch.no_grad():
            ref_param = [torch.cat([p.view(-1) for p in ref_param])]

        ref_optim = ref_optim_class(ref_param, **ref_optim_option)
        tst_optim = tst_optim_class(tst_param, apex.optimizers.FusedAdam,
            grad_clip=5.0, align=64, **tst_optim_option)

        return (ref_param, tst_param, ref_optim, tst_optim)

    def gen_mixed_grad(self, tst_param, scale=1.0, random=True):
        tst_grads = []
        for p_tst in tst_param:
            if random:
                grad = torch.randn_like(p_tst).half()
            else:
                grad = torch.zeros_like(p_tst).half()
            tst_grads.append(grad)
            p_tst.grad = grad

        with torch.no_grad():
            ref_grads = [torch.cat([g.view(-1) for g in tst_grads])]
        return ref_grads, tst_grads

    def print_max_diff_elem(self, ref, tst):
        with torch.no_grad():
            ref = torch.cat([p.half().view(-1) for p in ref])
            tst = torch.cat([p.half().view(-1) for p in tst])
        ref, tst = ref.flatten(), tst.flatten()
        diff = (ref - tst).abs().max()
        idx = (ref - tst).abs().argmax()
        print("{}:{} Max diff: idx: {}, diff: {:.6f}, ref: {:.6f}, tst: {:.6f}".format(
            self.world_size, self.rank, idx, diff, ref[idx], tst[idx]))

    def assert_print_max_diff_elem(self, ref, tst):
        with torch.no_grad():
            ref = torch.cat([p.half().view(-1) for p in ref])
            tst = torch.cat([p.half().view(-1) for p in tst])
        ref, tst = ref.flatten(), tst.flatten()
        diff = (ref - tst).abs().max()
        idx = (ref - tst).abs().argmax()
        print("{}:{} Max diff: idx: {}, diff: {:.6f}, ref: {:.6f}, tst: {:.6f}".format(
            self.world_size, self.rank, idx, diff, ref[idx], tst[idx]))
        self.assertTrue(torch.allclose(ref, tst, atol=1e-3, rtol=1e-2))

    def norm(self, alist):
        tot_norm = 0.0
        norm_type = 2
        for p in alist:
            p_norm = p.norm(norm_type)
            tot_norm += p_norm.item() ** norm_type
        tot_norm = tot_norm ** (1. / norm_type)
        return tot_norm

    def test_fully_distributed_optimizer_function(self):
        iters = 4
        sizes = self.gnmt_sizes
        adam_option = {'lr':1e-2, 'betas':(0.9, 0.999), 'eps':1e-08,
            'weight_decay':0, 'amsgrad':False}
        scale = 4.0

        ref_param, tst_param, ref_optim, tst_optim = \
            self.gen_test_inputs(sizes, apex.optimizers.FusedAdam,
            dist_optimizer.FullyDistributedOptimizer,
            adam_option, adam_option, random=True)

        for i in range(iters):
            ref_grads, tst_grads = self.gen_mixed_grad(tst_param, random=True)

            torch.distributed.all_reduce(ref_grads[0], async_op=False)
            ref_optim.step(grads=ref_grads, scale=scale)
            tst_optim.step(grads=tst_grads, scale=scale)

            self.print_max_diff_elem(ref_param, tst_param)
           
    def test_intra_node_accelerated_distributed_optimizer_function(self):
        iters = 4
        sizes = self.gnmt_sizes
        adam_option = {'lr':1e-2, 'betas':(0.9, 0.999), 'eps':1e-08,
            'weight_decay':0, 'amsgrad':False}
        scale = 4.0

        ref_param, tst_param, ref_optim, tst_optim = \
            self.gen_test_inputs(sizes, apex.optimizers.FusedAdam,
            dist_optimizer.IntraNodeAcceleratedOptimizer,
            adam_option, adam_option, random=True)

        for i in range(iters):
            ref_grads, tst_grads = self.gen_mixed_grad(tst_param, random=True)

            torch.distributed.all_reduce(ref_grads[0], async_op=False)
            ref_optim.step(grads=ref_grads, scale=scale)
            tst_optim.step(grads=tst_grads, scale=scale)

            self.print_max_diff_elem(ref_param, tst_param)
           
    def test_hierarchical_distributed_optimizer_function(self):
        iters = 4
        sizes = self.gnmt_sizes
        adam_option = {'lr':1e-2, 'betas':(0.9, 0.999), 'eps':1e-08,
            'weight_decay':0, 'amsgrad':False}
        scale = 4.0

        ref_param, tst_param, ref_optim, tst_optim = \
            self.gen_test_inputs(sizes, apex.optimizers.FusedAdam,
            dist_optimizer.HierarchicalDistributedOptimizer,
            adam_option, adam_option, random=True)

        for i in range(iters):
            ref_grads, tst_grads = self.gen_mixed_grad(tst_param, random=True)

            torch.distributed.all_reduce(ref_grads[0], async_op=False)
            ref_optim.step(grads=ref_grads, scale=scale)
            tst_optim.step(grads=tst_grads, scale=scale)

            self.print_max_diff_elem(ref_param, tst_param)
           
    def test_dist_opt_perf(self):
        iters = 1000
        sizes = self.gnmt_sizes
        adam_option = {'lr':1e-3, 'betas':(0.9, 0.999), 'eps':1e-08,
            'weight_decay':0, 'amsgrad':False}
        scale = 4.0

        ref_param, tst_param, ref_optim, tst_optim = \
            self.gen_test_inputs(sizes, apex.optimizers.FusedAdam,
            dist_optimizer.IntraNodeAcceleratedOptimizer,
            adam_option, adam_option)
        ref_grads, tst_grads = self.gen_mixed_grad(tst_param, random=False)

        # Warm up
        torch.distributed.all_reduce(ref_grads[0], async_op=False)
        ref_optim.step(grads=ref_grads, scale=scale)
        tst_optim.step(grads=tst_grads, scale=scale)

        self.barrier()
        ts = time.time()
        for i in range(iters):
            torch.distributed.all_reduce(ref_grads[0], async_op=False)
            ref_optim.step(grads=ref_grads, scale=scale)
        self.barrier()
        td = time.time()
        print("{}:{} Ref time {:.2f} s elapsed for {} iterations, norm {:.4f}".format(
            self.world_size, self.rank, td - ts, iters, self.norm(ref_param)))

        self.barrier()
        ts = time.time()
        for i in range(iters):
            tst_optim.step(grads=tst_grads, scale=scale)
        self.barrier()
        td = time.time()
        print("{}:{} Opt time {:.2f} s elapsed for {} iterations, norm {:.4f}".format(
            self.world_size, self.rank, td - ts, iters, self.norm(tst_param)))

if __name__ == '__main__':
    test = FullyDistributedOptimizerTest()
    test.setUp()

    torch.distributed.barrier()
    print("Checking FullyDistributedOptimizer functionality ...")
    test.test_fully_distributed_optimizer_function()

    torch.distributed.barrier()
    print("Checking IntraNodeAcceleratedOptimizer functionality ...")
    test.test_intra_node_accelerated_distributed_optimizer_function()

    torch.distributed.barrier()
    print("Checking HierarchicalDistributedOptimizer functionality ...")
    test.test_hierarchical_distributed_optimizer_function()

    torch.distributed.barrier()
    print("Checking distributed optimizer performance ...")
    test.test_dist_opt_perf()

