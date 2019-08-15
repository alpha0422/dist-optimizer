import os
import time
import math
import unittest
import argparse

import torch
import apex
import dist_optimizer

from functools import reduce
from operator import mul
from torch.nn.utils import clip_grad_norm_

def parse_args():
    parser = argparse.ArgumentParser()

    distributed = parser.add_argument_group('distributed setup')
    distributed.add_argument('--rank', default=0, type=int,
        help='global rank of the process, do not set!')
    distributed.add_argument("--local_rank", default=0, type=int,
        help='local rank of the process, do not set!')

    args, unknownargs = parser.parse_known_args()
    return args

class RefOptimizer(torch.optim.Optimizer):
    def __init__(self, params, grad_clip=float('inf'), **options):
        self.grad_clip = grad_clip
        self.params = params
        self.optim = apex.optimizers.FusedAdam(params, **options)
        self.world_size = torch.distributed.get_world_size()

    def step(self, grads, scale=1.0):
        scale *= self.world_size
        norm = (torch.cat(grads) / scale).norm(2).item()

        if math.isfinite(norm):
            clip_coef = self.grad_clip / (norm + 1e-6)
            if clip_coef >= 1:
                clip_coef = scale
            else:
                clip_coef = scale / clip_coef

            self.optim.step(grads=grads, scale=clip_coef)
            return True

        return False

class DistributedOptimizerTest(unittest.TestCase):
    def setUp(self):
        args = parse_args()

        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        assert torch.distributed.is_initialized()

        self.world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()
        self.local_rank = args.local_rank
        self.device_count = torch.cuda.device_count()

        torch.manual_seed(4321)
        torch.cuda.manual_seed(1234)

        # MLPerf GNMT has 160671297 parameters
        self.gnmt_sizes = [[4096, 1024], [4096, 1024], [4096], [4096], [4096, 1024], [4096, 1024], [4096], [4096], [4096, 2048], [4096, 1024], [4096], [4096], [4096, 1024], [4096, 1024], [4096], [4096], [4096, 1024], [4096, 1024], [4096], [4096], [32320, 1024], [4096, 1024], [4096, 1024], [4096], [4096], [1024], [1], [1024], [1024, 1024], [1024, 1024], [4096, 2048], [4096, 1024], [4096], [4096], [4096, 2048], [4096, 1024], [4096], [4096], [4096, 2048], [4096, 1024], [4096], [4096], [32320, 1024], [32320]]
        self.grad_clip = 5.0

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
            ref_param.append(torch.nn.Parameter(tensor.clone().half().float()))
            tst_param.append(torch.nn.Parameter(tensor.clone().half()))
        with torch.no_grad():
            ref_param = [torch.cat([p.view(-1) for p in ref_param])]

        ref_optim = ref_optim_class(ref_param,
            grad_clip=self.grad_clip, **ref_optim_option)
        tst_optim = tst_optim_class(tst_param, apex.optimizers.FusedAdam,
            grad_clip=self.grad_clip, align=64, **tst_optim_option)

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
            self.gen_test_inputs(sizes, RefOptimizer,
            dist_optimizer.FullyDistributedOptimizer,
            adam_option, adam_option, random=True)

        for i in range(iters):
            ref_grads, tst_grads = self.gen_mixed_grad(tst_param, random=True)

            torch.distributed.all_reduce(ref_grads[0], async_op=False)
            ref_steped = ref_optim.step(grads=ref_grads, scale=scale)
            tst_steped = tst_optim.step(grads=tst_grads, scale=scale)

            self.assertEqual(ref_steped, tst_steped)
            self.print_max_diff_elem(ref_param, tst_param)
           
    def test_intra_node_distributed_optimizer_function(self):
        iters = 4
        sizes = self.gnmt_sizes
        adam_option = {'lr':1e-2, 'betas':(0.9, 0.999), 'eps':1e-08,
            'weight_decay':0, 'amsgrad':False}
        scale = 4.0

        ref_param, tst_param, ref_optim, tst_optim = \
            self.gen_test_inputs(sizes, RefOptimizer,
            dist_optimizer.IntraNodeDistributedOptimizer,
            adam_option, adam_option, random=True)

        for i in range(iters):
            ref_grads, tst_grads = self.gen_mixed_grad(tst_param, random=True)

            torch.distributed.all_reduce(ref_grads[0], async_op=False)
            ref_steped = ref_optim.step(grads=ref_grads, scale=scale)
            tst_steped = tst_optim.step(grads=tst_grads, scale=scale)

            self.assertEqual(ref_steped, tst_steped)
            self.print_max_diff_elem(ref_param, tst_param)
           
    def test_intra_node_accelerated_optimizer_function(self):
        iters = 4
        sizes = self.gnmt_sizes
        adam_option = {'lr':1e-2, 'betas':(0.9, 0.999), 'eps':1e-08,
            'weight_decay':0, 'amsgrad':False}
        scale = 4.0

        ref_param, tst_param, ref_optim, tst_optim = \
            self.gen_test_inputs(sizes, RefOptimizer,
            dist_optimizer.IntraNodeAcceleratedOptimizer,
            adam_option, adam_option, random=True)

        for i in range(iters):
            ref_grads, tst_grads = self.gen_mixed_grad(tst_param, random=True)

            torch.distributed.all_reduce(ref_grads[0], async_op=False)
            ref_steped = ref_optim.step(grads=ref_grads, scale=scale)
            tst_steped = tst_optim.step(grads=tst_grads, scale=scale)

            self.assertEqual(ref_steped, tst_steped)
            self.print_max_diff_elem(ref_param, tst_param)
           
    def test_two_level_distributed_optimizer_function(self):
        iters = 4
        sizes = self.gnmt_sizes
        adam_option = {'lr':1e-2, 'betas':(0.9, 0.999), 'eps':1e-08,
            'weight_decay':0, 'amsgrad':False}
        scale = 4.0

        ref_param, tst_param, ref_optim, tst_optim = \
            self.gen_test_inputs(sizes, RefOptimizer,
            dist_optimizer.TwoLevelDistributedOptimizer,
            adam_option, adam_option, random=True)

        for i in range(iters):
            ref_grads, tst_grads = self.gen_mixed_grad(tst_param, random=True)

            torch.distributed.all_reduce(ref_grads[0], async_op=False)
            ref_steped = ref_optim.step(grads=ref_grads, scale=scale)
            tst_steped = tst_optim.step(grads=tst_grads, scale=scale)

            self.assertEqual(ref_steped, tst_steped)
            self.print_max_diff_elem(ref_param, tst_param)
           
    def test_hierarchical_distributed_optimizer_function(self):
        iters = 4
        sizes = self.gnmt_sizes
        adam_option = {'lr':1e-2, 'betas':(0.9, 0.999), 'eps':1e-08,
            'weight_decay':0, 'amsgrad':False}
        scale = 4.0

        ref_param, tst_param, ref_optim, tst_optim = \
            self.gen_test_inputs(sizes, RefOptimizer,
            dist_optimizer.HierarchicalDistributedOptimizer,
            adam_option, adam_option, random=True)

        for i in range(iters):
            ref_grads, tst_grads = self.gen_mixed_grad(tst_param, random=True)

            torch.distributed.all_reduce(ref_grads[0], async_op=False)
            ref_steped = ref_optim.step(grads=ref_grads, scale=scale)
            tst_steped = tst_optim.step(grads=tst_grads, scale=scale)

            self.assertEqual(ref_steped, tst_steped)
            self.print_max_diff_elem(ref_param, tst_param)
           
    def test_dist_opt_perf(self):
        iters = 1000
        sizes = self.gnmt_sizes
        adam_option = {'lr':1e-3, 'betas':(0.9, 0.999), 'eps':1e-08,
            'weight_decay':0, 'amsgrad':False}
        grad_clip = None
        scale = 1.0

        for size in range(30, 19, -1):
            sizes = [[2**(size-1)]]
            print("{}:{} Test {} half elements".format(
                self.world_size, self.rank, sizes[0][0]))

            ref_param, tst_param, ref_optim, tst_optim = \
                self.gen_test_inputs(sizes, RefOptimizer,
                dist_optimizer.FullyDistributedOptimizer,
                adam_option, adam_option)
            ref_grads, tst_grads = self.gen_mixed_grad(tst_param, random=False)
    
            # Test default all-reduce optimizer
            torch.cuda.empty_cache()
            torch.distributed.all_reduce(ref_grads[0], async_op=False)
            ref_optim.step(grads=ref_grads, scale=scale)
    
            self.barrier()
            ts = time.time()
            for i in range(iters):
                torch.distributed.all_reduce(ref_grads[0], async_op=False)
                ref_optim.step(grads=ref_grads, scale=scale)
            self.barrier()
            td = time.time()
            print("{}:{} Ref time {:.2f} s elapsed for {} iterations, norm {:.4f}".format(
                self.world_size, self.rank, td - ts, iters, self.norm(ref_param)))
            del ref_grads, ref_param, ref_optim
            
            # Test FullyDistributedOptimizer
            torch.cuda.empty_cache()
            tst_optim = dist_optimizer.FullyDistributedOptimizer(
                tst_param, apex.optimizers.FusedAdam,
                grad_clip=grad_clip, align=64, **adam_option)
            tst_optim.step(scale=scale)
    
            self.barrier()
            ts = time.time()
            for i in range(iters):
                tst_optim.step(scale=scale)
            self.barrier()
            td = time.time()
            print("{}:{} FullyDistributedOptimizer time {:.2f} s elapsed for {} iterations, norm {:.4f}".format(
                self.world_size, self.rank, td - ts, iters, self.norm(tst_param)))
            del tst_optim
    
            # Test IntraNodeDistributedOptimizer
            torch.cuda.empty_cache()
            tst_optim = dist_optimizer.IntraNodeDistributedOptimizer(
                tst_param, apex.optimizers.FusedAdam,
                grad_clip=grad_clip, align=64, **adam_option)
            tst_optim.step(scale=scale)
    
            self.barrier()
            ts = time.time()
            for i in range(iters):
                tst_optim.step(scale=scale)
            self.barrier()
            td = time.time()
            print("{}:{} IntraNodeDistributedOptimizer time {:.2f} s elapsed for {} iterations, norm {:.4f}".format(
                self.world_size, self.rank, td - ts, iters, self.norm(tst_param)))
            del tst_optim
    
            # Test IntraNodeAcceleratedOptimizer
            torch.cuda.empty_cache()
            tst_optim = dist_optimizer.IntraNodeAcceleratedOptimizer(
                tst_param, apex.optimizers.FusedAdam,
                grad_clip=grad_clip, align=64, **adam_option)
            tst_optim.step(scale=scale)
    
            self.barrier()
            ts = time.time()
            for i in range(iters):
                tst_optim.step(scale=scale)
            self.barrier()
            td = time.time()
            print("{}:{} IntraNodeAcceleratedOptimizer time {:.2f} s elapsed for {} iterations, norm {:.4f}".format(
                self.world_size, self.rank, td - ts, iters, self.norm(tst_param)))
            del tst_optim
    
            # Test TwoLevelDistributedOptimizer
            torch.cuda.empty_cache()
            tst_optim = dist_optimizer.TwoLevelDistributedOptimizer(
                tst_param, apex.optimizers.FusedAdam,
                grad_clip=grad_clip, align=64, **adam_option)
            tst_optim.step(scale=scale)
    
            self.barrier()
            ts = time.time()
            for i in range(iters):
                tst_optim.step(scale=scale)
            self.barrier()
            td = time.time()
            print("{}:{} TwoLevelDistributedOptimizer time {:.2f} s elapsed for {} iterations, norm {:.4f}".format(
                self.world_size, self.rank, td - ts, iters, self.norm(tst_param)))
            del tst_optim
            
            # Test HierarchicalDistributedOptimizer
            torch.cuda.empty_cache()
            tst_optim = dist_optimizer.HierarchicalDistributedOptimizer(
                tst_param, apex.optimizers.FusedAdam,
                grad_clip=grad_clip, align=64, **adam_option)
            tst_optim.step(scale=scale)
    
            self.barrier()
            ts = time.time()
            for i in range(iters):
                tst_optim.step(scale=scale)
            self.barrier()
            td = time.time()
            print("{}:{} HierarchicalDistributedOptimizer time {:.2f} s elapsed for {} iterations, norm {:.4f}".format(
                self.world_size, self.rank, td - ts, iters, self.norm(tst_param)))
            del tst_grads, tst_param, tst_optim

if __name__ == '__main__':
    test = DistributedOptimizerTest()
    test.setUp()

    torch.distributed.barrier()
    print("Checking distributed optimizer performance ...")
    test.test_dist_opt_perf()

    torch.cuda.empty_cache()

    torch.distributed.barrier()
    print("Checking FullyDistributedOptimizer functionality ...")
    test.test_fully_distributed_optimizer_function()

    torch.distributed.barrier()
    print("Checking IntraNodeDistributedOptimizer functionality ...")
    test.test_intra_node_distributed_optimizer_function()

    torch.distributed.barrier()
    print("Checking IntraNodeAcceleratedOptimizer functionality ...")
    test.test_intra_node_accelerated_optimizer_function()

    torch.distributed.barrier()
    print("Checking TwoLevelDistributedOptimizer functionality ...")
    test.test_two_level_distributed_optimizer_function()

    torch.distributed.barrier()
    print("Checking HierarchicalDistributedOptimizer functionality ...")
    test.test_hierarchical_distributed_optimizer_function()

