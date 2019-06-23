#!/usr/bin/env python

import math
import torch

class BasicDistributedOptimizer(object):
    """
    Basic virtual distributed optimizer.
    """
    def __init__(self, params, optimizer, grad_clip=None, align=64):
        # Assume params comes from model.half().parameters(), and params.grad
        # will be used for optimizer.step()
        assert torch.distributed.is_initialized(), \
            "Default process group hasn't been initialized."
        assert grad_clip is None or isinstance(grad_clip, float) or \
            callable(grad_clip), "Invalid gradient clipping argument."
        assert issubclass(optimizer, torch.optim.Optimizer), \
            "Expect optimizer as subclass of torch.optim.Optimizer."

        self.params = params
        self.grad_clip = grad_clip
        self.world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()

        self.devices = torch.cuda.device_count()
        assert self.world_size % self.devices == 0, \
            "Expect nodes to have the same number of devices."

        # FIXME: is it for sure rank % device_count is args.local_rank?
        self.nodes = self.world_size // self.devices
        self.device_rank = self.rank % self.devices
        self.node_rank = self.rank // self.devices

    def flatten_params_(self, params, align):
        """
        Flatten FP16 model's parameters.
        Return the flattened FP16 parameter, gradients and size with padding.
        Warn: gradients storage in the model can be changed/moved to other
              position in PyTorch, so it can't be flattened before training
              start, and must be flatten at each iteration currently.
        """
        # Count total elements with consideration of padding
        tot_nelem = 0
        for p in params:
            nelem = (p.numel() + align - 1) // align * align
            tot_nelem += nelem
    
        # Crate flattened storage and initialize to zeros
        # Warn: the padding value may affects the operations performed on
        #       flattened parameters/gradients, such as gradient clipping.
        flat_param = params[0].new_zeros(tot_nelem)
        flat_grad = params[0].new_zeros(tot_nelem)
        flat_param.grad = flat_grad
    
        with torch.no_grad():
            # Copy model's parameters to flattened parameters
            pointer = 0
            for p in params:
                flat_param[pointer:pointer+p.numel()].copy_(p.data.view(-1))
                nelem = (p.numel() + align - 1) // align * align
                pointer += nelem
    
            # Copy model's parameters to flattened parameters
            pointer = 0
            for p in params:
                flat_param[pointer:pointer+p.numel()].copy_(p.data.view(-1))
                nelem = (p.numel() + align - 1) // align * align
                pointer += nelem
    
            # Reset model's parameters/gradients to flattened parameters/gradients
            pointer = 0
            for p in params:
                p.set_(source=flat_param.storage(), storage_offset=pointer,
                    size=p.size())
                nelem = (p.numel() + align - 1) // align * align
                pointer += nelem
    
        return flat_param, flat_grad, tot_nelem

    def locate_rank_elem_(self, params, align, rank, rank_nelem):
        # Check the start parameter idex and offset
        pointer = 0
        rank_start_elem = rank * rank_nelem
        idx = None
        offset = None

        for i, p in enumerate(params):
            nelem = (p.numel() + align - 1) // align * align
            if idx is None and (pointer <= rank_start_elem < pointer + nelem):
                idx = i
                offset = rank_start_elem - pointer
                break
            pointer += nelem
        else:
            raise ValueError("Can't find chunk of data for given rank.")

        return idx, offset

    def initialize_fp32_params(self, fp32_params, params, params_start_idx, param_offset,
        align, rank_nelem):
        # Initialize rank local FP32 master parameters
        # Assume FP32 parameters has been fileed with zeros
        fp16_params = params
        fp32_pointer = 0

        with torch.no_grad():
            # Special process with first parameter as it may locate in the middle
            if param_offset < fp16_params[params_start_idx].numel():
                size = min(fp16_params[params_start_idx].numel() - param_offset, rank_nelem)
                fp32_params[fp32_pointer:fp32_pointer+size].copy_(
                    fp16_params[params_start_idx].view(-1)[param_offset:param_offset+size])
                nelem = (fp16_params[params_start_idx].numel() + align - 1) \
                    // align * align - param_offset
                fp32_pointer += nelem
            else:
                size = fp16_params[params_start_idx].numel()
                nelem = (size + align - 1) // align * align - param_offset
                assert nelem > 0, "Expect the position locates in the padding."
                fp32_pointer += nelem

            if fp32_pointer >= rank_nelem:
                return

            for p in fp16_params[params_start_idx+1:]:
                size = min(p.numel(), rank_nelem - fp32_pointer)
                fp32_params[fp32_pointer:fp32_pointer+size].copy_(
                    p.view(-1)[:size])
                nelem = (size + align - 1) // align * align
                fp32_pointer += nelem

                if fp32_pointer >= rank_nelem:
                    return
            else:
                raise ValueError("Expect copy of real data.")

    def flatten_fp16_grads_(self, fp16_grads, params, align):
        # Copy the model's FP16 gradients to the flattened gradients
        pointer = 0
        for p in params:
            fp16_grads[pointer:pointer+p.numel()].copy_(p.grad.view(-1))
            nelem = (p.numel() + align - 1) // align * align
            pointer += nelem

    def build_norm_buffers_(self, world_size):
        norms = []
        for i in range(world_size):
            norms.append(torch.empty(1, dtype=torch.float, device='cuda'))
        return norms

    def build_fp16_grads_chunks_(self, fp16_grads, align, world_size, rank_nelem):
        # Build chunks of FP16 gradients
        # Share the same storage as the flattened gradients
        assert fp16_grads.numel() == world_size * rank_nelem, \
            "Invalid gradient size."

        pointer = 0
        buffers = []
        with torch.no_grad():
            for i in range(world_size):
                g = fp16_grads.new_empty((rank_nelem))
                g.set_(source=fp16_grads.storage(), storage_offset=pointer,
                    size=g.size())
                pointer += g.numel()
                buffers.append(g)

        return buffers

    def build_all_gather_weights_(self, fp16_params, align, world_size, rank_nelem):
        # Build buffer for all gather FP16 weights
        # Share the same storage as the flattened weights
        assert fp16_params.numel() == world_size * rank_nelem, \
            "Invalid parameter size."

        pointer = 0
        buffers = []
        with torch.no_grad():
            for i in range(world_size):
                p = fp16_params.new_empty((rank_nelem))
                p.set_(source=fp16_params.storage(), storage_offset=pointer,
                    size=p.size())
                pointer += p.numel()
                buffers.append(p)

        return buffers

    def grad_norm(self, norm_buffer, fp16_grads_list, rank, group=None):
        # Get the norm of gradients distributed in the process group
        total_norm = 0
        norm_type = 2
        clip_coef = 1.0

        if hasattr(fp16_grads_list, 'norm'):
            return fp16_grads_list.norm(norm_type)

        norm = fp16_grads_list[rank].norm(norm_type)
        norm_buffer[rank][0] = norm

        # All-gather other ranks' norm
        if group is not None:
            torch.distributed.all_gather(norm_buffer,
                norm_buffer[rank], group=group, async_op=False)
        else:
            torch.distributed.all_gather(norm_buffer,
                norm_buffer[rank], async_op=False)

        for p in norm_buffer:
            total_norm += p.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

        return total_norm

    def step(self, **args):
        raise NotImplementedError

class FullyDistributedOptimizer(BasicDistributedOptimizer):
    """
    Distributed optimizer with mixed precision for PyTorch.
    Distributed strategy:
    1. reduce-scatter gradients among all ranks;
    2. weight update;
    3. all-gather weights among all ranks;
    """
    def __init__(self, params, optimizer, grad_clip=None, align=64, **args):
        super(FullyDistributedOptimizer, self).__init__(params,
            optimizer, grad_clip, align)

        # Round up alignment to LCM of world size and CUDA performance requirement
        self.align = abs(self.world_size * align) // math.gcd(self.world_size, align)

        # Flatten model's FP16 parameters, self.fp16_grads must be flattened
        # each time bprop is finish
        self.fp16_params, self.fp16_grads, self.nelem = \
            self.flatten_params_(params, self.align)
        self.rank_nelem = self.nelem // self.world_size

        # Locate rank start, offset with respect to total parameters
        self.params_start_idx, self.param_offset = \
            self.locate_rank_elem_(params, self.align, self.rank, self.rank_nelem)

        # Create per rank local FP32 master parameters and initialize
        self.fp32_params = torch.zeros(self.rank_nelem, dtype=torch.float,
            device='cuda')
        self.initialize_fp32_params(self.fp32_params, params, self.params_start_idx,
            self.param_offset, self.align, self.rank_nelem)

        # Create FP16 gradient buffer for reduce-scatter
        self.fp16_grads_list = self.build_fp16_grads_chunks_(self.fp16_grads,
            self.align, self.world_size, self.rank_nelem)

        # Create real optimizer
        self.optimizer = optimizer([self.fp32_params], **args)

        # Create buffers for all-gather norm
        if self.grad_clip:
            self.norms = self.build_norm_buffers_(self.world_size)

        # Create list of tensors for all-gather FP6 weights
        # The buffer for gather share the same storage as flattened FP16 weight
        self.fp16_params_list = self.build_all_gather_weights_(self.fp16_params,
            self.align, self.world_size, self.rank_nelem)

    def step(self, **args):
        # Flatten the model's fp16 gradients
        self.flatten_fp16_grads_(self.fp16_grads, self.params, self.align)

        # Reduce-scatter fp16 gradients
        torch.distributed.reduce_scatter(self.fp16_grads_list[self.rank],
            self.fp16_grads_list, async_op=False)

        # Collect gradient norm if need gradient clipping
        if self.grad_clip:
            total_norm = self.grad_norm(self.norms, self.fp16_grads_list, self.rank)
            clip_coef = self.grad_clip / (total_norm + 1e-6)
            clip_coef = min(clip_coef, 1)

            # Skill step if norm is illegal
            if not math.isfinite(total_norm):
                return False

        # Step optimizer with scale
        args['scale'] /= clip_coef
        args['grads'] = [self.fp16_grads_list[self.rank]]
        args['output_params'] = [self.fp16_params_list[self.rank]]
        self.optimizer.step(**args)

        # All gather FP16 parameters
        # Since the flattened FP16 parameters and model parameters share the
        # same storage, all_gather is the last step here
        torch.distributed.all_gather(self.fp16_params_list,
            self.fp16_params_list[self.rank], async_op=False)

        return True

class IntraNodeDistributedOptimizer(BasicDistributedOptimizer):
    """
    Intra-node distributed optimizer with mixed precision for PyTorch.
    Distributed strategy:
    1. reduce-scatter gradients among all ranks;
    2. number of devices parallel inter-node all-gather;
    3. weight update of 1 / devices portion;
    4. intra-node all-gather weights;
    """
    def __init__(self, params, optimizer, grad_clip=None, align=64, **args):
        super(IntraNodeDistributedOptimizer, self).__init__(params,
            optimizer, grad_clip, align)

        # Create process group for ranks with the same local rank
        self.device_pg = []
        if self.nodes > 1:
            for  i in range(self.devices):
                self.device_pg.append(torch.distributed.new_group(ranks=
                    list(range(i, self.world_size, self.devices))))

        # Create process group for ranks within the same node
        self.node_pg = []
        for i in range(self.nodes):
            self.node_pg.append(torch.distributed.new_group(ranks=
                list(range(i * self.devices, (i + 1) * self.devices))))

        # Align to LCM of (nodes, devices, CUDA performance requirement)
        self.align = abs(self.nodes * self.devices * align) // \
            math.gcd(math.gcd(self.nodes, self.devices), align)

        # Flatten model's FP16 parameters, self.fp16_grads must be flattened
        # each time bprop is finish
        self.fp16_params, self.fp16_grads, self.nelem = \
            self.flatten_params_(params, self.align)

        # Locate rank start, offset with respect to total parameters
        self.rank_nelem = self.nelem // self.devices
        self.params_start_idx, self.param_offset = \
            self.locate_rank_elem_(params, self.align, self.device_rank, self.rank_nelem)

        # Create per rank local FP32 master parameters and initialize
        self.fp32_params = torch.zeros(self.rank_nelem, dtype=torch.float,
            device='cuda')
        self.initialize_fp32_params(self.fp32_params, params, self.params_start_idx,
            self.param_offset, self.align, self.rank_nelem)

        # Create FP16 gradient buffer for reduce-scatter
        self.fp16_grads_list_global = self.build_fp16_grads_chunks_(self.fp16_grads,
            self.align, self.world_size, self.nelem // self.world_size)

        # Create FP16 gradient buffer for all-gather
        self.fp16_grads_list_device = self.build_fp16_grads_chunks_(self.fp16_grads,
            self.align, self.devices, self.rank_nelem)

        # Create real optimizer
        self.optimizer = optimizer([self.fp32_params], **args)

        # Create buffers for all-gather norm
        if self.grad_clip:
            self.norms = self.build_norm_buffers_(self.devices)

        # Create list of tensors for all-gather FP6 weights
        # The buffer for gather share the same storage as flattened FP16 weight
        self.fp16_params_list = self.build_all_gather_weights_(self.fp16_params,
            self.align, self.devices, self.rank_nelem)

    def step(self, **args):
        # Flatten the model's fp16 gradients
        self.flatten_fp16_grads_(self.fp16_grads, self.params, self.align)

        # Reduce-scatter fp16 gradients among all ranks
        torch.distributed.reduce_scatter(self.fp16_grads_list_global[self.rank],
            self.fp16_grads_list_global, async_op=False)

        # All-gather fp16 gradients within the same local ranks
        if self.nodes > 1:
            torch.distributed.all_gather(self.fp16_grads_list_global[
                self.device_rank : self.world_size : self.nodes],
                self.fp16_grads_list_global[self.rank],
                group=self.device_pg[self.device_rank], async_op=False)

        # Collect gradient norm if need gradient clipping
        if self.grad_clip:
            total_norm = self.grad_norm(self.norms, self.fp16_grads_list_device,
                self.device_rank, group=self.node_pg[self.node_rank])
            clip_coef = self.grad_clip / (total_norm + 1e-6)
            clip_coef = min(clip_coef, 1)

            # Skill step if norm is illegal
            if not math.isfinite(total_norm):
                return False

        # Step optimizer with scale
        args['scale'] /= clip_coef
        args['grads'] = [self.fp16_grads_list_device[self.device_rank]]
        args['output_params'] = [self.fp16_params_list[self.device_rank]]
        self.optimizer.step(**args)

        # All gather FP16 parameters
        # Since the flattened FP16 parameters and model parameters share the
        # same storage, all_gather is the last step here
        torch.distributed.all_gather(self.fp16_params_list,
            self.fp16_params_list[self.device_rank], group=self.node_pg[self.node_rank],
            async_op=False)

        return True

class IntraNodeAcceleratedOptimizer(BasicDistributedOptimizer):
    """
    Intra-node accelerated optimizer with mixed precision for PyTorch.
    Distributed strategy:
    1. all-reduce gradients among all ranks;
    2. weight update with 1 / devices protion;
    3. intra-node all-gather weights;
    """
    def __init__(self, params, optimizer, grad_clip=None, align=64, **args):
        super(IntraNodeAcceleratedOptimizer, self).__init__(params,
            optimizer, grad_clip, align)

        # Create process group for ranks within the same node
        self.node_pg = []
        for i in range(self.nodes):
            self.node_pg.append(torch.distributed.new_group(ranks=
                list(range(i * self.devices, (i + 1) * self.devices))))

        # Align to LCM of (nodes, devices, CUDA performance requirement)
        self.align = abs(self.nodes * self.devices * align) // \
            math.gcd(math.gcd(self.nodes, self.devices), align)

        # Flatten model's FP16 parameters, self.fp16_grads must be flattened
        # each time bprop is finish
        self.fp16_params, self.fp16_grads, self.nelem = \
            self.flatten_params_(params, self.align)

        # Locate rank start, offset with respect to total parameters
        self.rank_nelem = self.nelem // self.devices
        self.params_start_idx, self.param_offset = \
            self.locate_rank_elem_(params, self.align, self.device_rank, self.rank_nelem)

        # Create per rank local FP32 master parameters and initialize
        self.fp32_params = torch.zeros(self.rank_nelem, dtype=torch.float,
            device='cuda')
        self.initialize_fp32_params(self.fp32_params, params, self.params_start_idx,
            self.param_offset, self.align, self.rank_nelem)

        # Create FP16 gradient buffer for all-gather
        self.fp16_grads_list = self.build_fp16_grads_chunks_(self.fp16_grads,
            self.align, self.devices, self.rank_nelem)

        # Create real optimizer
        self.optimizer = optimizer([self.fp32_params], **args)

        # Create list of tensors for all-gather FP6 weights
        # The buffer for gather share the same storage as flattened FP16 weight
        self.fp16_params_list = self.build_all_gather_weights_(self.fp16_params,
            self.align, self.devices, self.rank_nelem)

    def step(self, **args):
        # Flatten the model's fp16 gradients
        self.flatten_fp16_grads_(self.fp16_grads, self.params, self.align)

        # All-reduce fp16 gradients
        torch.distributed.all_reduce(self.fp16_grads, async_op=False)

        # Collect gradient norm if need gradient clipping
        # Since each rank contains whole gradients, we don't use distributed
        # norm here
        if self.grad_clip:
            total_norm = self.grad_norm([], self.fp16_grads,
                self.device_rank, group=self.node_pg[self.node_rank])
            clip_coef = self.grad_clip / (total_norm + 1e-6)
            clip_coef = min(clip_coef, 1)

            # Skill step if norm is illegal
            if not math.isfinite(total_norm):
                return False

        # Step optimizer with scale
        args['scale'] /= clip_coef
        args['grads'] = [self.fp16_grads_list[self.device_rank]]
        args['output_params'] = [self.fp16_params_list[self.device_rank]]
        self.optimizer.step(**args)

        # All gather FP16 parameters
        # Since the flattened FP16 parameters and model parameters share the
        # same storage, all_gather is the last step here
        torch.distributed.all_gather(self.fp16_params_list,
            self.fp16_params_list[self.device_rank], group=self.node_pg[self.node_rank],
            async_op=False)

        return True

class HierarchicalDistributedOptimizer(BasicDistributedOptimizer):
    """
    Hierarchical distributed optimizer with mixed precision for PyTorch.
    Distributed strategy:
    1. intra-node reduce-scatter gradients;
    2. number of devices parallel inter-node all-reduce;
    3. weight update;
    4. intra-node all-gather weights;
    """
    def __init__(self, params, optimizer, grad_clip=None, align=64, **args):
        super(HierarchicalDistributedOptimizer, self).__init__(params,
            optimizer, grad_clip, align)

        # Create process group for ranks with the same local rank
        self.device_pg = []
        if self.nodes > 1:
            for  i in range(self.devices):
                self.device_pg.append(torch.distributed.new_group(ranks=
                    list(range(i, self.world_size, self.devices))))

        # Create process group for ranks within the same node
        self.node_pg = []
        for i in range(self.nodes):
            self.node_pg.append(torch.distributed.new_group(ranks=
                list(range(i * self.devices, (i + 1) * self.devices))))

        # Align to LCM of (nodes, devices, CUDA performance requirement)
        self.align = abs(self.nodes * self.devices * align) // \
            math.gcd(math.gcd(self.nodes, self.devices), align)

        # Flatten model's FP16 parameters, self.fp16_grads must be flattened
        # each time bprop is finish
        self.fp16_params, self.fp16_grads, self.nelem = \
            self.flatten_params_(params, self.align)

        # Locate rank start, offset with respect to total parameters
        self.rank_nelem = self.nelem // self.devices
        self.params_start_idx, self.param_offset = \
            self.locate_rank_elem_(params, self.align, self.device_rank, self.rank_nelem)

        # Create per rank local FP32 master parameters and initialize
        self.fp32_params = torch.zeros(self.rank_nelem, dtype=torch.float,
            device='cuda')
        self.initialize_fp32_params(self.fp32_params, params, self.params_start_idx,
            self.param_offset, self.align, self.rank_nelem)

        # Create FP16 gradient buffer for reduce-scatter
        self.fp16_grads_list = self.build_fp16_grads_chunks_(self.fp16_grads,
            self.align, self.devices, self.rank_nelem)

        # Create real optimizer
        self.optimizer = optimizer([self.fp32_params], **args)

        # Create buffers for all-gather norm
        if self.grad_clip:
            self.norms = self.build_norm_buffers_(self.devices)

        # Create list of tensors for all-gather FP6 weights
        # The buffer for gather share the same storage as flattened FP16 weight
        self.fp16_params_list = self.build_all_gather_weights_(self.fp16_params,
            self.align, self.devices, self.rank_nelem)

    def step(self, **args):
        # Flatten the model's fp16 gradients
        self.flatten_fp16_grads_(self.fp16_grads, self.params, self.align)

        # Intra-node reduce-scatter fp16 gradients
        torch.distributed.reduce_scatter(self.fp16_grads_list[self.device_rank],
            self.fp16_grads_list, group=self.node_pg[self.node_rank], async_op=False)

        # Inter-node all-reduce fp16 gradients
        if self.nodes > 1:
            # FIXME: instead of making reduce_scatter sync_op, we should have a barrier
            # among device_pg[device_rank] here, but that somehow doesn't work
            torch.distributed.all_reduce(self.fp16_grads_list[self.device_rank],
                group=self.device_pg[self.device_rank], async_op=False)
        # FIXME: instead of making all_reduce sync_op, we should have a barrier
        # among node_pg[node_rank] here, but that somehow doesn't work

        # Collect gradient norm if need gradient clipping
        if self.grad_clip:
            total_norm = self.grad_norm(self.norms, self.fp16_grads_list,
                self.device_rank, group=self.node_pg[self.node_rank])
            clip_coef = self.grad_clip / (total_norm + 1e-6)
            clip_coef = min(clip_coef, 1)

            # Skill step if norm is illegal
            if not math.isfinite(total_norm):
                return False

        # Step optimizer with scale
        args['scale'] /= clip_coef
        args['grads'] = [self.fp16_grads_list[self.device_rank]]
        args['output_params'] = [self.fp16_params_list[self.device_rank]]
        self.optimizer.step(**args)

        # All gather FP16 parameters
        # Since the flattened FP16 parameters and model parameters share the
        # same storage, all_gather is the last step here
        torch.distributed.all_gather(self.fp16_params_list,
            self.fp16_params_list[self.device_rank], group=self.node_pg[self.node_rank],
            async_op=False)

        return True

