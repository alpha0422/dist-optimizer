import torch

class DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, model, optimizer, grad_clip=None, **args):
        assert torch.distributed.is_initialized(), \
            "Default process group hasn't been initialized."
        assert grad_clip is None or isinstance(grad_clip, float) or \
            callable(grad_clip), "Invalid gradient clipping argument."
        assert isinstance(optimizer, torch.optim.Optimizer), \
            "Expect instance of torch.optim.Optimizer."
        super(DistributedOptimizer, self).__init__(params, args)

    def flatten_params_(self):
        pass

    def flatten_fp16_grads_(self):
        pass

    def step(self, **args):
        pass

