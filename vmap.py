import time
import torch 
import torch.nn as nn
import copy

class ResNet(torch.nn.Module):
    def __init__(self, dim_hidden=64, num_layers=3):
        super().__init__()
        '''
        a small resnet that maps [128] -> [2]
        '''
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        self.first = nn.Linear(2, self.dim_hidden)

        self.res_block = nn.Sequential(
            nn.Linear(self.dim_hidden, self.dim_hidden),
            nn.ELU(),
            nn.Linear(self.dim_hidden, self.dim_hidden),
        ) 

        self.last = nn.Linear(self.dim_hidden, 2)

        self.blocks = nn.ModuleList([copy.deepcopy(self.res_block) for _ in range(3)])

    def forward(self, x):
        x = self.first(x)
        for block in self.blocks:
            x = x + block(x)
        x = self.last(x)
        return x

from torch.func import stack_module_state, functional_call

if __name__ == '__main__':
    n = 100
    a = torch.randn(n, 5, 128)
    b = torch.randn(5, 128)

    model = ResNet()

    t0 = time.time()
    out = model(b)
    t1 = time.time() - t0
    print(f'single forward: {t1:.6f}')


    nets = [model for _ in range(n)]
    params, buffers = stack_module_state(nets)

    base = copy.deepcopy(nets[0])
    base = base.to('meta')

    def fmodel(params, buffers, a):
        return functional_call(base, (params, buffers), (a,))

    t0 = time.time()
    out = torch.vmap(fmodel)(params, buffers, a)
    t2 = time.time() - t0
    print(f'vmap forward: {t2:.6f}')
    print(f'slowdown: {t2/t1:.6f}x for n={n}')
