import math
import copy
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.func import stack_module_state, functional_call

from functools import partial
from einops import rearrange, repeat
from siren_pytorch import SirenNet

from plot import plot_line
from plot import plot_basis

def unit_circle(angle):
    '''
    map line in [0,1] to x, y coordinates of unit circle 
    '''

    angle *= 2 * math.pi
    x = torch.cos(angle)
    y = torch.sin(angle)

    return torch.cat([x, y], dim=-1)


class NbModel(nn.Module):
    def __init__(self, n_basis, dim_hidden):
        super().__init__()
        self.n_basis = n_basis
        self.n_hidden = dim_hidden

        net = partial(
            SirenNet,
            dim_in=2,
            dim_hidden=dim_hidden,
            dim_out=1,
            num_layers=4,
            w0_initial=30.,
        )
        self.sirens = nn.ModuleList([net() for _ in range(2*n_basis**2)])

        #self.sirens = [net() for _ in range(2*n_basis**2)]
        #self.params, self.buffers = stack_module_state(self.sirens)

        #print('hmm')
        #a = torch.randn(5, 2)
        #b = torch.randn(2*n_basis**2, 5, 2)

        #print(self.sirens[0](a).shape)
        #print(self.pmap(b).shape)
    
    #def pmap(self, x):
        #return torch.vmap(self.v_model)(self.params, self.buffers, x)

    #def v_model(self, params, buffers, x):
        #base = copy.deepcopy(self.sirens[0]).to('meta')
        #return functional_call(base, (params, buffers), (x,))

    def forward(self, lines):
        print(lines)
        self.sirens(lines)
        quit()

        return y

    # TODO: parallelize this! 
    def circ2basis(self, circle):
        out = []
        for i, siren in enumerate(self.sirens):
            out.append(siren(circle[i]))
        return torch.stack(out, dim=0)
    
    def coeff_optim(self, x):
        b, c, h, w = x.shape
        k = self.n_basis

        # params to shift and scale basis
        # we want to cycle shift each dim (think a torus)
        mag = torch.randn(b, k**2, c).to(x.device)
        shift = torch.rand(b, 2*k**2, c).to(x.device)
        print(shift.shape)

        #A = nn.Parameter(A)
        #optim = Adam([A], lr=1e-1)

        # stack of rows [0,1], we want 2 overlapping filters for n-basis^2 grid
        # 1st filter is horizontal, 2nd is vertical (or vice versa?)
        line = torch.linspace(0, 1, 32).to(x.device)
        lines_w = repeat(line, 'w -> b k2 c h w', b=b, k2=k**2, c=c, h=h)
        lines_h = repeat(line, 'h -> b k2 c h w', b=b, k2=k**2, c=c, w=w)
        lines = torch.cat([lines_w, lines_h], dim=1)
        print(lines.shape)
        print(shift.shape)
        quit()

        # cyclically shift each row
        pre = (lines + shift[..., None, None]) % 1
        pre = rearrange(pre, 'b n c h w -> n (b c h w) 1')
        circle = unit_circle(pre)

        # get basis given domain provided
        with torch.no_grad():
            basis_line = self.circ2basis(circle) 

        # add vertical and horizontal basis
        basis_stack = rearrange(basis_line, 'n (b c h w) 1 -> b n c h w', b=b, c=c, h=h, w=w)
        basis_w = basis_stack[:, :k**2]; basis_h = basis_stack[:, k**2:]
        basis = basis_w + basis_h



        # send all to cuda
        import time
        t0 = time.time()
        for _ in range(50):
            # multiply basis by coefficients
            y = A * basis
            y = y.sum(dim=1)

            loss = (x - y).pow(2).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()
        print(time.time()-t0)
        quit()

        return A.detach()

    def orthon_sample(self, n_samples, device='cuda', plot=False):
        # sample random point in [0,1]
        x = torch.rand(n_samples)[..., None].to(device)
        circle = unit_circle(x)
        y = rearrange(self.siren(circle), 'n k -> k n')
        x = rearrange(x, 'n k -> k n')

        if plot: plot_line(x.detach(), y.detach(), 'line.png')

        # compute inner product of all pairs
        ip = (y @ y.T) / n_samples
        triu_ip = torch.triu(ip, diagonal=0)

        # orthonormally coresponds to identity
        tgt = torch.eye(triu_ip.shape[0]).to(device)
        orthon_loss = (triu_ip - tgt).pow(2).mean()

        return orthon_loss
