import torch
import torch.nn as nn
import math
from torch.optim import Adam
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

        self.siren = SirenNet(
            dim_in = 2,
            dim_hidden = dim_hidden,
            dim_out = n_basis,
            num_layers = 4,
            w0_initial = 30.
        )

    def forward(self, x, plot=False):
        bs = x.shape[0]
        line = torch.linspace(0, 1, 32)[..., None].to(x.device)
        print(line.shape)
        quit()

        circle = unit_circle(line)
        basis_line = self.siren(circle)

        # make into plane combining all combination of basis 
        (w, q) = basis_line.shape
        bl_x = repeat(basis_line[None, None, ...], '1 1 h k -> k q h w', w=w, q=q)
        bl_y = repeat(basis_line[None, None, ...], '1 1 h k -> q k w h', w=w, q=q)
        basis = rearrange(bl_x + bl_y, 'k q h w -> (k q) h w')

        # repeat along batch and color channels
        basis = repeat(basis[None, :, None, ...], '1 k 1 h w -> b k c h w', b=bs, c=3)
        if plot: plot_basis(basis[0,:,0], basis_line, 'basis.png')

        # find optimal coeffients for basis 
        A = self.coeff_optim(
            basis.detach().clone().cpu(),
            x.detach().clone().cpu()
        ).to(x.device)
        y = (A * basis).sum(dim=1)

        return y
    
    def coeff_optim(self, basis, x):
        A = torch.ones(basis.shape[:3]) / basis.shape[1] 
        A = A[..., None, None]
        A = nn.Parameter(A)
        optim = Adam([A], lr=1e-1)

        # send all to cuda
        for _ in range(50):
            # multiply basis by coefficients
            y = A * basis
            y = y.sum(dim=1)

            loss = (x - y).pow(2).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

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
