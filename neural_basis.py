import torch
import torch.nn as nn
from torch.optim import Adam
from einops import rearrange, repeat
from siren_pytorch import SirenNet

from plot import plot_basis

class NbModel(nn.Module):
    def __init__(self, n_basis, dim_hidden):
        super().__init__()
        self.n_basis = n_basis
        self.n_hidden = dim_hidden

        self.siren = SirenNet(
            dim_in = 2,
            dim_hidden = dim_hidden,
            dim_out = n_basis**2,
            num_layers = 5,
            w0_initial = 30.
        )

    def forward(self, x, plot=False):
        bs = x.shape[0]
        line = torch.linspace(-1, 1, 32)
        plane = torch.meshgrid(line, line)
        plane = torch.stack(plane, dim=-1)

        plane = rearrange(plane, 'h w d -> (h w) d')
        basis_ = self.siren(plane)
        basis = rearrange(basis_, '(h w) k -> k h w', h=32)[None, :, None, ...]
        basis = repeat(basis, '1 k 1 h w -> b k c h w', b=bs, c=3)

        if plot: plot_basis(basis[0,:,0])

        # find optimal coeffients for basis 
        A = self.coeff_optim(basis.clone().detach(), x)
        y = (A[..., None, None] * basis).sum(dim=1)

        return y, rearrange(basis_, 'n k -> k n')
    
    def coeff_optim(self, basis, x):
        A = torch.ones(basis.shape[:3], requires_grad=True)
        optim = Adam([A], lr=1e-1)

        for _ in range(100):
            # multiply basis by coefficients
            y = A[..., None, None] * basis
            y = y.sum(dim=1)

            loss = (x - y).pow(2).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

        return A.detach()