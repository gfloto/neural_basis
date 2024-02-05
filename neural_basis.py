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
            dim_in = 1,
            dim_hidden = dim_hidden,
            dim_out = n_basis,
            num_layers = 12,
            w0_initial = 30.
        )

    def forward(self, x, plot=False):
        bs = x.shape[0]
        line = torch.linspace(-1, 1, 32)[..., None].to(x.device)
        basis_line = self.siren(line)

        # make into plane combining all combination of basis 
        (w, q) = basis_line.shape
        bl_x = repeat(basis_line[None, None, ...], '1 1 h k -> k q h w', w=w, q=q)
        bl_y = repeat(basis_line[None, None, ...], '1 1 h k -> q k w h', w=w, q=q)
        basis = rearrange(bl_x + bl_y, 'k q h w -> (k q) h w')

        # repeat along batch and color channels
        basis = repeat(basis[None, :, None, ...], '1 k 1 h w -> b k c h w', b=bs, c=3)
        if plot: plot_basis(basis[0,:,0], 'basis.png')

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

    def orthon_sample(self, n_samples, device='cuda'):
        # sample random point in [0,1]
        x = torch.rand(n_samples)[..., None].to(device)
        y = rearrange(self.siren(x), 'n k -> k n')

        # compute inner product of all pairs
        ip = (y @ y.T) / n_samples
        triu_ip = torch.triu(ip, diagonal=0)

        # orthonormally coresponds to identity
        tgt = torch.eye(triu_ip.shape[0]).to(device)
        orthon_loss = (triu_ip - tgt).pow(2).mean()

        return orthon_loss
