import math
import torch
import torch.nn as nn
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

import copy
from torch.func import stack_module_state, functional_call
class NbModel(nn.Module):
    def __init__(self, n_basis, dim_hidden, n_layers):
        super().__init__()
        self.n_basis = n_basis
        self.n_hidden = dim_hidden


        self.sirens = nn.ModuleList([
            SirenNet(
                dim_in=4, # x and y both are have 2d coords
                dim_hidden=dim_hidden,
                dim_out=1, num_layers=n_layers,
                w0_initial= i/12 + 1
            )
            for i in range(2*n_basis**2 - 1)
        ])

    # TODO: parallelize this! 
    def torus2basis(self, torus):
        out = []
        for i, siren in enumerate(self.sirens):
            out.append(siren(torus)[..., 0])
        return torch.stack(out, dim=0)

    def forward(self, x, plot):
        b, c, h, w = x.shape
        k = self.n_basis

        # each image gets 2 groups of k**2 basis functions
        line_x = torch.linspace(0, 1, 32).to(x.device)
        line_y = torch.linspace(0, 1, 32).to(x.device)
        grid_x, grid_y = torch.meshgrid(line_x, line_y, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)

        # map [0,1] to x,y unit torus coords (for nn continuity)
        torus_grid = unit_circle(grid)
        torus_grid = repeat(torus_grid, 'h w d -> c h w d', c=3)

        # get basis given domain provided
        stack = rearrange(torus_grid, 'c h w d -> (c h w) d')
        stack_basis = self.torus2basis(stack)
        basis = rearrange(stack_basis, 'k (c h w) -> k c h w', c=c, h=h, w=w)

        if plot: plot_basis(basis.detach().cpu())

        # get coeffs
        coeffs = torch.einsum('k c h w, b c h w -> b k c', basis, x)

        # reconstruct image
        y = torch.einsum('b k c, k c h w -> b c h w', coeffs, basis)
        return y

    # we want a complete biorthonormal basis: https://mathworld.wolfram.com/GeneralizedFourierSeries.html
    def orthon_sample(self, n_samples, device='cuda'):
        # sample random point in [0,1]
        x = torch.rand(n_samples, 2).to(device)
        torus_x = unit_circle(x)

        # get output of each basis function
        basis = self.torus2basis(torus_x)

        # each basis filter should sum to 1
        integral = basis.sum(dim=(1)).abs() / basis.shape[1]

        # inner prods should also be 0
        inner_prod = (basis @ basis.T).abs() / basis.shape[1]
        return integral.mean() + inner_prod.mean()
