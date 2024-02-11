import math
import copy
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

class NbModel(nn.Module):
    def __init__(self, n_basis, dim_hidden, n_layers):
        super().__init__()
        self.n_basis = n_basis
        self.n_hidden = dim_hidden

<<<<<<< HEAD
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
=======

        self.sirens = nn.ModuleList([
            SirenNet(
                dim_in=2,
                dim_hidden=dim_hidden,
                dim_out=1, num_layers=1,
                w0_initial= (i % n_basis**2) + 2
            )
            for i in range(2*n_basis**2)
        ])
>>>>>>> origin/main

    # TODO: parallelize this! 
    def circ2basis(self, circle, grad=False):
        out = []
        if grad:
            for i, siren in enumerate(self.sirens):
                out.append(siren(circle[i]))
        else:
            with torch.no_grad():
                for i, siren in enumerate(self.sirens):
                    out.append(siren(circle[i]))
        return torch.stack(out, dim=0)

    def forward(self, x, mag, shift, plot):
        y = self.neural_recon(x, mag, shift, plot, grad=True)
        return y
    
    def neural_recon(self, x, mag, shift, plot=False, grad=False):
        b, c, h, w = x.shape
        k = self.n_basis

        # each image gets 2 groups of k**2 basis functions
        # the group are vertically and horizontally oriented and added
        line = torch.linspace(0, 1, 32).to(x.device)
        lines_w = repeat(line, 'w -> b k2 c w', b=b, k2=k**2, c=c)
        lines_h = repeat(line, 'h -> b k2 c h', b=b, k2=k**2, c=c)
        lines = torch.cat([lines_w, lines_h], dim=1)

        # cyclically shift each line independently
        pre = (lines + shift[..., None]) % 1
        pre = rearrange(pre, 'b n c f -> n (b c f) 1')

        # map [0,1] to x,y unit circle coords (for nn continuity)
        circle = unit_circle(pre.clone())

        # get basis given domain provided
        basis_line = self.circ2basis(circle, grad)

        # add vertical and horizontal basis
        basis_stack = rearrange(basis_line, 'n (b c f) 1 -> b n c f', b=b, c=c, f=h)
        basis_w = basis_stack[:, :k**2]; basis_h = basis_stack[:, k**2:]

        # create vertically and horizontally oriented 2d basis
        basis_w = repeat(basis_w, 'b k c w -> b k c h w', h=h, w=w)
        basis_h = repeat(basis_h, 'b k c h -> b k c h w', h=h, w=w)

        basis = basis_w + basis_h
        if plot: plot_basis(basis[0], 'basis.png')

        # scale basis
        scaled_basis = basis * mag[..., None, None]
        y = scaled_basis.sum(dim=1)
        return y

    def orthon_sample(self, n_samples, device='cuda', plot=False):
        # sample random point in [0,1]
        x = torch.rand(n_samples)[..., None].to(device)
        x = repeat(x, 'n 1 -> k n 1', k=2*self.n_basis**2)

        circle = unit_circle(x.clone())
        y = self.circ2basis(circle, grad=True)

        x = x[..., -1]; y = y[..., -1]
        y_w = y[:self.n_basis**2]; y_h = y[self.n_basis**2:]

        if plot: plot_line(x[0].detach(), y_w.detach(), y_h.detach(), 'line.png')

        # compute inner product of all pairs
        ip = ( (y_w @ y_w.T) + (y_h @ y_h.T) ) / n_samples 
        triu_ip = torch.triu(ip, diagonal=0)

        # orthonormally coresponds to identity
        orthon_loss = triu_ip.pow(2).mean()
        return orthon_loss

    def coeff_optim(self, x):
        b, c, h, w = x.shape
        k = self.n_basis

        # params to shift and scale basis
        # we want to cycle shift each dim (think a torus)
        mag = torch.randn(b, k**2, c).to(x.device) / (2*k**2)
        shift = torch.rand(b, 2*k**2, c).to(x.device)
        print(mag.shape, shift.shape)
        quit()

        # optim
        mag = nn.Parameter(mag)
        shift = nn.Parameter(shift)
        optim = Adam([mag, shift], lr=1e-2)

        # get optimal shift and scale to align basis 
        for _ in range(100):
            y = self.neural_recon(x, mag, shift)

            loss = (x - y).pow(2).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

        return mag.detach(), shift.detach()