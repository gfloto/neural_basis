import os
import math
import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat

from plot import basis_video, video_compare 
from models.siren import EigenFunction

def func_inner(x, y):
    assert x.shape == y.shape
    return torch.einsum('... h w, ... h w -> ...', x, y)

def func_norm(x):
    return func_inner(x, x).sqrt()

def make_domain(h, w):
    xh = torch.linspace(0, 1, h).cuda()
    xw = torch.linspace(0, 1, w).cuda()
    xh, xw = torch.meshgrid(xh, xw, indexing='ij')
    dom = torch.stack([xh, xw], dim=-1) 

    t = torch.linspace(0, 1, 50).cuda()

    dom = repeat(dom, 'h w c -> t h w c', t=t.shape[0])
    t = repeat(t, 't -> t h w 1', h=h, w=w)

    return torch.cat([dom, t], dim=-1)

# take residual of x by removing
# contributions from last learned basis components
def basis_residual(x, basis):
    coeffs = torch.einsum('b t h w, t e h w -> b t e', x, basis)
    recon = torch.einsum('b t e, t e h w -> b t h w', coeffs, basis)

    return x - recon, coeffs

# ensure that new eigen-function is orthogonal to previous basis
def orthogonalize(eigen_f, basis):
    coeff = torch.einsum('t h w, t e h w -> t e', eigen_f, basis)
    recon = torch.einsum('t e, t e h w -> t h w', coeff, basis)
    eigen_f = eigen_f - recon

    # checkthat eigen_f is orthogonal to basis
    #ef = repeat(eigen_f, 't h w -> t e h w', e=basis.shape[1])
    #ip = func_inner(ef, basis)
    #assert torch.allclose(ip, torch.zeros_like(ip), atol=1e-5)

    return eigen_f

def save_basis(eigen_f, basis, path):
    eigen_f = eigen_f.detach()[:, None]
    new_basis = torch.cat([basis, eigen_f], dim=1)
    torch.save(new_basis, f'{path}/basis.pt')

    return new_basis

def fourier(x, n_basis):
    k = math.ceil( (n_basis/2) ** 0.5)

    coeffs = torch.fft.fft2(x)
    coeffs[..., k:, :] = 0
    coeffs[..., :, k:] = 0

    recon = torch.fft.ifft2(coeffs).real
    return recon

class LegendreBasis:
    def __init__(self, n=32, T=50):
        self.n = n# number of basis functions (different from n_basis for spatial basis)
        self.T = T
        self.basis = self.make_basis().cuda()

    def make_basis(self):
        x = torch.linspace(-3/4, 3/4, self.T)
        basis = torch.zeros((self.n, self.T))
        basis[0] = 1.
        basis[1] = x

        for i in range(2, self.n):
            basis[i] = ((2*i - 1) * x * basis[i-1] - (i-1) * basis[i-2]) / i
        
        return basis

    def recon_error(self, x, eigen_f):
        # get eigen-values of x from most recent eigen-function
        c = torch.einsum('b t h w, b t h w -> b t', x, eigen_f)
        c = (c - c.mean()) / c.std()

        # ensure that eigen-value function through time is well
        # represented by the legendre basis
        coeffs = torch.einsum('b t, l t -> b l', c, self.basis)
        recon = torch.einsum('b l, l t -> b t', coeffs, self.basis)

        return c - recon

def train(siren_model, loader, optim, basis, basis_num, hps): 
    siren_model.train()
    siren_model = siren_model.cuda()

    leg_basis = LegendreBasis()

    h, w = 64, 64 
    bs = hps.batch_size

    # domain of eigen-function
    dom = make_domain(h, w).cuda()

    for _ in range(10):
        inner_loss_t, reg_loss_t, recon_t, fourier_t = [], [], [], []
        for i, (x, _) in enumerate(tqdm(loader)):
            x = x.cuda()

            # get eigen-function, stack copies into batch
            eigen_f = siren_model(dom)

            # ensure that eigen-function is normalized and orthogonal to basis
            eigen_f = orthogonalize(eigen_f, basis)
            eigen_f = eigen_f / func_norm(eigen_f)[..., None, None]
            eigen_f = repeat(eigen_f, 't h w -> b t h w', b=bs)

            # train on the residual
            x_res, coeffs = basis_residual(x, basis)

            # different losses
            inner = func_inner(eigen_f, x_res)
            inner_loss = -inner.abs().mean() / h 

            # projection onto legendre basis
            reg_loss = hps.eigen_reg * leg_basis.recon_error(
                x, eigen_f
            ).square().mean() / h

            loss = inner_loss + reg_loss

            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            # save losses
            inner_loss_t.append(inner_loss.item())
            reg_loss_t.append(reg_loss.item())

            # recon
            recon = x - x_res.detach()
            recon_loss = (recon - x).abs().mean()
            recon_t.append(recon_loss.item())

            # get fourier recon
            recon_fourier = fourier(x, hps.n_basis)
            fourier_loss = (recon_fourier - x).abs().mean()
            fourier_t.append(fourier_loss.item())

        print(f'inner: {np.mean(inner_loss_t)}')
        print(f'reg: {np.mean(reg_loss_t)}')
        print(f'avg recon: {np.mean(recon_t)}')
        print(f'avg fourier: {np.mean(fourier_t)}')

    print('saving model')
    torch.save(siren_model.state_dict(), f'{hps.exp_path}/siren_model.pt')
    basis = save_basis(eigen_f[0], basis, hps.exp_path)

    if basis_num % 1 == 0:
        print('making video')
        res, coeffs = basis_residual(x, basis)

        video_compare(
            coeffs[0],
            recon[0],
            x[0],
            recon_fourier[0],
            ['neural basis', 'fourier'],
            os.path.join(hps.exp_path, f'compare.gif')
        )

        #basis_video(
            #basis,
            #os.path.join(hps.exp_path, f'basis.gif')
        #)

    return basis

def test(loader, basis, hps):
    leg_basis = LegendreBasis()
    recon_t, fourier_t = [], []
    for i, (x, _) in enumerate(tqdm(loader)):
        x = x.cuda()

        # train on the residual
        x_res, _ = basis_residual(x, basis)

        # recon
        recon = x - x_res.detach()
        recon_loss = (recon - x).abs().mean()
        recon_t.append(recon_loss.item())

        # get fourier recon
        recon_fourier = fourier(x, hps.n_basis)
        fourier_loss = (recon_fourier - x).abs().mean()
        fourier_t.append(fourier_loss.item())

    print('final results')
    print(f'avg recon: {np.mean(recon_t):.5f}')
    print(f'avg fourier: {np.mean(fourier_t):.5f}')

'''
get optimal functional pca basis
for a R^{\times d} domain

entry point
'''

def pca_train(loader, hps):
    h, w = 64, 64

    # load basis if exists
    if os.path.exists(f'{hps.exp_path}/basis.pt'):
        print('loading basis')
        basis = torch.load(f'{hps.exp_path}/basis.pt').cuda()
    else:
        # set first basis function to be constant
        constant = torch.ones((50, 1, h, w)).cuda()

        # require that norm of basis function is 1
        norm = func_norm(constant)
        basis = constant / norm[..., None, None]

    for nb in range(basis.shape[1], hps.n_basis):
        print(f'optimizing basis {nb}')

        siren_model = EigenFunction(
            dim_in = 3, # 2 spatial dims with 1 time
            dim_out = hps.channels, # number of channels
            dim_hidden = hps.dim_hidden,
            num_layers = hps.n_layers,
        )

        # load previous model if exists
        n_model = basis.shape[1] 
        if os.path.exists(f'{hps.exp_path}/siren_model_{n_model}.pt'):
            print(f'loading model: {n_model}')
            siren_model.load_state_dict(torch.load(f'{hps.exp_path}/siren_model.pt'))

        optim = torch.optim.Adam(siren_model.parameters(), lr=hps.lr)
        basis = train(siren_model, loader, optim, basis, nb, hps)

def pca_test(loader, hps):
    # load basis
    basis = torch.load(f'{hps.exp_path}/basis.pt').cuda()
    test(loader, basis, hps)


from load import navier_loader
if __name__ == '__main__':
    exp_path = 'results/dev'
    mode = 'test'
    batch_size = 32
    num_workers = 4
    nav_type = 'series'

    basis = torch.load(f'{exp_path}/basis.pt').cuda()

    loader = navier_loader(
        'data/navier.mat', mode, batch_size,
        num_workers, None, nav_type=nav_type 
    )

    recon_stack = None
    for i, (x, _) in enumerate(tqdm(loader)):
        x = x.cuda()

        # train on the residual
        x_res, _ = basis_residual(x, basis)
        recon = x - x_res

        # stack recon
        if recon_stack is None: recon_stack = recon
        else: recon_stack = torch.cat([recon_stack, recon], dim=0)

    torch.save(recon_stack, f'{exp_path}/recon-{mode}.pt')
