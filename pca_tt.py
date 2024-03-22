import os
import math
import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat

from legendre import LegendreBasis
from plot import basis_video, video_compare 
from models.siren import EigenFunction
from models.parallel_siren import ParallelEigenFunction

# plot temporal basis
import matplotlib.pyplot as plt
def plot_temp(x):
    fig = plt.figure(figsize=(6, 4))

    for i in range(x.shape[1]):
        plt.plot(x[:, i], label=f'{i}')
    
    if not os.path.exists('temp'): os.makedirs('temp')
    plt.savefig('temp/temp_basis.png')
    plt.close()

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

def temporal_reg(x, eigen_f, temporal_basis):
    # projection onto temporal basis
    c = torch.einsum('b t h w, b t h w -> b t', x, eigen_f)
    c = (c - c.mean()) / c.std()

    coeffs = torch.einsum('b t, t e -> b e', c, temporal_basis)
    recon = torch.einsum('b e, t e -> b t', coeffs, temporal_basis)

    return recon - c 

def train(spatial_model, temporal_model, loader, optim, basis, basis_num, hps): 
    spatial_model.train(); temporal_model.train()
    spatial_model = spatial_model.cuda(); temporal_model = temporal_model.cuda()

    h, w = 64, 64 
    bs = hps.batch_size

    # domain of eigen-function
    spatial_dom = make_domain(h, w).cuda()
    temporal_dom = torch.linspace(0, 1, 50)[..., None].cuda()

    for _ in range(25):
        inner_loss_t, reg_loss_t, recon_t, fourier_t = [], [], [], []
        for i, (x, _) in enumerate(tqdm(loader)):
            x = x.cuda()

            # get eigen-function, stack copies into batch
            eigen_f = spatial_model(spatial_dom)

            # ensure that eigen-function is normalized and orthogonal to basis
            eigen_f = orthogonalize(eigen_f, basis)
            eigen_f = eigen_f / func_norm(eigen_f)[..., None, None]
            eigen_f = repeat(eigen_f, 't h w -> b t h w', b=bs)

            # train on the residual
            x_res, coeffs = basis_residual(x, basis)

            # different losses
            inner = func_inner(eigen_f, x_res)
            inner_loss = -inner.abs().mean() / (h * w) 

            out = temporal_model(temporal_dom)
            temporal_basis = out / out.square().sum(dim=0, keepdim=True).sqrt()

            reg_loss = temporal_reg(
                x, eigen_f, temporal_basis
            ).square().mean() / h

            # encourage orthogonality of temporal basis
            prod = temporal_basis.T @ temporal_basis
            prod = torch.triu(prod, diagonal=1)
            ortho = prod.square().sum() / (prod.shape[0] * prod.shape[1])

            reg_loss = reg_loss + ortho

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

            # plot basis
            if i % 50 == 0:
                plot_temp(temporal_basis.detach().cpu())

        print(f'inner: {np.mean(inner_loss_t)}')
        print(f'reg: {np.mean(reg_loss_t)}')
        print(f'avg recon: {np.mean(recon_t)}')
        print(f'avg fourier: {np.mean(fourier_t)}')

    print('saving model')
    torch.save(spatial_model.state_dict(), f'{hps.exp_path}/spatial_model.pt')
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

        spatial_model = EigenFunction(
            dim_in = 3, # 2 spatial dims with 1 time
            dim_out = hps.channels, # number of cgghannels
            dim_hidden = hps.dim_hidden,
            num_layers = hps.n_layers,
        )

        n_temp = 12
        w0 = torch.ones(n_temp).cuda()
        w0_initial = torch.linspace(1, n_temp / 2, n_temp).cuda() 

        temporal_model = ParallelEigenFunction(
            ensembles = n_temp,
            dim_in = 1,
            dim_hidden = 64,
            dim_out = 1,
            num_layers = 6,
            w0 = w0,
            w0_initial = w0_initial,
        )

        # load previous model if exists
        n_model = basis.shape[1] 
        if os.path.exists(f'{hps.exp_path}/spatial_model_{n_model}.pt'):
            print(f'loading model: {n_model}')
            spatial_model.load_state_dict(torch.load(f'{hps.exp_path}/spatial_model.pt'))

        params = list(spatial_model.parameters()) + list(temporal_model.parameters())
        optim = torch.optim.Adam(params, lr=hps.lr)

        basis = train(spatial_model, temporal_model, loader, optim, basis, nb, hps)

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
