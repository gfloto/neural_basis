import os
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
def basis_residual(x, basis, return_coeffs=False):
    coeffs = torch.einsum('b t h w, t e h w -> b t e', x, basis)
    recon = torch.einsum('b t e, t e h w -> b t h w', coeffs, basis)

    if not return_coeffs: return x - recon
    else: return x - recon, coeffs

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

def train(siren_model, loader, optim, basis, basis_num, hps): 
    siren_model.train()
    siren_model = siren_model.cuda()

    h, w = 64, 64 
    bs = hps.batch_size

    # domain of eigen-function
    dom = make_domain(h, w).cuda()

    for _ in range(5):
        loss_t = []
        for i, (x, _) in enumerate(tqdm(loader)):
            x = x.cuda()

            # get eigen-function, stack copies into batch
            eigen_f = siren_model(dom)

            # ensure that eigen-function is normalized and orthogonal to basis
            eigen_f = orthogonalize(eigen_f, basis)
            eigen_f = eigen_f / func_norm(eigen_f)[..., None, None]
            eigen_f = repeat(eigen_f, 't h w -> b t h w', b=bs)

            # train on the residual
            x_res = basis_residual(x, basis)

            # different losses
            loss = -func_inner(eigen_f, x_res).abs().mean() / (h * w)

            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_t.append(loss.item())

            # print
            if i % 50 == 0:
                print(f'avg loss: {np.mean(loss_t[-100:])}')

    print('saving model')
    torch.save(siren_model.state_dict(), f'{hps.exp_path}/siren_model_{basis_num}.pt')
    basis = save_basis(eigen_f[0], basis, hps.exp_path)

    print('making video')
    res, coeff = basis_residual(x, basis, return_coeffs=True)

    video_compare(
        coeff[0],
        x[0] - res[0],
        x[0],
        os.path.join(hps.exp_path, f'compare.gif')
    )

    basis_video(
        basis,
        os.path.join(hps.exp_path, f'basis.gif')
    )

    return basis

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
            siren_model.load_state_dict(torch.load(f'{hps.exp_path}/siren_model_{n_model}.pt'))

        optim = torch.optim.Adam(siren_model.parameters(), lr=hps.lr)
        basis = train(siren_model, loader, optim, basis, nb, hps)

def pca_test():
    pass