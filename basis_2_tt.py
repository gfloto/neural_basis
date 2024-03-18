import os
import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat

from plot import basis_video, video_compare 
from models.parallel_siren import EigenFunction

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

def train(siren_model, loader, optim, hps): 
    siren_model.train()
    siren_model = siren_model.cuda()

    h, w = 64, 64 
    bs = hps.batch_size

    # domain of eigen-function
    dom = make_domain(h, w).cuda()

    while True:
        loss_t = []
        for i, (x, _) in enumerate(tqdm(loader)):
            x = x.cuda()

            # get eigen-function, stack copies into batch
            eigen_f = siren_model(dom)

            # ensure that eigen-function is normalized and orthogonal to basis
            eigen_f = eigen_f / func_norm(eigen_f)[..., None, None]

            # recon x using basis
            coeffs = torch.einsum('b t h w, t e h w -> b t e', x, eigen_f)
            recon = torch.einsum('b t e, t e h w -> b t h w', coeffs, eigen_f)

            # different losses
            loss = (x - recon).abs().mean() 

            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_t.append(loss.item())

            # print
            if i % 25 == 0:
                print(f'avg loss: {np.mean(loss_t[-100:])}')

        video_compare(
            coeffs[0].detach(),
            recon[0].detach(),
            x[0],
            os.path.join(hps.exp_path, f'compare.gif')
        )

        print('saving model')
        torch.save(siren_model.state_dict(), f'{hps.exp_path}/siren_model.pt')

    basis = save_basis(eigen_f[0], basis, hps.exp_path)

    print('making video')
    res, coeff = basis_residual(x, basis, return_coeffs=True)

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

def basis_2_train(loader, hps):
    h, w = 64, 64

    w0 = torch.ones(hps.n_basis).cuda()
    w0_initial = 30. * torch.ones(hps.n_basis).cuda()

    siren_model = EigenFunction(
        ensembles=hps.n_basis,
        dim_in = 3, # 2 spatial dims with 1 time
        dim_out = 1, # number of channels
        dim_hidden = hps.dim_hidden,
        num_layers = hps.n_layers,
        w0 = w0,
        w0_initial = w0_initial,
    )

    # load if exists
    if os.path.exists(f'{hps.exp_path}/siren_model.pt'):
        print('loading siren_model')
        siren_model.load_state_dict(torch.load(f'{hps.exp_path}/siren_model.pt'))

    optim = torch.optim.Adam(siren_model.parameters(), lr=hps.lr)
    basis = train(siren_model, loader, optim, hps)

def basis_2_test():
    pass