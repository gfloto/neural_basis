import os
import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from einops import rearrange
import matplotlib.pyplot as plt
from mamba_ssm import Mamba

from load import navier_loader
from plot import video_compare

def plot_comp(x, y, recon, path):
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.imshow(y, cmap='inferno')
    ax2.imshow(x, cmap='inferno')
    ax3.imshow(recon, cmap='inferno')

    ax1.set_title('mamba')
    ax2.set_title('original')
    ax3.set_title('neural basis')

    # save
    plt.savefig(path)
    plt.close()

def basis_recon_seed(x, basis):
    coeffs = torch.einsum('b t h w, t e h w -> b t e', x, basis)
    recon = torch.einsum('b t e, t e h w -> b t h w', coeffs, basis)
    return recon, coeffs

def coeff_recon_seed(coeffs, basis):
    return torch.einsum('b t e, t e h w -> b t h w', coeffs, basis)

def train(model, basis_func, coeff_func, loader, opt, exp_path):
    for _ in range(10):
        loss_t, basis_t, mamba_t = [], [], []
        for i, (x, _) in enumerate(tqdm(loader)):
            x = x.cuda()

            # train on the residual
            basis_recon, coeffs = basis_func(x)

            z = model(coeffs)

            loss = (z[:, :-1] - coeffs[:, 1:]).abs().mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            # record perf
            loss_t.append(loss.item())

            basis_loss = (x - basis_recon).abs().mean()
            basis_t.append(basis_loss.item())

            mamba_recon = coeff_func(z)
            mamba_loss = (x - mamba_recon).abs().mean()
            mamba_t.append(mamba_loss.item())

        print(f'train loss: {np.mean(loss_t) :.7f}')
        print(f'mamba: {np.mean(mamba_t) :.5f}')
        print(f'basis: {np.mean(basis_t) :.5f}')

    path = os.path.join(exp_path, 'mamba-train.gif')

    video_compare(
        z[0, 1:].detach().cpu(),
        mamba_recon[0, 1:].detach().cpu(),
        x[0, 1:].detach().cpu(),
        basis_recon[0, 1:].detach().cpu(),
        ['mamba', 'neural basis'],
        path,
    )

    # save model
    torch.save(model.state_dict(), f'{exp_path}/mamba.pt')

@torch.no_grad()
def test(model, basis_func, coeff_func, loader, exp_path):
    loss_t, basis_t, mamba_t = [], [], []
    for i, (x, _) in enumerate(tqdm(loader)):
        x = x.cuda()

        # train on the residual
        basis_recon, coeffs = basis_func(x)
        out = coeffs[:,0][:, None]

        conv_state = torch.zeros(x.shape[0], 2*model.d_model, model.d_conv).cuda()
        hidden_state = torch.zeros(x.shape[0], 2*model.d_model, model.d_state).cuda()

        z = None
        for t in range(x.shape[1]):
            out, conv_state, hidden_state = model.step(out, conv_state, hidden_state)

            if z is None: z = out
            else: z = torch.cat((z, out), dim=1)

        loss = (coeffs - z).abs().mean()

        # record perf
        loss_t.append(loss.item())

        basis_loss = (x - basis_recon).abs().mean()
        basis_t.append(basis_loss.item())

        mamba_recon = coeff_func(z)
        mamba_loss = (x - mamba_recon).abs().mean()
        mamba_t.append(mamba_loss.item())

        if i % 25 == 0 :
            print(f'test loss: {np.mean(loss_t[-50:]) :.5f}')
            print(f'mamba: {np.mean(mamba_t[-50:]) :.5f}')
            print(f'basis: {np.mean(basis_t[-50:]) :.5f}')

    path = os.path.join(exp_path, 'mamba-test.gif')

    video_compare(
        z[0, 1:].detach().cpu(),
        mamba_recon[0, 1:].detach().cpu(),
        x[0, 1:].detach().cpu(),
        basis_recon[0, 1:].detach().cpu(),
        ['mamba', 'neural basis'],
        path,
    )

if __name__ == '__main__':
    exp_path = 'results/dev'
    batch_size = 32
    num_workers = 4
    nav_type = 'series'
    lr = 1e-4

    basis = torch.load(f'{exp_path}/basis.pt').cuda()
    basis_func = partial(basis_recon_seed, basis=basis)
    coeff_func = partial(coeff_recon_seed, basis=basis)

    train_loader = navier_loader(
        'data/navier.mat', 'train', batch_size,
        num_workers, None, nav_type=nav_type 
    )
    test_loader = navier_loader(
        'data/navier.mat', 'test', batch_size,
        num_workers, None, nav_type=nav_type 
    )

    model = Mamba(
        d_model=basis.shape[1],
        d_state=16,
        d_conv=4,
        expand=2,
    ).cuda()

    print('total params:', sum(p.numel() for p in model.parameters()))

    if os.path.exists(f'{exp_path}/mamba.pt'):
        print('loading model')
        model.load_state_dict(torch.load(f'{exp_path}/mamba.pt'))

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    while True:
        train(model, basis_func, coeff_func, train_loader, opt, exp_path)
        test(model, basis_func, coeff_func, test_loader, exp_path)
        
