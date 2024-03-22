import os
import torch
import numpy as np
from tqdm import tqdm
from x_unet import XUnet
from functools import partial
from einops import rearrange
import matplotlib.pyplot as plt

from load import navier_loader

def plot_comp(x, y, recon, path):
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.imshow(y, cmap='inferno')
    ax2.imshow(x, cmap='inferno')
    ax3.imshow(recon, cmap='inferno')

    ax1.set_title('unet')
    ax2.set_title('original')
    ax3.set_title('recon')

    # save
    plt.savefig(path)
    plt.close()

def basis_recon(x, basis):
    coeffs = torch.einsum('b t h w, t e h w -> b t e', x, basis)
    return torch.einsum('b t e, t e h w -> b t h w', coeffs, basis)

def train(unet, recon_func, loader, opt, exp_path):
    loss_t, basis_t = [], []
    for i, (x, _) in enumerate(tqdm(loader)):
        x = x.cuda()

        # train on the residual
        recon = recon_func(x)
        recon_batch = rearrange(recon, 'b t h w -> (b t) 1 h w')

        y = unet(recon_batch)
        y = rearrange(y, '(b t) 1 h w -> b t h w', b=x.shape[0])

        loss = (x - y).abs().mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        # record perf
        loss_t.append(loss.item())
        basis_loss = (x - recon).abs().mean()
        basis_t.append(basis_loss.item())

        if i % 10 == 0 :
            print(f'train unet: {np.mean(loss_t[-50:]) :.5f}')
            print(f'train basis: {np.mean(basis_t[-50:]) :.5f}')
            print()

            t = torch.randint(0, x.shape[1], (1,)).item()
            path = f'results/dev/unet-train.png'

            plot_comp(
                x[0,t].detach().cpu(),
                y[0,t].detach().cpu(),
                recon[0,t].detach().cpu(),
                path,
            )

    # save model
    torch.save(unet.state_dict(), f'{exp_path}/unet.pt')

@torch.no_grad()
def test(unet, recon_func, loader, exp_path):
    loss_t, basis_t = [], []
    for i, (x, _) in enumerate(tqdm(loader)):
        x = x.cuda()

        # train on the residual
        recon = recon_func(x)
        recon_batch = rearrange(recon, 'b t h w -> (b t) 1 h w')

        y = unet(recon_batch)
        y = rearrange(y, '(b t) 1 h w -> b t h w', b=x.shape[0])

        loss = (x - y).abs().mean()

        # record perf
        loss_t.append(loss.item())
        basis_loss = (x - recon).abs().mean()
        basis_t.append(basis_loss.item())

        if i % 25 == 0 :
            print(f'test unet: {np.mean(loss_t) :.5f}')
            print(f'test basis: {np.mean(basis_t) :.5f}')
            print()

            t = torch.randint(0, x.shape[1], (1,)).item()
            path = f'results/dev/unet-test.png'

            plot_comp(
                x[0,t].detach().cpu(),
                y[0,t].detach().cpu(),
                recon[0,t].detach().cpu(),
                path,
            )

if __name__ == '__main__':
    exp_path = 'results/dev'
    batch_size = 4
    num_workers = 4
    nav_type = 'series'
    lr = 5e-5

    basis = torch.load(f'{exp_path}/basis.pt').cuda()
    recon_func = partial(basis_recon, basis=basis)

    train_loader = navier_loader(
        'data/navier.mat', 'train', batch_size,
        num_workers, None, nav_type=nav_type 
    )
    test_loader = navier_loader(
        'data/navier.mat', 'test', batch_size,
        num_workers, None, nav_type=nav_type 
    )

    unet = XUnet(
        dim = 64,
        channels = 1,
        dim_mults = (1, 2, 4, 4),
        nested_unet_depths = (4, 4, 2, 1),
        consolidate_upsample_fmaps = True,
    ).cuda()
    print('total params:', sum(p.numel() for p in unet.parameters()))

    if os.path.exists(f'{exp_path}/unet.pt'):
        print('loading model')
        unet.load_state_dict(torch.load(f'{exp_path}/unet.pt'))

    opt = torch.optim.Adam(unet.parameters(), lr=lr)

    while True:
        train(unet, recon_func, train_loader, opt, exp_path)
        test(unet, recon_func, test_loader, exp_path)
        
