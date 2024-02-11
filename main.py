import os
import argparse
from einops import repeat, rearrange
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import Adam
import matplotlib.pyplot as plt

from plot import plot_recon
from models import CoeffModel
from neural_basis import NbModel
from fourier_basis import fft_compression

def get_hps():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_path', type=str, default='dev')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_ortho', type=int, default=int(1e3))

    parser.add_argument('--n_basis', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--dim_hidden', type=int, default=64)

    return parser.parse_args()

def cifar10_loader(batch_size):
    dataset = datasets.CIFAR10(
        root='./data', train=True, 
        download=False, transform=transforms.ToTensor()
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    return loader

if __name__ == '__main__':
    hps = get_hps()
    os.makedirs(hps.exp_path, exist_ok=True)

    loader = cifar10_loader(hps.batch_size)
    #percept_loss = lpips.LPIPS(net='alex')

    # make neural basis model to train
    nb_model = NbModel(hps.n_basis, hps.dim_hidden, hps.n_layers).to(hps.device)
    print(f'neural basis params: {sum(p.numel() for p in nb_model.parameters())}')

    coeff_model = CoeffModel(hps.n_basis).to(hps.device)
    print(f'coeff model params: {sum(p.numel() for p in coeff_model.parameters())}')

    optim = Adam(
        params=list(nb_model.parameters()) + list(coeff_model.parameters()),
        lr=hps.lr
    )

    # if model and optimizer exist, load them
    if os.path.exists(f'{hps.exp_path}/model.pth'):
        print('loading model and optimizer')
        #model.load_state_dict(torch.load(f'{hps.exp_path}/model.pth'))
        #optim.load_state_dict(torch.load(f'{hps.exp_path}/optim.pth'))

    freq = 100
    while True:
        rl, ol = [], []
        print('fresh epoch')
        for i, (x, _) in enumerate(loader):
            plot = i % freq == 0
            x = x.to(hps.device)
            x = 2*x - 1

            # get reconstruction
            y = model(x, plot=plot) 
            recon = (x - y).pow(2).mean() 

            # encourage orthonormality
            ortho = nb_model.orthon_sample(hps.n_ortho, device=hps.device, plot=plot)
            loss = recon #+ 1e-2*ortho

            optim.zero_grad()
            loss.backward()
            optim.step()

            rl.append(recon.item())
            ol.append(ortho.item())
            ol.append(0)

            if i % freq == 0:
                print(f'recon: {sum(rl)/len(rl):.4f}, ortho: {sum(ol)/len(ol):.4f}')

                # get fft
                fft = fft_compression(x, hps.n_basis)

                plot_recon(
                    x[0].permute(1, 2, 0).detach().cpu(),
                    y[0].permute(1, 2, 0).detach().cpu(),
                    fft[0].permute(1, 2, 0).detach().cpu(),
                )

        # save model and optimizer
        torch.save(nb_model.state_dict(), f'{hps.exp_path}/nb_model.pth')
        torch.save(coeff_model.state_dict(), f'{hps.exp_path}/coeff_model.pth')
        torch.save(optim.state_dict(), f'{hps.exp_path}/optim.pth')