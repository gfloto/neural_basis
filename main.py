import os
import argparse
from einops import repeat
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import Adam
import matplotlib.pyplot as plt

from fourier_basis import fft_compression
from neural_basis import NbModel
from plot import plot_recon

def get_hps():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_path', type=str, default='dev')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--n_basis', type=int, default=4)
    parser.add_argument('--dim_hidden', type=int, default=64)
    parser.add_argument('--n_ortho', type=int, default=int(1e4))
    parser.add_argument('--device', type=str, default='cpu')

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
    model = NbModel(hps.n_basis, hps.dim_hidden).to(hps.device)
    optim = Adam(model.parameters(), lr=hps.lr)

    # if model and optimizer exist, load them
    if os.path.exists(f'{hps.exp_path}/model.pth'):
        print('loading model and optimizer')
        #model.load_state_dict(torch.load(f'{hps.exp_path}/model.pth'))
        #optim.load_state_dict(torch.load(f'{hps.exp_path}/optim.pth'))

    freq = 20
    while True:
        rl, ol = [], []
        print('fresh epoch')
        for i, (x, _) in enumerate(loader):
            plot = i % freq == 0
            x = x.to(hps.device)

            # get reconstruction
            y = model(x, plot=plot) 
            recon = (x - y).pow(2).mean() 

            # encourage orthonormality
            ortho = model.orthon_sample(hps.n_ortho, device=hps.device, plot=plot)
            loss = recon + ortho

            optim.zero_grad()
            loss.backward()
            optim.step()

            rl.append(recon.item())
            ol.append(ortho.item())

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
        torch.save(model.state_dict(), f'{hps.exp_path}/model.pth')
        torch.save(optim.state_dict(), f'{hps.exp_path}/optim.pth')