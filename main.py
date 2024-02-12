import os
import argparse
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import Adam
import lpips

from plot import plot_recon
from neural_basis import NbModel
from fourier_basis import fft_compression

def get_hps():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_path', type=str, default='dev')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_ortho', type=int, default=int(1e4))
    parser.add_argument('--ortho', type=bool, default=True)

    parser.add_argument('--n_basis', type=int, default=12)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--dim_hidden', type=int, default=64)

    return parser.parse_args()

def cifar10_loader(batch_size):
    dataset = datasets.CIFAR10(
        root='./data', train=True, 
        download=True, transform=transforms.ToTensor()
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    return loader

if __name__ == '__main__':
    hps = get_hps()
    hps.exp_path = f'results/{hps.exp_path}'
    os.makedirs(hps.exp_path, exist_ok=True)

    loader = cifar10_loader(hps.batch_size)
    percept_loss = lpips.LPIPS(net='vgg').to(hps.device)

    # make neural basis model to train
    nb_model = NbModel(hps.n_basis, hps.dim_hidden, hps.n_layers).to(hps.device)
    print(f'neural basis params: {sum(p.numel() for p in nb_model.parameters())}')

    optim = Adam(
        params=list(nb_model.parameters()),
        lr=hps.lr
    )

    # if model and optimizer exist, load them
    if os.path.exists(f'{hps.exp_path}/nb_model.pt'):
        print('loading model and optimizer')
        nb_model.load_state_dict(torch.load(f'{hps.exp_path}/nb_model.pt'))

    freq = 100
    while True:
        nb_nll_t, nb_lpips_t, fft_nll_t, fft_lpips_t, ortho_t = [], [], [], [], []
        print('fresh epoch')
        for i, (x, _) in enumerate(loader):
            plot = i % freq == 0

            # normalize data
            x = x.to(hps.device)
            mean = x.mean(dim=(1,2,3), keepdim=True)
            x -= mean

            # forward pass and optim
            nb_y = nb_model(x.clone(), plot=plot)
            
            nb_nll = (x - nb_y).abs().mean() 
            nb_lpips = percept_loss(x, nb_y).mean()

            # encourage orthonormality
            if hps.ortho:
                ortho = nb_model.orthon_sample(hps.n_ortho, device=hps.device)
            else:
                ortho = torch.tensor(0).to(hps.device)

            loss = nb_nll + nb_lpips / 5 + ortho

            optim.zero_grad()
            loss.backward()
            optim.step()

            # get fft
            fft_y = fft_compression(x, hps.n_basis)
            ft_nll = (x - fft_y).abs().mean()
            ft_lpips = percept_loss(x, fft_y).mean()

            # store loss for logging
            nb_nll_t.append(nb_nll.item())
            nb_lpips_t.append(nb_lpips.item())
            fft_nll_t.append(ft_nll.item())
            fft_lpips_t.append(ft_lpips.item())
            ortho_t.append(ortho.item())

            if i % freq == 0:
                m = lambda x: sum(x) / len(x)
                print(f'nb_nll: {m(nb_nll_t):.6f}, nb_lpips: {m(nb_lpips_t):.6f}, ortho: {m(ortho_t):.6f}')
                print(f'fft_nll: {m(fft_nll_t):.6f}, fft_lpips: {m(fft_lpips_t):.6f}')
                print(f'total nb: {m(nb_nll_t) + m(nb_lpips_t):.6f}, total fft: {m(fft_nll_t) + m(fft_lpips_t):.6f}\n')

                fix = lambda x: (x[0] + mean[0]).clamp(0, 1).permute(1, 2, 0).detach().cpu()
                x, nb_y, fft_y = fix(x), fix(nb_y), fix(fft_y) 

                plot_recon(x, nb_y, fft_y)

        # save model and optimizer
        torch.save(nb_model.state_dict(), f'{hps.exp_path}/nb_model.pt')