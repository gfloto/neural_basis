import os
import argparse
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import Adam
import lpips

from train import train
from test import test
from neural_basis import NbModel

def get_hps():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_path', type=str, default='dev')
    parser.add_argument('--test', type=bool, default=False)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_ortho', type=int, default=int(1e4))

    parser.add_argument('--n_basis', type=int, default=12)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--dim_hidden', type=int, default=64)

    return parser.parse_args()

def cifar10_loader(batch_size, test):
    train = not test
    dataset = datasets.CIFAR10(
        root='./data', train=train, 
        download=False, transform=transforms.ToTensor()
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    print(f'cifar10 {"test" if test else "train"} loader created')
    print(f'length: {len(loader)}')

    return loader

if __name__ == '__main__':
    hps = get_hps()
    hps.exp_path = f'results/{hps.exp_path}'
    os.makedirs(hps.exp_path, exist_ok=True)

    loader = cifar10_loader(hps.batch_size, hps.test)
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

    if not hps.test:
        train(nb_model, loader, optim, percept_loss, hps)
    else:
        test(nb_model, loader, percept_loss, hps)
