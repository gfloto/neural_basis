import os
import argparse
import torch
from torch.optim import Adam
import lpips

from load import cifar10_loader, navier_loader
from basis_tt import basis_train, basis_test
from implicit_tt import implicit_train, implicit_test
from ode_tt import ode_train, ode_test

from neural_basis import NbModel
from meta_neural import Swin, ImplicitSiren
from models.basic import Basic

def get_hps():
    parser = argparse.ArgumentParser()

<<<<<<< HEAD
    parser.add_argument('--exp_path', type=str, default='dev-nav-2')
=======
    parser.add_argument('--exp_path', type=str, default='ode')
>>>>>>> 949f09f3f2ccb0a160ac897af0ef2045f9ae90e2
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--dataset', type=str, default='navier')
    parser.add_argument('--task', type=str, default='fft-ode')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5)

    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--dim_hidden', type=int, default=512)

    # basis task params
    parser.add_argument('--n_ortho', type=int, default=int(1e4))
    parser.add_argument('--n_basis', type=int, default=12)

    # implicit task params
    parser.add_argument('--imp_dim', type=int, default=256)

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')

    hps = parser.parse_args()
    
    assert hps.device in ['cuda', 'cpu']
    assert hps.dataset in ['cifar10', 'navier', 'navier-fft']
    assert hps.task in ['implicit', 'basis', 'fft-ode']

    return hps

def fetch_tt(task):
    if task == 'basis': return basis_train, basis_test
    elif task == 'implicit': return implicit_train, implicit_test
    elif task == 'fft-ode': return ode_train, ode_test

if __name__ == '__main__':
    hps = get_hps()
    hps.exp_path = f'results/{hps.exp_path}'
    os.makedirs(hps.exp_path, exist_ok=True)

    if hps.dataset == 'cifar10':
        loader = cifar10_loader(hps.batch_size, hps.test, hps.num_workers)
    elif hps.dataset == 'navier':
        tt = 'test' if hps.test else 'train'
        basis = True if hps.task == 'basis' else False
        n_basis = hps.n_basis if hps.task == 'fft-ode' else None

        loader = navier_loader(
            'data/navier.mat', tt, hps.batch_size,
            hps.num_workers, basis=basis, n_basis=n_basis
        )
    percept_loss = lpips.LPIPS(net='vgg').to(hps.device)

    # make neural basis model to train
    if hps.task == 'basis':
        siren_model = NbModel(hps.n_basis, hps.dim_hidden, hps.n_layers).to(hps.device)
        print(f'neural basis params: {sum(p.numel() for p in siren_model.parameters())}')
    elif hps.task == 'implicit':
        swin_model = Swin(hps.imp_dim).to(hps.device) 
        siren_model = ImplicitSiren(hps.imp_dim, hps.dim_hidden, 1, hps.n_layers).to(hps.device) 
    elif hps.task == 'fft-ode':
        swin_model = Basic(2*hps.n_basis**2).to(hps.device)
        print(f'fft-ode params: {sum(p.numel() for p in swin_model.parameters())}')

    else: raise ValueError('task must be "basis" or "implicit"')

    if hps.task == 'basis':
        params = list(siren_model.parameters())
    elif hps.task == 'implicit':
        params = list(siren_model.parameters()) + list(swin_model.parameters())
    elif hps.task == 'fft-ode':
        params = list(swin_model.parameters())

    optim = Adam(
        params=params,
        lr=hps.lr
    )

    # load models
    if os.path.exists(f'{hps.exp_path}/siren_model.pt'):
        print('loading siren_model and optimizer')
        siren_model.load_state_dict(torch.load(f'{hps.exp_path}/siren_model.pt'))
    
    if os.path.exists(f'{hps.exp_path}/swin_model.pt'):
        swin_model.load_state_dict(torch.load(f'{hps.exp_path}/swin_model.pt'))

    # get train and test functions
    train, test = fetch_tt(hps.task)

    # run task
    if hps.task == 'basis':
        if not hps.test: train(siren_model, loader, optim, percept_loss, hps)
        else: test(siren_model, loader, percept_loss, hps)
    elif hps.task == 'implicit':
        if not hps.test: train(siren_model, swin_model, loader, optim, hps)
        else: test(siren_model, loader, hps)
    elif hps.task == 'fft-ode':
        if not hps.test: train(swin_model, loader, optim, hps)
        else: test(swin_model, loader, hps)
