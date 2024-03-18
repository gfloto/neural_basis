import os
import argparse
import torch
from torch.optim import Adam

from load import cifar10_loader, navier_loader
from basis_tt import basis_train, basis_test
from implicit_tt import implicit_train, implicit_test
from ode_fft_tt import ode_fft_train, ode_fft_test
from ode_implicit_tt import ode_imp_train, ode_imp_test
from pca_tt import pca_train, pca_test
from basis_2_tt import basis_2_train, basis_2_test

from neural_basis import NbModel
from meta_neural import Swin, ImplicitSiren
from models.basic import Basic

from models.siren import SirenNet

def get_hps():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_path', type=str, default='dev')
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--dataset', type=str, default='navier')
    parser.add_argument('--task', type=str, default='pca')
    parser.add_argument('--implicit_path', type=str, default='results/dev-nav-2')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--dim_hidden', type=int, default=128)

    # basis task params
    parser.add_argument('--n_ortho', type=int, default=int(1e4))
    parser.add_argument('--n_basis', type=int, default=512)

    # implicit task params
    parser.add_argument('--imp_dim', type=int, default=None)

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')

    hps = parser.parse_args()
    
    assert hps.device in ['cuda', 'cpu']
    assert hps.dataset in ['cifar10', 'navier', 'navier-fft', 'navier-ode']
    assert hps.task in ['basis_2', 'pca', 'implicit', 'basis', 'fft-ode', 'implicit-ode']

    # set number of channels based of dataset
    if hps.dataset == 'cifar10': hps.channels = 3
    elif hps.dataset == 'navier': hps.channels = 1

    return hps

def fetch_tt(task):
    if task == 'pca': return pca_train, pca_test
    elif task == 'basis': return basis_train, basis_test
    elif task == 'basis_2': return basis_2_train, basis_2_test
    elif task == 'implicit': return implicit_train, implicit_test
    elif task == 'fft-ode': return ode_fft_train, ode_fft_test
    elif task == 'implicit-ode': return ode_imp_train, ode_imp_test

if __name__ == '__main__':
    hps = get_hps()
    hps.exp_path = f'results/{hps.exp_path}'
    os.makedirs(hps.exp_path, exist_ok=True)

    if hps.dataset == 'cifar10':
        loader = cifar10_loader(hps.batch_size, hps.test, hps.num_workers)
    elif hps.dataset == 'navier':
        tt = 'test' if hps.test else 'train'
        if hps.task == 'fft-ode': nav_type = 'fft'
        elif hps.task in ['implicit-ode', 'pca', 'basis_2']: nav_type = 'series'
        else: nav_type = None

        loader = navier_loader(
            'data/navier.mat', tt, hps.batch_size,
            hps.num_workers, hps.n_basis, nav_type=nav_type 
        )
    #percept_loss = lpips.LPIPS(net='vgg').to(hps.device)

    # get train and test functions
    train, test = fetch_tt(hps.task)

    # make neural basis model to train
    if hps.task == 'basis':
        siren_model = NbModel(hps.n_basis, hps.dim_hidden, hps.n_layers).to(hps.device)
        print(f'neural basis params: {sum(p.numel() for p in siren_model.parameters())}')

    elif hps.task == 'pca':
        train(loader, hps)

    elif hps.task == 'basis_2':
        basis_2_train(loader, hps)

    elif hps.task == 'implicit':
        swin_model = Swin(hps.imp_dim).to(hps.device) 
        siren_model = ImplicitSiren(hps.imp_dim, hps.dim_hidden, 1, hps.n_layers).to(hps.device) 

    elif hps.task == 'fft-ode':
        op_model = Basic(2*hps.n_basis**2).to(hps.device)
        print(f'fft-ode params: {sum(p.numel() for p in op_model.parameters())}')

    elif hps.task == 'implicit-ode':
        op_model = Basic(hps.imp_dim).to(hps.device)
        print(f'implicit-ode params: {sum(p.numel() for p in op_model.parameters())}')

        swin_model = Swin(hps.imp_dim).to(hps.device)
        swin_model.load_state_dict(torch.load(f'{hps.implicit_path}/swin_model.pt'))
        siren_model = ImplicitSiren(hps.imp_dim, hps.dim_hidden, 1, hps.n_layers).to(hps.device)
        siren_model.load_state_dict(torch.load(f'{hps.implicit_path}/siren_model.pt'))
        print('loaded siren_model and swin_model')

    if hps.task == 'basis':
        params = list(siren_model.parameters())
    elif hps.task == 'pca':
        pass
    elif hps.task == 'implicit':
        params = list(siren_model.parameters()) + list(swin_model.parameters())
    elif hps.task == 'fft-ode':
        params = list(op_model.parameters())
    elif hps.task == 'implicit-ode':
        params = list(op_model.parameters())

    if hps.task != 'pca':
        optim = Adam(
            params=params,
            lr=hps.lr
        )

    # load models
    if hps.task != 'pca':
        if os.path.exists(f'{hps.exp_path}/siren_model.pt'):
            print('loading siren_model and optimizer')
            siren_model.load_state_dict(torch.load(f'{hps.exp_path}/siren_model.pt'))
        
        if os.path.exists(f'{hps.exp_path}/swin_model.pt'):
            swin_model.load_state_dict(torch.load(f'{hps.exp_path}/swin_model.pt'))
        
        if os.path.exists(f'{hps.exp_path}/op_model.pt'):
            op_model.load_state_dict(torch.load(f'{hps.exp_path}/op_model.pt'))

    # run task
    if hps.task == 'basis':
        if not hps.test: train(siren_model, loader, optim, percept_loss, hps)
        else: test(siren_model, loader, percept_loss, hps)
    elif hps.task == 'implicit':
        if not hps.test: train(siren_model, swin_model, loader, optim, hps)
        else: test(siren_model, loader, hps)
    elif hps.task == 'fft-ode':
        if not hps.test: train(op_model, loader, optim, hps)
        else: test(swin_model, loader, hps)
    elif hps.task == 'implicit-ode':
        if not hps.test: train(op_model, swin_model, siren_model, loader, optim, hps)
        else: test(op_model, swin_model, siren_model, loader, hps)
