import os
import torch

from plot import plot_recon, plot_coeff_hist
from fourier_basis import fft_compression

def basis_train(nb_model, loader, optim, percept_loss, hps, freq=100):
    print('training')
    nb_model.train()

    while True:
        coeffs_t = None
        nb_nll_t, nb_lpips_t, fft_nll_t, fft_lpips_t, ortho_t = [], [], [], [], []
        print('new epoch')
        for i, (x, _) in enumerate(loader):
            plot = i % freq == 0
            plot_path = hps.exp_path if plot else None

            # normalize data
            x = x.to(hps.device)
            mean = x.mean(dim=(1,2,3), keepdim=True)
            x -= mean

            # forward pass and optim
            nb_y, coeffs = nb_model(x.clone(), plot_path=plot_path)
            
            nb_nll = (x - nb_y).abs().mean() 
            nb_lpips = percept_loss(x, nb_y).mean()

            # encourage orthonormality
            ortho = nb_model.orthon_sample(hps.n_ortho, device=hps.device)

            lmbda = 0.2 if hps.dataset == 'cifar10' else 0
            loss = nb_nll + lmbda * nb_lpips + ortho

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

            # coeff distribution
            coeffs = coeffs.mean(dim=-1).detach()
            if coeffs_t is None: coeffs_t = coeffs
            else: coeffs_t = torch.cat([coeffs_t, coeffs], dim=0)

            if i % freq == 0:
                m = lambda x: sum(x) / len(x)
                print(f'nb_nll: {m(nb_nll_t):.6f}, nb_lpips: {m(nb_lpips_t):.6f}, ortho: {m(ortho_t):.6f}')
                print(f'fft_nll: {m(fft_nll_t):.6f}, fft_lpips: {m(fft_lpips_t):.6f}')
                print(f'total nb: {m(nb_nll_t) + m(nb_lpips_t):.6f}, total fft: {m(fft_nll_t) + m(fft_lpips_t):.6f}\n')

                fix = lambda x: (x[0] + mean[0]).clamp(0, 1).permute(1, 2, 0).detach().cpu()
                x, nb_y, fft_y = fix(x), fix(nb_y), fix(fft_y) 

                plot_recon(x, nb_y, fft_y, os.path.join(hps.exp_path, f'recon.png'))
                plot_coeff_hist(coeffs_t, os.path.join(hps.exp_path, f'coeffs.png'))

        # save model and optimizer
        torch.save(nb_model.state_dict(), f'{hps.exp_path}/nb_model.pt')

@torch.no_grad()
def basis_test(nb_model, loader, percept_loss, hps, freq=100):
    print('testing')
    nb_model.eval()

    nb_score, fft_score = None, None
    for i, (x, _) in enumerate(loader):
        # normalize data
        x = x.to(hps.device)
        mean = x.mean(dim=(1,2,3), keepdim=True)
        x -= mean

        # forward pass and optim
        nb_y = nb_model(x.clone())
        
        nb_nll = (x - nb_y).abs().mean(dim=(1,2,3))
        nb_lpips = percept_loss(x, nb_y).mean(dim=(1,2,3))

        # get fft
        fft_y = fft_compression(x, hps.n_basis)
        fft_nll = (x - fft_y).abs().mean(dim=(1,2,3))
        fft_lpips = percept_loss(x, fft_y).mean(dim=(1,2,3))

        # store loss for logging
        if nb_score is None:
            nb_score = (nb_nll + nb_lpips)
            fft_score = (fft_nll + fft_lpips)
        else:
            nb_score = torch.cat([nb_score, (nb_nll + nb_lpips)], dim=0)
            fft_score = torch.cat([fft_score, (fft_nll + fft_lpips)], dim=0)

        # print metrics
        if i % freq == 0 or i == len(loader) - 1:
            # print means
            print(f'nb_score: {nb_score.mean():.6f}, fft_score: {fft_score.mean():.6f}')
            
            # percent nb better than fft
            nb_better = (nb_score < fft_score).float().mean()
            print(f'nb better than fft: {nb_better:.4f}')