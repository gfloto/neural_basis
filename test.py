import torch 
import numpy as np

from fourier_basis import fft_compression

@torch.no_grad()
def test(nb_model, loader, percept_loss, hps, freq=100):
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

