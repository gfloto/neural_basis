import torch.fft as fft
import matplotlib.pyplot as plt

def fft_compression(x, n_basis):
    '''
    perform 2d fft on image tensor
    keep subset (n_basis total) of coefficients and return the compressed image
    '''
    assert x.dim() == 4, 'input tensor must be 4d'

    # perform 2d fft
    c = fft.fftn(x, dim=(-2, -1))
    c[..., n_basis:, n_basis:] = 0
    recon = fft.ifftn(c, dim=(-2, -1)).real

    return recon
