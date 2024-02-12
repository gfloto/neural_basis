import math
import torch 
from einops import repeat, rearrange
import torch.fft as fft

from main import cifar10_loader
from plot import plot_basis, plot_recon
from fourier_basis import fft_compression

def fourier_basis(n_lines, plot=False):
    x_ = torch.linspace(0, 2*math.pi, 32)
    y_ = torch.linspace(0, 2*math.pi, 32)
    x, y = torch.meshgrid(x_, y_, indexing='ij')


    sin_basis, cos_basis = [], []
    for i in range(n_lines):
        for j in range(n_lines):
            sin_basis.append(torch.sin(i*x + j*y))
            cos_basis.append(torch.cos(i*x + j*y))

    sin_basis = torch.stack(sin_basis); cos_basis = torch.stack(cos_basis)

    plot = True
    if plot:
        plot_basis(sin_basis, 'imgs/sin_cbos.png')
        plot_basis(cos_basis, 'imgs/cos_cbos.png')

    ch_reshape = lambda x: repeat(x, 'b h w -> b c h w', c=3)
    sin_basis = ch_reshape(sin_basis); cos_basis = ch_reshape(cos_basis)
    return sin_basis, cos_basis

if __name__ == '__main__':
    '''
    initial attempt at custom:
    complete biorthogonal lines representation for images
    '''

    n_lines = 12
    plot = False
    synth = False

    # make basis
    sin_basis, cos_basis = fourier_basis(n_lines)

    if synth:
        # make up some coeffs
        a = torch.randn(n_lines**2, 3)
        b = torch.randn(n_lines**2, 3)
        sb = torch.einsum('n c h w, n c -> c h w', sin_basis, a)[None, ...]
        cb = torch.einsum('n c h w, n c -> c h w', cos_basis, b)[None, ...]
        img = sb + cb

        # squash to [0, 1]
        img = img - img.min()
        img = img / img.max()

    else:
        # get image
        loader = cifar10_loader(1)
        img = next(iter(loader))[0]
        mean, std = img.mean(), img.std()
        img = (img - mean) / std

    # get coefficients
    batch_img = repeat(img, '1 c h w -> b c h w', b=n_lines**2)
    a = torch.einsum('n c h w, n c h w -> n c', sin_basis, batch_img) / (32**2)
    b = torch.einsum('n c h w, n c h w -> n c', cos_basis, batch_img) / (32**2)

    # reconstruct image
    img_sin = torch.einsum('n c h w, n c -> c h w', sin_basis, a)
    img_cos = torch.einsum('n c h w, n c -> c h w', cos_basis, b)
    img_recon = img_sin + img_cos

    # get fft exact
    fft = fft_compression(img, n_lines)

    fix = lambda x: x * std + mean
    img = fix(img); img_recon = fix(img_recon); fft = fix(fft)

    # plot
    plot_recon(
        img[0].permute(1, 2, 0),
        img_recon.permute(1, 2, 0),
        fft[0].permute(1, 2, 0),
        'imgs/recon.png'
    )

    # print loss
    print('recon loss:', (img - img_recon).pow(2).mean().item())
    print('fft loss:', (img - fft).pow(2).mean().item())
    