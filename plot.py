import os
import torch
import matplotlib.pyplot as plt

def plot_coeff_hist(coeffs, path):
    '''
    coeffs is a tensor of [n, k]
    this plots a bar chart of the mean of each coefficient
    it also plots the std of each coefficient as error bars
    '''

    mean = coeffs.abs().mean(dim=0).cpu()
    std = coeffs.abs().std(dim=0).cpu()

    #mean, idx = mean.sort(descending=True)
    #std = std[idx]

    plt.plot(mean, '+')
    plt.xlabel('basis function')
    plt.ylabel('mean abs inner product')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def make_gif(imgs, path):
    '''
    imgs is a tensor of [n, w, h]
    this function make a gif with n frames
    save each frame to path, then use imagemagick to make gif
    '''

    # make directory
    os.makedirs('temp', exist_ok=True)

    # save each frame
    for i in range(imgs.shape[0]):
        plt.imshow(imgs[i], cmap='inferno')
        plt.axis('off')
        num = str(i).zfill(3)
        plt.savefig(f'temp/{num}.png')
        plt.close()

    # make gif
    os.system(f'convert -delay 10 temp/*.png {path}')
    os.system('rm -r temp')

    print('done')
    quit()
    

def plot_basis(x, path):
    '''
    x : [2*n**2, 3, 32, 32]
    we want to plot an n x n grid of images
    '''

    for i in range(x.shape[1]):
        pb(x[:, i], i, os.path.join(path, f'basis_{i}.png'))

def pb(x, i, path):
    # first filter is zeros
    zero = torch.zeros_like(x[0])
    x = torch.cat([zero[None], x], dim=0)
    n = int( ((x.shape[0])/2) ** 0.5 )

    # r g b
    if i == 0: cmap = 'Reds'
    elif i == 1: cmap = 'Greens'
    elif i == 2: cmap = 'Blues'
    else: raise ValueError('invalid channel')

    fig, axs = plt.subplots(n, 2*n, figsize=(10, 5))
    for i in range(n):
        for j in range(2*n):
            axs[i, j].imshow(x[i+n*j], cmap=cmap)
            axs[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_recon(x, y, z, path):
    if x.shape[-1] != 1:
        x /= x.max()
        y /= y.max()
        z /= z.max()

    fig = plt.figure(figsize=(9,3))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    x = (x + 1) / 2
    y = (y + 1) / 2
    z = (z + 1) / 2

    if x.shape[-1] == 1: cmap = 'inferno'
    else: cmap = None

    ax1.imshow(x, cmap=cmap)
    ax2.imshow(y, cmap=cmap)
    ax3.imshow(z, cmap=cmap)
    
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    plt.savefig(path)
    plt.close()
