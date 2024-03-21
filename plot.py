import os
import math
import torch
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt

plt.style.use('dark_background')

def basis_video(x, path):
    t, nb, h, w = x.shape

    # map to [0, 1]
    x = (x - x.min()) / (x.max() - x.min())

    # pad images with zeros
    p = 1
    zeros = torch.zeros(t, nb, h+2*p, w+2*p).cuda()
    zeros[..., p:-p, p:-p] = x
    x = zeros

    # reshape to square with tiles
    # first add empty tiles
    k = math.ceil(nb ** 0.5)
    add_tiles = torch.zeros(t, k**2 - nb, h+2*p, w+2*p).cuda()
    x = torch.cat([x, add_tiles], dim=1)

    x = rearrange(x, 't (k1 k2) h w -> t (k1 h) (k2 w)', k1=k, k2=k).cpu()

    # make directory
    os.makedirs('temp', exist_ok=True)

    # save each frame
    for i in range(x.shape[0]):
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(x[i], cmap='inferno')
        plt.axis('off')

        num = str(i).zfill(3)
        plt.savefig(f'temp/{num}.png')
        plt.close()

    # make gif
    os.system(f'convert -delay 10 temp/*.png {path}')
    os.system('rm -r temp')

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

# just like make_gif, but 2 frames
def video_compare(coeff, x, z, y, labels, path):
    '''
    x, y are tensors of [n, w, h]
    this function makes a gif with n frames
    each frame is a side by side comparison of x and y
    '''

    x = x.cpu(); y = y.cpu(); z = z.cpu(); coeff = coeff.cpu()

    # make directory
    os.makedirs('temp', exist_ok=True)
    
    # get n uniform samples from 
    colors = plt.cm.inferno(np.linspace(0.2, 1, coeff.shape[1]))

    # save each frame
    for i in range(x.shape[0]-1, x.shape[0]):
        fig = plt.figure(figsize=(10, 10)) 

        # coeff line plot
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 3, 7)
        ax4 = fig.add_subplot(3, 3, 8)
        ax5 = fig.add_subplot(3, 3, 9)

        # plot loss for fourier and neural basis
        loss_1 = (z - x).abs().mean(dim=(1, 2))
        loss_2 = (z - y).abs().mean(dim=(1, 2))
        ax1.plot(loss_1[:i], label=labels[0])
        ax1.plot(loss_2[:i], label=labels[1])
        ax1.set_title('reconstruction loss')
        ax1.legend()

        # plot neural eigenvalues
        for j in range(coeff.shape[1]):
            color = colors[j]
            ax2.plot(coeff[:i, j], color=color)
        ax2.set_title('neural eigenvalues')

        ax3.imshow(x[i], cmap='inferno')
        ax4.imshow(z[i], cmap='inferno')
        ax5.imshow(y[i], cmap='inferno')

        # titles
        ax3.set_title(labels[0])
        ax4.set_title('original')
        ax5.set_title(labels[1])

        ax3.axis('off')
        ax4.axis('off')
        ax5.axis('off')

        num = str(i).zfill(3)
        plt.savefig(f'temp/{num}.png')
        plt.close()

    # make gif
    #os.system(f'convert -delay 10 temp/*.png {path}')
    #os.system('rm -r temp')

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


if __name__ == '__main__':
    basis_path = 'results/dev/basis.pt'
    basis = torch.load(basis_path, map_location='cpu')

    basis = basis[:, 1]

    basis = basis.detach()
    make_gif(basis, 'basis.gif')
