import torch
import matplotlib.pyplot as plt

def plot_basis(x, path='basis'):
    '''
    x : [2*n**2, 3, 32, 32]
    we want to plot an n x n grid of images
    '''

    for i in range(x.shape[1]):
        pb(x[:, i], i, f'imgs/{path}_{i}.png')

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
            axs[i, j].imshow(x[i*n+j], cmap=cmap)
            axs[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_recon(x, y, z, path):
    y = y.clamp(0, 1)
    z = z.clamp(0, 1)

    fig = plt.figure(figsize=(9,3))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    x = (x + 1) / 2
    y = (y + 1) / 2
    z = (z + 1) / 2

    y = y.clamp(0, 1)
    z = z.clamp(0, 1)

    ax1.imshow(x)
    ax2.imshow(y)
    ax3.imshow(z)
    
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    plt.savefig(path)
<<<<<<< HEAD
=======
    plt.tight_layout()
    plt.savefig('imgs/nb.png')
>>>>>>> 9c38d717b727e89690058558689c3de77594a45b
    plt.close()

def plot_line(x, y_w, y_h, path):
    # get idx to sort x
    idx = x.argsort()

    x = x[idx].cpu()
    y_w = y_w[:, idx].cpu()
    y_h = y_h[:, idx].cpu()

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    for i in range(y_h.shape[0]):
        ax1.plot(x, y_w[i], alpha=0.5)
        ax2.plot(x, y_h[i], alpha=0.5)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
