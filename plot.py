import matplotlib.pyplot as plt

def plot_basis(x, line, path):
    '''
    x : [n**2, 32, 32]
    we want to plot an n x n grid of images
    '''

    n = int(x.shape[0] ** 0.5)
    x = x.detach().cpu()

    fig, axs = plt.subplots(n, n, figsize=(10, 10))
    for i in range(n):
        for j in range(n):
            axs[i, j].imshow(x[i*n+j], cmap='cividis')
            axs[i, j].axis('off')

    plt.savefig(path)
    plt.close()


def plot_recon(x, y, z):
    y = y.clamp(0, 1)
    z = z.clamp(0, 1)

    fig = plt.figure(figsize=(9,3))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.imshow(x)
    ax2.imshow(y)
    ax3.imshow(z)
    
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    plt.savefig(f'nb.png')
    plt.close()

def plot_line(x, y, path):
    x = x[0]

    # get idx to sort x
    idx = x.argsort()

    x = x[idx].cpu()
    y = y[:, idx].cpu()

    for i in range(y.shape[0]):
        plt.plot(x, y[i])
    plt.savefig(path)
    plt.close()
