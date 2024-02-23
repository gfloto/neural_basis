import torch
from torch.nn.functional import sigmoid
from einops import rearrange, repeat
import matplotlib.pyplot as plt

def plot_recon(y, tgt):
    fix = lambda x: x.squeeze().detach().cpu()
    y, tgt = fix(y), fix(tgt)

    fig = plt.figure(figsize=(6,3))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(y, cmap='inferno')
    ax2.imshow(tgt, cmap='inferno')

    ax1.axis('off')
    ax2.axis('off')

    plt.savefig('recon.png')
    plt.close()


def implicit_train(siren_model, swin_model, loader, optim, hps):
    siren_model.train(); swin_model.train()

    h, w = 64, 64 
    xh = torch.linspace(0, 1, h).cuda()
    xw = torch.linspace(0, 1, w).cuda()
    xh, xw = torch.meshgrid(xh, xw)
    x = torch.stack([xh, xw], dim=-1) 
    dom = rearrange(x, 'h w c -> (h w) c')
    dom = repeat(dom, 'd i -> b d i', b=hps.batch_size)

    while True:
        loss_t = []
        for i, (x, _) in enumerate(loader):
            x = sigmoid(x.cuda())
            x_ = repeat(x, 'b 1 h w -> b c h w', c=3)
            z = swin_model(x_)
            z = repeat(z, 'b i -> b d i', d=h*w)

            y = siren_model(dom, z)
            y = sigmoid(y)
            y = rearrange(y, 'b (h w) c -> b c h w', h=h, w=w)
            loss = (y - x).abs().mean()

            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_t.append(loss.item())

            # print
            if i % 100 == 0:
                print(f'loss: {sum(loss_t) / len(loss_t)}')
                plot_recon(y[0], x[0])

        print('saving model')
        torch.save(siren_model.state_dict(), f'{hps.exp_path}/siren_model.pt')
        torch.save(swin_model.state_dict(), f'{hps.exp_path}/swin_model.pt')

def implicit_test():
    pass