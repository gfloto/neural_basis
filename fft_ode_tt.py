import torch 
import torch.fft as fft
from einops import rearrange, repeat
from torchdiffeq import odeint as odeint

from plot import video_compare

def ode_train(model, loader, optim, hps):
    while True:
        loss_track = []
        for i, (z, x) in enumerate(loader):
            optim.zero_grad()
            z = z.to(hps.device); x = x.to(hps.device)
            b, t, h, w = x.shape; n_basis = hps.n_basis

            # forward pass
            z0 = z[:, 0, ...]
            t_ = torch.linspace(0, 1, z.shape[1]).to(hps.device)
            y = odeint(model, z0, t_, method='midpoint')
            y = rearrange(y, 't b e -> b t e')

            #loss = (x - x_recon).abs().mean()
            loss = (y - z).abs().mean()
            loss_track.append(loss.item())

            loss.backward()
            optim.step()

            if i % 100 == 0:
                print(f'loss: {sum(loss_track)/len(loss_track)}')

                # compare in image space
                y = rearrange(y, 'b t (c h w) -> b t c h w', c=2, h=n_basis, w=n_basis)
                y = y[:, :, 0, ...] + 1j*y[:, :, 1, ...]
                y *= y.shape[-1] * y.shape[-2]

                y_full = torch.zeros((b, t, h, w), dtype=torch.complex64).to(hps.device)
                y_full[..., :n_basis, :n_basis] = y
                x_recon = fft.ifftn(y_full, dim=(-2, -1)).abs()

                video_compare(x[0].detach().cpu(), x_recon[0].detach().cpu(), 'ode.gif')

        print('saving model')
        torch.save(model.state_dict(), f'{hps.exp_path}/swin_model.pt')

def ode_test():
    pass