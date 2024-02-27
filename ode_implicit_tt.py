import torch 
from einops import rearrange, repeat
from torchdiffeq import odeint as odeint

from plot import video_compare

@ torch.no_grad()
def compress(x, swin_model, bs=256):
    b, t, h, w = x.shape
    x = rearrange(x, 'b t h w -> (b t) h w')
    x = repeat(x, 'b h w -> b c h w', c=3)
    
    out = []
    for i in range(0, x.shape[0], bs):
        x_ = x[i : i+bs]
        z = swin_model(x_)
        out.append(z)
    
    out = torch.cat(out, dim=0)
    out = rearrange(out, '(b t) d -> b t d', b=b, t=t)

    return out

def decompress(z, dom, siren_model, bs=8):
    b, t, d = z.shape
    z = rearrange(z, 'b t d -> (b t) d')
    z = repeat(z, 'b i -> b d i', d=64**2)

    out = []
    for i in range(0, z.shape[0], bs):
        z_ = z[i : i+bs]
        y = siren_model(dom, z_)
        out.append(y)
    
    out = torch.cat(out, dim=0)
    out = rearrange(out, '(b t) (h w) 1 -> b t h w', b=b, t=t, h=64, w=64)
    return out

def ode_imp_train(op_model, swin_model, siren_model, loader, optim, hps):
    op_model.train(); swin_model.eval(); siren_model.eval()
    op_model.to(hps.device); swin_model.to(hps.device); siren_model.to(hps.device)

    # domain for decompression
    h, w = 64, 64 
    xh = torch.linspace(0, 1, h).cuda()
    xw = torch.linspace(0, 1, w).cuda()
    xh, xw = torch.meshgrid(xh, xw)
    x = torch.stack([xh, xw], dim=-1) 
    dom = rearrange(x, 'h w c -> (h w) c')
    dom = repeat(dom, 'd i -> b d i', b=8)

    t_sample = 12 # sample 6 frames
    max_t = 50 - t_sample

    while True:
        recon_track = []; lat_track = [] 
        for i, (x, _) in enumerate(loader):
            optim.zero_grad()
            x = x.to(hps.device)

            # sample 6 frame windows
            t0 = torch.randint(0, max_t, (x.shape[0],))
            x_ = []
            for j, t0_ in enumerate(t0):
                x_.append(x[j, t0_ : t0_+t_sample])
            x = torch.stack(x_, dim=0)

            # compress
            z = compress(x, swin_model)

            # forward pass
            z0 = z[:, 0, ...]
            t_ = torch.linspace(0, 1, t_sample).to(hps.device)
            y = odeint(op_model, z0, t_, method='midpoint')
            y = rearrange(y, 't b e -> b t e')

            # decompress
            x_recon = decompress(y, dom, siren_model)

            # loss and backprop
            lat_loss = (y - z).abs().mean()
            recon_loss = (x - x_recon).abs().mean()
            loss = lat_loss + recon_loss

            lat_track.append(lat_loss.item())
            recon_track.append(recon_loss.item())
            #recon_track.append(0)

            loss.backward()
            optim.step()

            if i % 20 == 0:
                print(f'lat_loss: {lat_loss.item()}, recon_loss: {recon_loss.item()}')
            if i % 100 == 0:
                print('saving gif')
                video_compare(x[0].detach().cpu(), x_recon[0].detach().cpu(), 'ode.gif')

                print('saving model')
                torch.save(op_model.state_dict(), f'{hps.exp_path}/swin_model.pt')

def ode_imp_test():
    pass