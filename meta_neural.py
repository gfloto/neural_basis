import torch
from torch import nn
from torchvision.models import swin_v2_s

from models.orig_siren import SirenNet

class Swin(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.swin = swin_v2_s()
        self.linear = nn.Linear(1000, dim)

    def forward(self, x):
        x = self.swin(x)
        x = self.linear(x)
        return x

class ImplicitSiren(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, n_layers):
        super().__init__()
        self.siren = SirenNet(
            dim_in=2*dim_in, dim_hidden=dim_hidden,
            dim_out=dim_out, num_layers=n_layers,
        )

        self.linear = nn.Linear(2, dim_in)

    def forward(self, dom, z):
        proj_dom = self.linear(dom)
        x = torch.cat([proj_dom, z], dim=-1)
        return self.siren(x)

if __name__ == '__main__':
    device = 'cuda'
    swin = swin_v2_s().to(device)

    x = torch.rand(1, 3, 64, 64).to(device)
    swin = Swin(48).to(device)
    y = swin(x)
    print(y.shape)