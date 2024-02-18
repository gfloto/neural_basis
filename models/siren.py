import math
import torch
import torch.nn as nn
from torch.distributions.exponential import Exponential

def exists(val):
    return val is not None

class Sine(nn.Module):
    def __init__(self, w0):
        super().__init__()
        self.w0 = w0.cuda()
    def forward(self, x):
        return torch.sin(x * self.w0[None, :, None])

class SirenLayer(nn.Module):
    def __init__(
            self,
            ensembles,
            dim_in,
            dim_out,
            w0,
            layer,
            c = 6.,
            is_first = False,
            use_bias = True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(ensembles, dim_in, dim_out)
        bias = torch.zeros(ensembles, dim_out) if use_bias else None
        self.init_(weight, bias, w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None

        self.activation = Sine(w0)


    def init_(self, weight, bias, w0, c=6):
        dim = self.dim_in

        for i in range(weight.shape[0]):
            w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0[i])

            weight[i].uniform_(-w_std, w_std)

            if exists(bias):
                bias[i].uniform_(-w_std, w_std)

    def forward(self, x):
        out = torch.einsum('eio,bei->beo', self.weight, x)
        o_ = x[:,0,:] @ self.weight[0]
        o = out[:,0,:]

        if exists(self.bias):
            out += self.bias[None, ...]
        
        out = self.activation(out)
        return out


class SirenNet(nn.Module):
    def __init__(
            self,
            ensembles,
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0_initial,
            w0,
            use_bias = True,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            is_first = i == 0
            is_last = i == num_layers - 1

            layer_dim_in = dim_in if is_first else dim_hidden
            layer_dim_out = dim_out if is_last else dim_hidden
            layer_w0 = w0_initial if is_first else w0

            layer = SirenLayer(
                ensembles = ensembles,
                dim_in = layer_dim_in,
                dim_out = layer_dim_out,
                w0 = layer_w0,
                layer=i,
                use_bias = use_bias,
                is_first = is_first,
            )

            self.layers.append(layer)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

from einops import repeat
if __name__ == '__main__':
    batch_size = 128
    ensembles = 6
    w0 = torch.tensor([(i+1) for i in range(ensembles)])
    w0_initial = torch.ones((ensembles,))

    model = SirenNet(
        ensembles = ensembles,
        dim_in = 1,
        dim_hidden = 64,
        dim_out = 1,
        num_layers = 3,
        w0_initial = w0_initial,
        w0 = w0,
    )

    x = torch.linspace(0, 1, batch_size)
    x = repeat(x, 'b -> b e 1', e=ensembles)
    out = model(x)
    
    x = x[..., 0].detach()
    out = out[..., 0].detach()

    # plot outputs
    import matplotlib.pyplot as plt
    for i in range(ensembles):
        plt.plot(x[:, i], out[:, i])
    
    plt.savefig('siren.png')
        