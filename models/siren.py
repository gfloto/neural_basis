import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

# helpers
def exists(val):
    return val is not None

# sin activation
class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0.cuda()
    def forward(self, x):
        return torch.sin(self.w0[None, :, None] * x)

def identity(x):
    return x

# siren layer
class Siren(nn.Module):
    def __init__(
        self,
        ensembles,
        dim_in,
        dim_out,
        w0,
        is_first = False,
        use_bias = True,
        activation = None,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight, bias = [], []
        for i in range(ensembles):
            weight.append(torch.zeros(dim_out, dim_in))
            if use_bias:
                bias.append(torch.zeros(dim_out))

        self.init_(weight, bias, w0 = w0)

        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, w0, c=6):
        dim = self.dim_in
        self.weight, self.bias = [], []

        for i in range(w0.shape[0]):
            w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0[i])
            weight[i].uniform_(-w_std, w_std)

            if len(bias) > 0:
                bias[i].uniform_(-w_std, w_std)

        # convert to nn.Parameter
        self.weight = nn.Parameter(torch.stack(weight, dim=0))
        if len(bias) > 0:
            self.bias = nn.Parameter(torch.stack(bias, dim=0))
        else:
            self.bias = None

    def forward(self, x):
        y = torch.einsum('e o i, b e i -> b e o', self.weight, x)
        if exists(self.bias):
            y += self.bias
        return self.activation(y)
        

        #out = []
        #for i in range(self.weight.shape[0]):
            #y = F.linear(x[:,i], self.weight[i], self.bias[i])
            #y = self.activation(y, i)
            #out.append(y)
        
        #return torch.stack(out, dim=1)

# siren network

class SirenNet(nn.Module):
    def __init__(
        self,
        ensembles,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0,
        w0_initial = 30.,
        use_bias = True,
        final_activation = None,
        dropout = 0.
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layer = Siren(
                ensembles = ensembles,
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first,
            )

            self.layers.append(layer)

        final_activation = identity if not exists(final_activation) else final_activation

        self.last_layer = Siren(
            ensembles=ensembles,
            dim_in = dim_hidden,
            dim_out = dim_out,
            w0 = w0,
            use_bias = use_bias,
            activation = final_activation
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.last_layer(x)
        