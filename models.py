import torch 
import torch.nn as nn 
from torchvision.models import swin_v2_s

class CoeffModel(nn.Module):
    def __init__(self, n_basis):
        super().__init__()
        '''
        take pretrained swin model different logits
        '''

        # use pretrained swin_v2_s
        self.swin = swin_v2_s()
        self.linear = nn.Linear(1000, 3 * 3*n_basis**2)

    def forward(self, x):
        x = self.swin(x)
        x = self.linear(x)
        x = torch.sigmoid(x)
        x = 2*x - 1
        return x

