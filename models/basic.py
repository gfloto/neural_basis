import torch.nn as nn 

'''
a basic resnet with for 1d data...
'''

class Block(nn.Module):
    def __init__(self, dim):
        super(Block, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x_):
        x = self.fc1(x_)
        x = self.bn(x)
        x = nn.functional.elu(x)
        x = self.fc2(x)

        return x + x_

class Basic(nn.Module):
    def __init__(self, dim, dim_hidden=256, blocks=2):
        super(Basic, self).__init__()
        self.proj_in = nn.Linear(dim, dim_hidden)
        self.proj_out = nn.Linear(dim_hidden, dim)

        self.blocks = nn.ModuleList([
            Block(dim_hidden) for _ in range(blocks)
        ])
    
    def forward(self, t, x):
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x)
        x = self.proj_out(x)
        return x
