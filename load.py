import h5py
import torch
from torch.utils.data import Dataset, DataLoader

from plot import make_gif

def navier_loader(path, mode, batch_size, workers=4):
    dataset = NavierDataset(path, mode)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    return loader

class NavierDataset(Dataset):
    def __init__(self, path, mode):
        if mode == 'train': self.len = 8000
        elif mode == 'test': self.len = 2000
        else: raise ValueError('mode must be "train" or "test"')
        
        self.mode = mode
        self.f = h5py.File(path, 'r')
        self.idx_map = torch.randperm(self.len)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            x = self.f['u'][..., self.idx_map[idx]]
        else:
            idx = idx + 8000
            x = self.f['u'][..., self.idx_map[idx]]
        
        return torch.tensor(x)

if __name__ == '__main__':
    batch_size = 5
    mode = 'train'
    path = "data/navier.mat"
    train_loader = navier_loader(path, mode, batch_size)

    for i, x in enumerate(train_loader):
        imgs = x[0]
        make_gif(imgs, 'test.gif')
