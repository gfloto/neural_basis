import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from plot import make_gif

def cifar10_loader(batch_size, test, num_workers):
    train = not test
    dataset = datasets.CIFAR10(
        root='./data', train=train, 
        download=False, transform=transforms.ToTensor()
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )
    print(f'cifar10 {"test" if test else "train"} loader created')
    print(f'length: {len(loader)}')

    return loader

def navier_loader(path, mode, batch_size, num_workers, basis=False):
    '''
    basis is a flag for training basis functions
    in this case we sample across time and random seed
    '''

    dataset = NavierDataset(path, mode, basis=basis)

    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )

    return loader

class NavierDataset(Dataset):
    def __init__(self, path, mode, basis=False):
        if mode == 'train': self.len = 8000
        elif mode == 'test': self.len = 2000
        else: raise ValueError('mode must be "train" or "test"')

        self.mode = mode
        self.basis = basis
        self.f = h5py.File(path, 'r')
        self.idx_map = torch.randperm(self.len)

    def __len__(self):
        return 50*self.len
    
    def __getitem__(self, idx):
        idx = torch.randint(0, self.len, (1,)).item()
        if self.mode == 'train':
            x = self.f['u'][..., self.idx_map[idx]]
        else:
            idx = idx + 8000
            x = self.f['u'][..., self.idx_map[idx]]

        if self.basis:
            t = torch.randint(0, 50, (1,)).item()
            x = x[t][None, ...]
        
        return torch.tensor(x), torch.tensor(0)

if __name__ == '__main__':
    batch_size = 5
    mode = 'train'
    path = "data/navier.mat"
    train_loader = navier_loader(path, mode, 0, batch_size)

    for i, x in enumerate(train_loader):
        print(x.shape)
        quit()
        imgs = x[0]
        make_gif(imgs, 'test.gif')
