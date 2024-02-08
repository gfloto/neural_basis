import torch 
import torch.fft as fft 

from main import cifar10_loader

'''
slow discrete fourier transform using torch
'''

if __name__ == '__main__':
    # get image from cifar10 loader
    loader = cifar10_loader(1)
    x, _ = next(iter(loader))

    print(x.shape)
    quit()
