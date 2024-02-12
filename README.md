# Neural Basis Functions

## Motivation
Functional representations are important for memory and time series prediction. Some examples:

* [Legendre Memory Units](https://papers.nips.cc/paper_files/paper/2019/hash/952285b9b7e7a1be5aa7849f32ffff05-Abstract.html)
* [Optimal Polyomial Projections](https://arxiv.org/abs/2008.07669)
* [Neural Fourier Operator](https://arxiv.org/abs/2010.08895)

Why shouldn't we learn an optimal [functional basis](https://en.wikipedia.org/wiki/Basis_function)?

## Initial Experiment
We compare to a FFT representation of CIFAR-10, using the same number of basis functions. [Siren](https://arxiv.org/abs/2006.09661) is used to represent the basis functions. Currently the model isn't effectively parallelized (ie. there is a siren model for each basis function that must be passed in a for loop).

Nevertheless, the results are promising. Using L1 and LPIPS loss, we can see that we can impliment a neural basis that outperforms the FFT representation on the task of image compression and reconstruction. 
