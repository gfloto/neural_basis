import torch

class LegendreBasis:
    def __init__(self, n=8, T=50):
        self.n = n# number of basis functions (different from n_basis for spatial basis)
        self.T = T
        self.basis = self.make_basis().cuda()

    def make_basis(self):
        x = torch.linspace(-0.9, 0.9, self.T)
        basis = torch.zeros((self.n, self.T))
        basis[0] = 1.
        basis[1] = x

        for i in range(2, self.n):
            basis[i] = ((2*i - 1) * x * basis[i-1] - (i-1) * basis[i-2]) / i
        
        return basis

    def recon_error(self, x, eigen_f):
        # get eigen-values of x from most recent eigen-function
        c = torch.einsum('b t h w, b t h w -> b t', x, eigen_f)
        c = (c - c.mean()) / c.std()

        # ensure that eigen-value function through time is well
        # represented by the legendre basis
        coeffs = torch.einsum('b t, l t -> b l', c, self.basis)
        recon = torch.einsum('b l, l t -> b t', coeffs, self.basis)

        return c - recon