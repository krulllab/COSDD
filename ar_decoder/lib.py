import torch
from torch import nn
from torch.distributions import Categorical, Normal, MixtureSameFamily


class Rotate90(nn.Module):
    def __init__(self, k, dims):
        super().__init__()
        self.k = k
        self.dims = dims

    def forward(self, x, s_code):
        x = torch.rot90(x, k=self.k, dims=self.dims)
        s_code = torch.rot90(s_code, k=self.k, dims=self.dims)
        return x, s_code


def sample_mixture_model(logweights, loc, scale):
    px = MixtureSameFamily(Categorical(logits=logweights), Normal(loc, scale))
    return px.sample()
