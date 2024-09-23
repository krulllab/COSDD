import torch
from torch import nn
from torch.distributions import Categorical, Normal


class Rotate90(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x, s_code):
        return torch.rot90(x, k=self.k, dims=[2, 3]), torch.rot90(
            s_code, k=self.k, dims=[2, 3]
        )


def sample_mixture_model(weights, loc, scale):
    mixture_samples = Normal(loc, scale).rsample()
    component_idx = Categorical(weights).sample()

    return torch.gather(mixture_samples, dim=-1, index=component_idx.unsqueeze(dim=-1))
