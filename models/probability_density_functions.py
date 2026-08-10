import torch
from torch import nn
import torch.nn.functional as F


def log_normal_pdf(x, loc, scale):
    a = -torch.log(scale)
    b = torch.log(torch.tensor(2, device=loc.device) * torch.pi)
    c = ((x - loc) / scale)**2
    return a - 0.5 * (b + c)


class DiscreteLogistic(nn.Module):
    # TODO: where is this code from? I might need to include license
    def __init__(self, min_bound=0, max_bound=255, num_vals=256):
        super().__init__()
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.num_vals = num_vals

    def forward(self, y, means, log_scales, mixture_logits):
        inv_scales = torch.exp(-log_scales).to(y.dtype)

        y_range = self.max_bound - self.min_bound
        # explained in text
        epsilon = (0.5 * y_range) / (self.num_vals - 1)
        # convenience variable
        y = y.unsqueeze(-1)
        y = torch.repeat_interleave(y, means.shape[-1], -1)
        centered_y = y - means
        # inputs to our sigmoid functions
        upper_bound_in = inv_scales * (centered_y + epsilon)
        lower_bound_in = inv_scales * (centered_y - epsilon)
        # remember: cdf of logistic distr is sigmoid of above input format
        upper_cdf = torch.sigmoid(upper_bound_in)
        lower_cdf = torch.sigmoid(lower_bound_in)
        # finally, the probability mass and equivalent log prob
        prob_mass = upper_cdf - lower_cdf
        vanilla_log_prob = torch.log(torch.clamp(prob_mass, min=1e-12)).to(y.dtype)

        # edges
        low_bound_log_prob = upper_bound_in - F.softplus(
            upper_bound_in
        ).to(y.dtype)  # log probability for edge case of 0 (before scaling)
        upp_bound_log_prob = -F.softplus(
            lower_bound_in
        ).to(y.dtype)  # log probability for edge case of 255 (before scaling)
        # middle
        mid_in = inv_scales * centered_y
        log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in).to(y.dtype)
        log_prob_mid = log_pdf_mid - torch.log((self.num_vals - 1) / 2).to(y.dtype)

        # Create a tensor with the same shape as 'y', filled with zeros
        log_probs = torch.zeros_like(centered_y)
        # conditions for filling in tensor
        is_near_min = y < self.min_bound + 1e-3
        is_near_max = y > self.max_bound - 1e-3
        is_prob_mass_sufficient = prob_mass > 1e-5 
        # And then fill it in accordingly
        # lower edge
        log_probs[is_near_min] = low_bound_log_prob[is_near_min]
        # upper edge
        log_probs[is_near_max] = upp_bound_log_prob[is_near_max]
        # vanilla case
        log_probs[~is_near_min & ~is_near_max & is_prob_mass_sufficient] = vanilla_log_prob[
            ~is_near_min & ~is_near_max & is_prob_mass_sufficient
        ]
        # extreme case where prob mass is too small
        log_probs[~is_near_min & ~is_near_max & ~is_prob_mass_sufficient] = log_prob_mid[
            ~is_near_min & ~is_near_max & ~is_prob_mass_sufficient
        ]

        # modeling which mixture to sample from
        log_probs = log_probs + F.log_softmax(mixture_logits, dim=-1, dtype=y.dtype)

        # log likelihood
        log_likelihood = torch.logsumexp(log_probs, dim=-1).to(y.dtype)

        return log_likelihood