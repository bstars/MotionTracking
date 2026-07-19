import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def construct_mlp(dims, zero_final_layer=False):
    layers = []
    
    # hidden layers
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(nn.ELU())
        # layers.append(nn.SiLU())

    final_layer = nn.Linear(dims[-2], dims[-1])
    
    if zero_final_layer:
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
    
    layers.append(final_layer)
    
    return nn.Sequential(*layers)


class V(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.state_dim = state_dim
        self.net = construct_mlp([state_dim] + hidden_dim + [1], zero_final_layer=True)
    
    def forward(self, states):
        return self.net(states).squeeze(-1)


class Pi(nn.Module):
    """
    A policy that output the a distribution
    For PPO, output a unbounded gaussian 
    """
    def __init__(self, state_dim, action_dim, hidden_dim, init_log_std=0.):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = construct_mlp([state_dim] + hidden_dim + [action_dim], zero_final_layer=True)
        self.log_std = nn.Parameter(torch.ones(action_dim) * init_log_std)

    def forward(self, states):
        mean = self.net(states)
        std = self.log_std.exp()

        normal = Normal(mean, std)
        actions = normal.rsample()

        
        log_probs = normal.log_prob(actions).sum(-1)
        return normal, actions, log_probs


class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4, device="cpu"):
        self.device = device
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = epsilon

    @torch.no_grad()
    def update(self, x):
        """
        Update running mean and variance using a batch of samples.

        Parameters
        ----------
        x : torch.Tensor
            Shape: (batch_size, *shape)
        """
        x = x.to(dtype=torch.float32, device=self.device)

        # Batch statistics
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)  # population variance
        batch_count = float(x.shape[0])

        # Difference between batch mean and current mean
        delta = batch_mean - self.mean

        # Total number of samples
        total_count = self.count + batch_count

        # Update mean
        new_mean = self.mean + delta * (batch_count / total_count)

        # Convert variances to sums of squared deviations
        old_M2 = self.var * self.count
        batch_M2 = batch_var * batch_count

        # Correction for difference in means
        correction = delta.pow(2) * (self.count * batch_count / total_count)

        # Combined sum of squared deviations
        new_M2 = old_M2 + batch_M2 + correction

        # Convert back to variance
        new_var = new_M2 / total_count

        # Store updated statistics
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count = total_count

    def normalize(self, x, eps=1e-8):
        """
        Normalize data using current running statistics.
        """
        x = x.to(dtype=torch.float32, device=self.device)
        return (x - self.mean) / torch.sqrt(self.var + eps)