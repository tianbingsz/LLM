import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical


class MLP(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, act_dim, out_activation) -> None:
        super().__init__()
        print(f"obs_dim: {obs_dim}, hidden_sizes: {hidden_sizes}, act_dim: {act_dim}")
        self.layers = nn.ModuleList(
            [
                nn.Linear(obs_dim, hidden_sizes),
                nn.Linear(hidden_sizes, act_dim)
            ]
        )
        self.activation = nn.ReLU()
        self.out_act = out_activation


        # init
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                nn.init.zeros_(layer.bias)



    def forward(self, obs):
        """
            obs: (b, dim)
        """
        x = obs
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)

        return  self.out_act(x) if self.out_act is not None else x



class BasePolicy(nn.Module):
    def __init__(self):
        super().__init__()

    def get_policy_distribution(self, obs: torch.Tensor):
        """Return torch.distributions.Distribution object."""
        raise NotImplementedError

    def get_action(self, obs: torch.Tensor):
        """Sample action for env.step()."""
        raise NotImplementedError

    def get_log_prob(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Log π(a|s) for given actions."""
        dist = self.get_policy_distribution(obs)
        return dist.log_prob(act) # (b, a_d)

    def compute_loss(self, obs: torch.Tensor, act: torch.Tensor, advs: torch.Tensor):
        """Vanilla policy gradient objective."""
        logp = self.get_log_prob(obs, act) # (b, a_d)
        return -(logp * advs).mean() # scalar


class GaussianPolicy(BasePolicy):
    def __init__(self, obs_dim, hidden_sizes, act_dim, out_activation):
        super().__init__()

        self.mu = MLP(obs_dim, hidden_sizes, act_dim, out_activation)
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim)) # -0.5 * I_d

    def get_policy_distribution(self, obs: torch.Tensor):
        return Normal(self.mu(obs), torch.exp(self.log_std))

    def get_action(self, obs: torch.Tensor)-> np.ndarray:
        dist = self.get_policy_distribution(obs)
        return dist.sample().detach().numpy()

    def get_log_prob(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return super().get_log_prob(obs, act).sum(dim=-1) # (b,)

    def compute_loss(self, obs: torch.Tensor, act: torch.Tensor, advs: torch.Tensor):
        return super().compute_loss(obs, act, advs) # (b, )


class CategoricalPolicy(BasePolicy):
    def __init__(self, obs_dim, hidden_sizes, act_dim, out_activation):
        super().__init__()
        self.logits = MLP(obs_dim, hidden_sizes, act_dim, out_activation)


    def get_policy_distribution(self, obs: torch.Tensor):
        return Categorical(logits=self.logits(obs))

    def get_action(self, obs: torch.Tensor) -> int:
        dist = self.get_policy_distribution(obs)
        return dist.sample().item()

    def get_log_prob(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        dist = self.get_policy_distribution(obs)
        return dist.log_prob(act) # (b,)

    def compute_loss(self, obs: torch.Tensor, act: torch.Tensor, advs: torch.Tensor):
        logp = self.get_log_prob(obs, act) # (b,)
        return -(logp * advs).mean() # scalar



class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, out_activation):
        super().__init__()
        # output float value
        self.value_func = MLP(obs_dim, hidden_sizes, 1, out_activation)


    def forward(self, obs):
        """
            V(s_t)
            obs: (b, o_dim)
            output: value (b, 1)
        """
        value = self.value_func(obs)
        return value.squeeze(-1) #(b, )

    def compute_loss(self, obs, ret):
        """
            loss = MSE(val, ret)
        """
        value = self.forward(obs) # (b, 1)
        loss = F.mse_loss(value, ret) # scalar
        return loss
