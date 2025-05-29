import torch
import torch.nn as nn
from torch import autograd

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims=[1024, 512], activation=nn.ReLU, 
                 reward_scale=2.0, reward_clamp_epsilon=0.0001):
        super().__init__()
        self.reward_scale = reward_scale
        self.reward_clamp_epsilon = reward_clamp_epsilon
        
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(activation())
            last_dim = h
        
        # Build trunk and separate output layer for better control
        self.trunk = nn.Sequential(*layers)
        self.output_layer = nn.Linear(last_dim, 1)
        
        # Initialize the final layer with small weights (following DeepMimic and amp-rsl-rl)
        nn.init.uniform_(self.output_layer.weight, -0.02, 0.02)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        h = self.trunk(x)
        d = self.output_layer(h)
        return d
    
    def compute_grad_pen(self, expert_state, expert_next_state, lambda_=10.0):
        """
        Computes gradient penalty for WGAN-GP style training.
        Improved discriminator regularization from amp-rsl-rl.
        """
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True

        disc = self.forward(expert_data)
        ones = torch.ones(disc.size(), device=disc.device)

        grad = autograd.grad(
            outputs=disc,
            inputs=expert_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen
    
    def predict_reward(self, state, next_state, normalizer=None):
        """
        Improved reward prediction using log-based formulation from amp-rsl-rl.
        This is more stable than the DeepMimic quadratic formula.
        """
        with torch.no_grad():
            if normalizer is not None:
                state = normalizer.normalize(state)
                next_state = normalizer.normalize(next_state)

            discriminator_logit = self.forward(torch.cat([state, next_state], dim=-1))
            prob = torch.sigmoid(discriminator_logit)

            # Use log-based reward formulation for better stability
            # reward = -log(1 - p) where p is the probability of being fake
            reward = -torch.log(
                torch.maximum(
                    1 - prob,
                    torch.tensor(self.reward_clamp_epsilon, device=prob.device),
                )
            )

            reward = self.reward_scale * reward
            return reward.squeeze()
