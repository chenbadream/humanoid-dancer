import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims=[1024, 512], activation=nn.ReLU):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(activation())
            last_dim = h
        
        # Final output layer - output raw logits (NO sigmoid activation)
        # DeepMimic discriminator outputs raw logits, not probabilities
        # The reward formula is applied directly to these logits
        self.output_layer = nn.Linear(last_dim, 1)
        layers.append(self.output_layer)
        # NO sigmoid activation here - DeepMimic uses raw logits
        self.model = nn.Sequential(*layers)
        
        # Initialize the final layer with small weights (following DeepMimic)
        # DiscInitOutputScale: 1 means initialize to small values
        nn.init.uniform_(self.output_layer.weight, -0.02, 0.02)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        return self.model(x)
