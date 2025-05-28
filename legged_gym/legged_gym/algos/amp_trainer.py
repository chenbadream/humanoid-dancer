import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np

class AMPTrainer:
    def __init__(self, 
                 policy: nn.Module,
                 discriminator: nn.Module,
                 policy_optimizer: torch.optim.Optimizer,
                 disc_optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 cfg: Dict):
        self.policy = policy
        self.discriminator = discriminator
        self.policy_optimizer = policy_optimizer
        self.disc_optimizer = disc_optimizer
        self.device = device
        self.cfg = cfg
        
        # Training parameters
        self.disc_coef = cfg.get('disc_coef', 0.5)
        self.policy_epochs = cfg.get('policy_epochs', 10)
        self.disc_epochs = cfg.get('disc_epochs', 5)
        self.batch_size = cfg.get('batch_size', 4096)
        
        # Storage for expert demonstrations
        self.expert_buffer = []
        self.agent_buffer = []
        
    def store_expert_demo(self, obs: torch.Tensor):
        """Store expert demonstration data"""
        self.expert_buffer.append(obs.detach().cpu())
        
    def store_agent_data(self, obs: torch.Tensor, rewards: torch.Tensor):
        """Store agent rollout data"""
        self.agent_buffer.append((obs.detach().cpu(), rewards.detach().cpu()))
        
    def clear_buffers(self):
        """Clear stored data"""
        self.expert_buffer = []
        self.agent_buffer = []
        
    def compute_discriminator_reward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute discriminator reward for given observations"""
        with torch.no_grad():
            disc_logits = self.discriminator(obs.to(self.device))
            # DeepMimic-style bonus
            bonus = -torch.log1p(torch.exp(-disc_logits)).squeeze(1)
        return bonus.cpu()
    
    def update_policy(self, obs: torch.Tensor, actions: torch.Tensor, 
                     old_values: torch.Tensor, old_log_probs: torch.Tensor,
                     advantages: torch.Tensor, returns: torch.Tensor):
        """Update policy network"""
        self.policy.train()
        
        for _ in range(self.policy_epochs):
            # Compute discriminator rewards for current batch
            disc_rewards = self.compute_discriminator_reward(obs)
            
            # Combine environment and discriminator rewards
            total_rewards = returns + self.disc_coef * disc_rewards
            
            # Compute new values and log probs
            new_values = self.policy.get_value(obs)
            new_actions, new_log_probs, _ = self.policy.act(obs)
            
            # Compute PPO loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(new_values, total_rewards)
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss
            
            # Optimize
            self.policy_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()
            
    def update_discriminator(self):
        """Update discriminator network"""
        if not self.expert_buffer or not self.agent_buffer:
            return
            
        self.discriminator.train()
        
        # Prepare expert and agent data
        expert_obs = torch.cat(self.expert_buffer, dim=0)
        agent_obs = torch.cat([obs for obs, _ in self.agent_buffer], dim=0)
        
        # Ensure balanced batches
        min_size = min(len(expert_obs), len(agent_obs))
        expert_obs = expert_obs[:min_size]
        agent_obs = agent_obs[:min_size]
        
        for _ in range(self.disc_epochs):
            # Sample batches
            indices = torch.randperm(min_size)
            for start_idx in range(0, min_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, min_size)
                batch_indices = indices[start_idx:end_idx]
                
                expert_batch = expert_obs[batch_indices].to(self.device)
                agent_batch = agent_obs[batch_indices].to(self.device)
                
                # Compute discriminator outputs
                expert_logits = self.discriminator(expert_batch)
                agent_logits = self.discriminator(agent_batch)
                
                # Compute discriminator loss
                expert_loss = F.binary_cross_entropy_with_logits(
                    expert_logits, torch.ones_like(expert_logits))
                agent_loss = F.binary_cross_entropy_with_logits(
                    agent_logits, torch.zeros_like(agent_logits))
                disc_loss = expert_loss + agent_loss
                
                # Optimize discriminator
                self.disc_optimizer.zero_grad()
                disc_loss.backward()
                self.disc_optimizer.step()
                
    def train_step(self, obs: torch.Tensor, actions: torch.Tensor,
                  old_values: torch.Tensor, old_log_probs: torch.Tensor,
                  advantages: torch.Tensor, returns: torch.Tensor):
        """Perform one training step with separate policy and discriminator updates"""
        # First update policy
        self.update_policy(obs, actions, old_values, old_log_probs, advantages, returns)
        
        # Then update discriminator
        self.update_discriminator()
        
        # Clear buffers after training
        self.clear_buffers() 