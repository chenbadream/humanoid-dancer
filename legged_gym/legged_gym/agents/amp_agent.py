import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any

class AMPAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda'))
        
        # Initialize networks
        self._init_networks()
        
        # Initialize optimizers
        self._init_optimizers()
        
        # Initialize buffers
        self._init_buffers()
        
    def _init_networks(self):
        """Initialize actor, critic and discriminator networks"""
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(self.config['obs_dim'], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.config['action_dim'])
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(self.config['obs_dim'], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Discriminator network
        self.discriminator = nn.Sequential(
            nn.Linear(self.config['amp_obs_dim'], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Move networks to device
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.discriminator.to(self.device)
        
    def _init_optimizers(self):
        """Initialize optimizers for all networks"""
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.config.get('actor_lr', 3e-4)
        )
        
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config.get('critic_lr', 3e-4)
        )
        
        self.disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.get('disc_lr', 3e-4)
        )
        
    def _init_buffers(self):
        """Initialize replay buffers"""
        self.amp_obs_demo_buffer = ReplayBuffer(
            self.config['amp_obs_demo_buffer_size'],
            self.device
        )
        
        self.amp_replay_buffer = ReplayBuffer(
            self.config['amp_replay_buffer_size'],
            self.device
        )
        
    def update(self, batch: Dict[str, torch.Tensor]):
        """Update networks using a batch of data"""
        # Update actor and critic
        self._update_actor_critic(batch)
        
        # Update discriminator
        self._update_discriminator(batch)
        
    def _update_actor_critic(self, batch: Dict[str, torch.Tensor]):
        """Update actor and critic networks"""
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']
        
        # Compute critic loss
        current_q = self.critic(obs)
        next_q = self.critic(next_obs)
        target_q = rewards + (1 - dones) * self.config['gamma'] * next_q
        critic_loss = F.mse_loss(current_q, target_q.detach())
        
        # Compute actor loss
        actor_loss = -self.critic(obs).mean()
        
        # Update networks
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
    def _update_discriminator(self, batch: Dict[str, torch.Tensor]):
        """Update discriminator network"""
        amp_obs = batch['amp_obs']
        amp_obs_demo = batch['amp_obs_demo']
        amp_obs_replay = batch['amp_obs_replay']
        
        # Compute discriminator loss
        disc_agent_logit = self.discriminator(amp_obs)
        disc_agent_replay_logit = self.discriminator(amp_obs_replay)
        disc_demo_logit = self.discriminator(amp_obs_demo)
        
        disc_loss_agent = self._disc_loss_neg(torch.cat([disc_agent_logit, disc_agent_replay_logit]))
        disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)
        
        # Add regularization
        if self.config['disc_logit_reg'] > 0:
            logit_weights = self.discriminator.parameters()
            disc_logit_loss = torch.sum(torch.square(logit_weights))
            disc_loss += self.config['disc_logit_reg'] * disc_logit_loss
            
        # Add gradient penalty
        if self.config['disc_grad_penalty'] > 0:
            disc_demo_grad = torch.autograd.grad(
                disc_demo_logit, amp_obs_demo,
                grad_outputs=torch.ones_like(disc_demo_logit),
                create_graph=True, retain_graph=True
            )[0]
            disc_grad_penalty = torch.mean(torch.sum(torch.square(disc_demo_grad), dim=-1))
            disc_loss += self.config['disc_grad_penalty'] * disc_grad_penalty
            
        # Update discriminator
        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()
        
    def _disc_loss_neg(self, disc_logits):
        """Compute negative discriminator loss"""
        return F.binary_cross_entropy_with_logits(
            disc_logits,
            torch.zeros_like(disc_logits)
        )
        
    def _disc_loss_pos(self, disc_logits):
        """Compute positive discriminator loss"""
        return F.binary_cross_entropy_with_logits(
            disc_logits,
            torch.ones_like(disc_logits)
        )
        
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action from actor network"""
        with torch.no_grad():
            action = self.actor(obs)
            if not deterministic:
                action += torch.randn_like(action) * self.config.get('action_noise', 0.1)
            return action
            
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value from critic network"""
        with torch.no_grad():
            return self.critic(obs)
            
    def get_disc_reward(self, amp_obs: torch.Tensor) -> torch.Tensor:
        """Get discriminator reward"""
        with torch.no_grad():
            disc_logits = self.discriminator(amp_obs)
            prob = torch.sigmoid(disc_logits)
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            return disc_r * self.config['disc_reward_scale']

class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.position = 0
        
    def store(self, data: Dict[str, torch.Tensor]):
        """Store data in buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch from buffer"""
        indices = np.random.choice(len(self.buffer), batch_size)
        batch = {k: torch.cat([self.buffer[i][k] for i in indices]) for k in self.buffer[0].keys()}
        return batch 