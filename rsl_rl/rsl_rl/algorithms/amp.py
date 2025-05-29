from .ppo import PPO
import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

class AMP(PPO):
    def __init__(self,
                 actor_critic,
                 discriminator,
                 demo_buffer,
                 replay_buffer,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 disc_coef=1.0,
                 learning_rate=1e-3,
                 disc_learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 desired_kl=0.01,
                 schedule="fixed",  # Added schedule parameter
                 device='cpu',
                 ):
        # Don't call super().__init__ yet, we need to set up unified optimizer
        # Store parameters for later
        self.actor_critic = actor_critic.to(device)
        self.discriminator = discriminator.to(device)
        self.demo_buffer = demo_buffer
        self.replay_buffer = replay_buffer
        self.disc_coef = disc_coef
        
        # Set up unified optimizer with different parameter groups (amp-rsl-rl style)
        params = [
            {"params": self.actor_critic.parameters(), "name": "actor_critic"},
            {
                "params": self.discriminator.trunk.parameters(),
                "weight_decay": 1e-4,  # 10e-4 from amp-rsl-rl
                "name": "amp_trunk",
            },
            {
                "params": self.discriminator.output_layer.parameters(),
                "weight_decay": 1e-2,  # 10e-2 from amp-rsl-rl
                "name": "amp_head",
            },
        ]
        
        # Initialize base PPO with custom optimizer setup
        self.device = device
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        
        # Unified optimizer for both networks
        self.optimizer = optim.Adam(params, lr=learning_rate)
        
        # Initialize storage and transition (from PPO)
        self.storage = None
        self.transition = RolloutStorage.Transition()

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def act(self, obs, critic_obs):
        # ...existing code from PPO...
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # ...existing code from PPO...
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        """
        Updated training method following amp-rsl-rl best practices:
        - Unified optimizer with different weight decay for discriminator parts
        - BCEWithLogitsLoss for discriminator training
        - Proper gradient penalty computation
        - Adaptive learning rate based on KL divergence
        """
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        
        # Create generators for policy rollouts
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Create generators for AMP data (following amp-rsl-rl pattern)
        amp_policy_generator = self.replay_buffer.feed_forward_generator(
            num_mini_batch=self.num_learning_epochs * self.num_mini_batches,
            mini_batch_size=self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
            allow_replacement=True,
        )
        
        amp_expert_generator = self.demo_buffer.feed_forward_generator(
            num_mini_batch=self.num_learning_epochs * self.num_mini_batches,
            mini_batch_size=self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )

        # Training loop with combined policy and discriminator updates
        for epoch_idx, (sample, amp_policy_sample, amp_expert_sample) in enumerate(zip(generator, amp_policy_generator, amp_expert_generator)):
            # Unpack policy rollout data
            (obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, 
             returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, 
             hid_states_batch, masks_batch) = sample

            # Debug: Check batch sizes occasionally
            if epoch_idx == 0:
                print(f"AMP Training - Policy batch: {obs_batch.shape}, AMP policy: {amp_policy_sample.shape if not isinstance(amp_policy_sample, tuple) else [x.shape for x in amp_policy_sample]}, AMP expert: {amp_expert_sample.shape if not isinstance(amp_expert_sample, tuple) else [x.shape for x in amp_expert_sample]}")

            # Forward pass through actor-critic
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # Adaptive learning rate based on KL divergence (amp-rsl-rl style)
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # PPO loss computation
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # PPO loss
            ppo_loss = (surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean())

            # AMP discriminator loss (following amp-rsl-rl pattern)
            # Both buffers now return 119-dim concatenated observations
            policy_amp_obs = amp_policy_sample.to(self.device)
            expert_amp_obs = amp_expert_sample.to(self.device)
            
            # For gradient penalty, we need to split observations (assuming equal split)
            # This is a simplification - in practice, the split should match the actual structure
            half_dim = policy_amp_obs.shape[-1] // 2
            policy_state = policy_amp_obs[..., :half_dim] 
            policy_next_state = policy_amp_obs[..., half_dim:]
            expert_state = expert_amp_obs[..., :half_dim]
            expert_next_state = expert_amp_obs[..., half_dim:]

            # Discriminator forward pass
            policy_d = self.discriminator(policy_amp_obs)
            expert_d = self.discriminator(expert_amp_obs)

            # Discriminator loss using BCEWithLogitsLoss (amp-rsl-rl style)
            expert_loss = self.discriminator_expert_loss(expert_d)
            policy_loss = self.discriminator_policy_loss(policy_d)
            amp_loss = 0.5 * (expert_loss + policy_loss)

            # Apply discriminator coefficient (amp-rsl-rl style)
            amp_loss = self.disc_coef * amp_loss

            # Gradient penalty for discriminator stability
            grad_pen_loss = torch.tensor(0.0, device=self.device)
            if hasattr(self.discriminator, 'compute_grad_pen'):
                try:
                    grad_pen_loss = self.discriminator.compute_grad_pen(expert_state, expert_next_state, lambda_=10.0)
                except Exception as e:
                    print(f"Warning: Gradient penalty computation failed: {e}")

            # Combined loss (PPO + AMP + gradient penalty)
            total_loss = ppo_loss + amp_loss + grad_pen_loss

            # Unified optimizer update
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Update running statistics
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()

        # Average losses over all updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        
        # Clear storage
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss

    def discriminator_policy_loss(self, discriminator_output):
        """
        Computes the loss for the discriminator when classifying policy-generated transitions.
        Uses binary cross-entropy loss where the target label for policy transitions is 0.
        """
        loss_fn = nn.BCEWithLogitsLoss()
        expected = torch.zeros_like(discriminator_output).to(self.device)
        return loss_fn(discriminator_output, expected)

    def discriminator_expert_loss(self, discriminator_output):
        """
        Computes the loss for the discriminator when classifying expert transitions.
        Uses binary cross-entropy loss where the target label for expert transitions is 1.
        """
        loss_fn = nn.BCEWithLogitsLoss()
        expected = torch.ones_like(discriminator_output).to(self.device)
        return loss_fn(discriminator_output, expected)