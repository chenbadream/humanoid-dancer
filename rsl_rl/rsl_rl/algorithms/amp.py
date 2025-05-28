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
        super().__init__(actor_critic, num_learning_epochs=num_learning_epochs, num_mini_batches=num_mini_batches,
                         clip_param=clip_param, gamma=gamma, lam=lam, value_loss_coef=value_loss_coef,
                         entropy_coef=entropy_coef, learning_rate=learning_rate, max_grad_norm=max_grad_norm,
                         use_clipped_value_loss=use_clipped_value_loss, desired_kl=desired_kl, schedule=schedule, device=device)
        self.discriminator = discriminator.to(self.device)
        self.demo_buffer = demo_buffer
        self.replay_buffer = replay_buffer
        self.disc_coef = disc_coef
        self.optimizer_disc = optim.Adam(self.discriminator.parameters(), lr=disc_learning_rate)

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
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_disc_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

                # PPO update (same as PPO)
                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # Surrogate loss
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

                # AMP: Discriminator update
                # Sample from demo buffer and replay buffer
                demo_obs = self.demo_buffer.sample(obs_batch.shape[0])  # already constructed AMP observations
                agent_obs = self.replay_buffer.sample(obs_batch.shape[0])  # already constructed AMP observations
                
                # Ensure both are on the same device as the discriminator
                demo_obs = demo_obs.to(self.device)
                agent_obs = agent_obs.to(self.device)

                disc_demo_logits = self.discriminator(demo_obs)  # Now outputs raw logits
                disc_agent_logits = self.discriminator(agent_obs)  # Now outputs raw logits
                
                # Add safety checks for extreme values
                if torch.any(torch.isnan(disc_demo_logits)) or torch.any(torch.isnan(disc_agent_logits)):
                    print("Warning: NaN detected in discriminator outputs, skipping update")
                    continue
                
                # Clamp logits to prevent extreme values
                disc_demo_logits = torch.clamp(disc_demo_logits, min=-10.0, max=10.0)
                disc_agent_logits = torch.clamp(disc_agent_logits, min=-10.0, max=10.0)
                
                # Use DeepMimic's custom discriminator loss (NOT BCEWithLogitsLoss)
                # Expert data should have logits close to +1, agent data should have logits close to -1
                # DeepMimic loss: expert_loss = 0.5 * (logits - 1)^2, agent_loss = 0.5 * (logits + 1)^2
                disc_loss_expert = 0.5 * torch.mean(torch.square(disc_demo_logits - 1.0))
                disc_loss_agent = 0.5 * torch.mean(torch.square(disc_agent_logits + 1.0))
                disc_loss = disc_loss_expert + disc_loss_agent
                
                # Add discriminator regularization terms (following DeepMimic)
                # L2 regularization on discriminator output weights
                disc_weight_decay = 0.0005  # DiscWeightDecay from DeepMimic
                disc_logit_reg_weight = 0.05  # DiscLogitRegWeight from DeepMimic
                
                # L2 regularization on discriminator weights
                disc_weight_loss = 0.0
                for param in self.discriminator.parameters():
                    disc_weight_loss += torch.sum(param ** 2)
                disc_loss += disc_weight_decay * disc_weight_loss
                
                # L2 regularization specifically on output layer (logit regularization)
                output_weight = self.discriminator.output_layer.weight
                disc_loss += disc_logit_reg_weight * torch.sum(output_weight ** 2)
                
                # Clamp discriminator loss to prevent instability
                disc_loss = torch.clamp(disc_loss, max=10.0)

                # Total PPO loss (without discriminator - train separately)
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Update actor-critic first
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update discriminator separately (multiple steps like DeepMimic)
                for _ in range(2):  # DiscStepsPerBatch: 2 from DeepMimic
                    # Re-sample for discriminator update
                    demo_obs_disc = self.demo_buffer.sample(obs_batch.shape[0])
                    agent_obs_disc = self.replay_buffer.sample(obs_batch.shape[0])
                    demo_obs_disc = demo_obs_disc.to(self.device)
                    agent_obs_disc = agent_obs_disc.to(self.device)
                    
                    disc_demo_logits_new = self.discriminator(demo_obs_disc)
                    disc_agent_logits_new = self.discriminator(agent_obs_disc)
                    
                    # DeepMimic discriminator loss
                    disc_loss_expert_new = 0.5 * torch.mean(torch.square(disc_demo_logits_new - 1.0))
                    disc_loss_agent_new = 0.5 * torch.mean(torch.square(disc_agent_logits_new + 1.0))
                    disc_loss_new = disc_loss_expert_new + disc_loss_agent_new
                    
                    # Add regularization
                    disc_weight_loss_new = 0.0
                    for param in self.discriminator.parameters():
                        disc_weight_loss_new += torch.sum(param ** 2)
                    disc_loss_new += disc_weight_decay * disc_weight_loss_new
                    
                    output_weight_new = self.discriminator.output_layer.weight
                    disc_loss_new += disc_logit_reg_weight * torch.sum(output_weight_new ** 2)
                    
                    disc_loss_new = torch.clamp(disc_loss_new, max=10.0)
                    
                    self.optimizer_disc.zero_grad()
                    disc_loss_new.backward()
                    nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
                    self.optimizer_disc.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_disc_loss += disc_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_disc_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_disc_loss