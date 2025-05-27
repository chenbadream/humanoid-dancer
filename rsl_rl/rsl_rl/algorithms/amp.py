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
                demo_obs_t, demo_obs_tp1 = self.demo_buffer.sample_pair(obs_batch.shape[0])  # implement sample_pair for consecutive demo frames
                agent_obs_t, agent_obs_tp1 = self.replay_buffer.sample_pair(obs_batch.shape[0])  # implement sample_pair for consecutive agent frames

                # --- Compose 119-dim AMP obs using actual reference state ---
                def make_amp_obs_pair(s_t, s_tp1):
                    # Ensure all tensors are on the same device
                    if s_tp1.device != s_t.device:
                        s_tp1 = s_tp1.to(s_t.device)
                    # Slicing for s_t
                    base_ang_vel = s_t[:, 0:3]
                    projected_gravity = s_t[:, 3:6]
                    commands = s_t[:, 6:9]
                    dof_pos = s_t[:, 9:28]
                    dof_vel = s_t[:, 28:47]
                    actions = s_t[:, 47:66] if s_t.shape[1] >= 66 else torch.zeros((s_t.shape[0], 19), device=s_t.device, dtype=s_t.dtype)
                    base_lin_vel = s_t[:, 66:69] if s_t.shape[1] >= 69 else torch.zeros((s_t.shape[0], 3), device=s_t.device, dtype=s_t.dtype)
                    # Slicing for s_tp1
                    ref_dof_pos = s_tp1[:, 9:28]
                    ref_dof_vel = s_tp1[:, 28:47]
                    ref_base_lin_vel = s_tp1[:, 66:69] if s_tp1.shape[1] >= 69 else torch.zeros((s_tp1.shape[0], 3), device=s_tp1.device, dtype=s_tp1.dtype)
                    ref_base_ang_vel = s_tp1[:, 0:3]
                    dof_diff = ref_dof_pos - dof_pos
                    dof_vel_diff = ref_dof_vel - dof_vel
                    diff_local_root_vel = ref_base_lin_vel - base_lin_vel
                    diff_local_root_ang_vel = ref_base_ang_vel - base_ang_vel
                    # --- Compute tan-norm (body orientation difference) ---
                    if s_t is s_tp1:
                        tan_norm = torch.zeros((s_t.shape[0], 8), device=s_t.device, dtype=s_t.dtype)
                    else:
                        if s_t.shape[1] >= 73 and s_tp1.shape[1] >= 73:
                            root_quat = s_t[:, -4:]
                            ref_body_rot = s_tp1[:, -4:]
                            import legged_gym.utils.torch_utils as torch_utils
                            heading_inv_rot = torch_utils.calc_heading_quat_inv(root_quat)
                            heading_rot = torch_utils.calc_heading_quat(root_quat)
                            diff_global_body_rot = torch_utils.quat_mul(ref_body_rot, torch_utils.quat_conjugate(root_quat))
                            diff_local_body_rot_flat = torch_utils.quat_mul(
                                torch_utils.quat_mul(heading_inv_rot, diff_global_body_rot), heading_rot)
                            tan_norm = torch_utils.quat_to_tan_norm(diff_local_body_rot_flat)
                        else:
                            tan_norm = torch.zeros((s_t.shape[0], 8), device=s_t.device, dtype=s_t.dtype)
                    obs = torch.cat([
                        base_ang_vel, projected_gravity, commands, dof_pos, dof_vel, actions, base_lin_vel,
                        tan_norm, diff_local_root_vel, diff_local_root_ang_vel, dof_diff, dof_vel_diff
                    ], dim=-1)
                    return obs

                # For demo: use consecutive demo frames
                demo_obs_full = make_amp_obs_pair(demo_obs_t, demo_obs_tp1)
                # For agent: use consecutive agent frames
                agent_obs_full = make_amp_obs_pair(agent_obs_t, agent_obs_tp1)
                # Ensure both are on the same device as the discriminator
                demo_obs_full = demo_obs_full.to(self.device)
                agent_obs_full = agent_obs_full.to(self.device)

                disc_demo_logits = self.discriminator(demo_obs_full)
                disc_agent_logits = self.discriminator(agent_obs_full)
                bce = nn.BCEWithLogitsLoss()
                disc_loss = 0.5 * (bce(disc_agent_logits, torch.zeros_like(disc_agent_logits)) +
                                   bce(disc_demo_logits, torch.ones_like(disc_demo_logits)))

                # Total loss
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + self.disc_coef * disc_loss

                # Update actor-critic
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update discriminator
                self.optimizer_disc.zero_grad()
                disc_loss.backward()
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