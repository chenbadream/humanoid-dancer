import numpy as np
import torch
from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import to_torch
from loguru import logger
from legged_gym.motions.motion_lib_h1 import MotionLibH1
from legged_gym.utils import torch_utils
from .h1_robot import H1Robot
from .h1_amp_config import Motion
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
from .h1_mimic import H1Mimic

class H1AMP(H1Mimic):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # Set amp_obs_dim BEFORE calling super().__init__() since _get_noise_scale_vec() needs it
        self.amp_obs_dim = 105  # Updated AMP observation dimension for humanoid
        
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # AMP-specific initialization
        self.amp_obs_buf = torch.zeros(self.num_envs, self.amp_obs_dim, dtype=torch.float, device=self.device, requires_grad=False)
        
        # Store previous states for AMP observation construction
        self.prev_dof_pos = torch.zeros_like(self.dof_pos)
        self.prev_dof_vel = torch.zeros_like(self.dof_vel)
        self.prev_base_quat = torch.zeros_like(self.base_quat)
        self.prev_base_lin_vel = torch.zeros_like(self.base_lin_vel)
        self.prev_base_ang_vel = torch.zeros_like(self.base_ang_vel)
        
        # Initialize discriminator reward storage
        self.disc_rewards = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.disc_coef = getattr(cfg.rewards.scales, 'disc_coef', 0.5)  # Default coefficient for discriminator reward
        
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.cfg.motion.resample_motions_for_envs_interval = np.ceil(self.cfg.motion.resample_motions_for_envs_interval_s / self.dt)

    def check_termination(self):
        if self.cfg.motion.terminate_by_1time_motion:
            time = (self.episode_length_buf) * self.dt + self.motion_start_times
            self.time_out_by_1time_motion = time > self.motion_len
            self.time_out_buf = self.time_out_by_1time_motion
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf |= torch.any(torch.abs(self.projected_gravity[:, 0:1]) > 0.7, dim=1)
        self.reset_buf |= torch.any(torch.abs(self.projected_gravity[:, 1:2]) > 0.7, dim=1)
        self.reset_buf |= self.time_out_buf

    def compute_observations(self):
        # For H1AMP, we use the AMP observations as the main observations
        # Compute AMP observations using consecutive timesteps (DeepMimic approach)
        self.obs_buf = self._compute_amp_observations()
        self.amp_obs_buf = self.obs_buf.clone()
        
        # Add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        
        # Update previous states for next timestep
        self._update_previous_states()
        
    def _compute_amp_observations(self):
        """Compute 105-dimensional AMP observations for discriminator using consecutive timesteps"""
        B = self.num_envs
        
        # Current state (pose at time t)
        curr_dof_pos = self.dof_pos
        curr_dof_vel = self.dof_vel
        curr_base_quat = self.base_quat
        curr_base_lin_vel = self.base_lin_vel
        curr_base_ang_vel = self.base_ang_vel
        
        # Previous state (pose at time t-1)
        prev_dof_pos = self.prev_dof_pos
        prev_dof_vel = self.prev_dof_vel
        prev_base_quat = self.prev_base_quat
        prev_base_lin_vel = self.prev_base_lin_vel
        prev_base_ang_vel = self.prev_base_ang_vel
        
        # Build AMP observation following DeepMimic pattern:
        # [current_pose, previous_pose, current_vel, previous_vel]
        
        # Current pose features
        curr_root_h = self.root_states[:, 2:3]  # Root height
        curr_heading_rot = torch_utils.calc_heading_quat(curr_base_quat)
        curr_heading_rot_inv = torch_utils.calc_heading_quat_inv(curr_base_quat)
        
        # Transform current orientation to heading-relative
        curr_local_rot = torch_utils.quat_mul(curr_heading_rot_inv, curr_base_quat)
        curr_rot_tan_norm = torch_utils.quat_to_tan_norm(curr_local_rot).view(B, -1)
        
        # Current joint positions (relative to default)
        curr_joint_pos = curr_dof_pos - self.default_dof_pos
        
        # Previous pose features  
        # TODO: Store previous root height properly - for now using current as approximation
        prev_root_h = self.root_states[:, 2:3]  # Should ideally use previous root height
        prev_heading_rot = torch_utils.calc_heading_quat(prev_base_quat)
        prev_heading_rot_inv = torch_utils.calc_heading_quat_inv(prev_base_quat)
        
        # Transform previous orientation to heading-relative
        prev_local_rot = torch_utils.quat_mul(prev_heading_rot_inv, prev_base_quat)
        prev_rot_tan_norm = torch_utils.quat_to_tan_norm(prev_local_rot).view(B, -1)
        
        # Previous joint positions (relative to default)
        prev_joint_pos = prev_dof_pos - self.default_dof_pos
        
        # Current velocity features
        curr_local_lin_vel = torch_utils.quat_rotate_inverse(curr_heading_rot, curr_base_lin_vel)
        curr_local_ang_vel = torch_utils.quat_rotate_inverse(curr_heading_rot, curr_base_ang_vel)
        
        # Previous velocity features
        prev_local_lin_vel = torch_utils.quat_rotate_inverse(prev_heading_rot, prev_base_lin_vel)
        prev_local_ang_vel = torch_utils.quat_rotate_inverse(prev_heading_rot, prev_base_ang_vel)
        
        # Concatenate all features to build 105-dimensional AMP observation
        # Following DeepMimic structure: poses (current + previous) + velocities (current + previous)
        amp_obs = torch.cat([
            # Current pose: root height (1) + root orientation (6) + joint positions (19) = 26
            curr_root_h,                                         # 1
            curr_rot_tan_norm,                                  # 6 (2*3 for tan_norm representation)
            curr_joint_pos * self.obs_scales.dof_pos,          # 19
            
            # Previous pose: root height (1) + root orientation (6) + joint positions (19) = 26  
            prev_root_h,                                         # 1
            prev_rot_tan_norm,                                  # 6
            prev_joint_pos * self.obs_scales.dof_pos,          # 19
            
            # Current velocity: root linear vel (3) + root angular vel (3) + joint velocities (19) = 25
            curr_local_lin_vel * self.obs_scales.lin_vel,      # 3
            curr_local_ang_vel * self.obs_scales.ang_vel,      # 3
            curr_dof_vel * self.obs_scales.dof_vel,            # 19
            
            # Previous velocity: root linear vel (3) + root angular vel (3) + joint velocities (19) = 25
            prev_local_lin_vel * self.obs_scales.lin_vel,      # 3
            prev_local_ang_vel * self.obs_scales.ang_vel,      # 3
            prev_dof_vel * self.obs_scales.dof_vel,            # 19
            
            # Additional features
            self.projected_gravity,                             # 3
        ], dim=-1)
        
        return amp_obs
    
    def _update_previous_states(self):
        """Update previous states for next timestep AMP observation construction"""
        self.prev_dof_pos.copy_(self.dof_pos)
        self.prev_dof_vel.copy_(self.dof_vel)
        self.prev_base_quat.copy_(self.base_quat)
        self.prev_base_lin_vel.copy_(self.base_lin_vel)
        self.prev_base_ang_vel.copy_(self.base_ang_vel)

    def reset_idx(self, env_ids):
        self._resample_motion_times(env_ids)
        return super().reset_idx(env_ids)

    def _reset_dofs(self, env_ids):
        motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times
        offset = self.env_origins
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset=offset)
        self.dof_pos[env_ids] = motion_res['dof_pos'][env_ids]
        self.dof_vel[env_ids] = motion_res['dof_vel'][env_ids]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        if self.custom_origins:
            raise NotImplementedError("Custom origins not implemented for H1AMP")
        else:
            motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times
            offset = self.env_origins
            motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset=offset)
            self.root_states[env_ids, :3] = motion_res['root_pos'][env_ids]
            self.root_states[env_ids, 3:7] = motion_res['root_rot'][env_ids]
            self.root_states[env_ids, 7:10] = motion_res['root_vel'][env_ids]
            self.root_states[env_ids, 10:13] = motion_res['root_ang_vel'][env_ids]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        env_ids_int32 = torch.arange(self.num_envs).to(dtype=torch.int32).to(self.device)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def post_physics_step(self):
        super().post_physics_step()
        if self.cfg.motion.sync:
            self._motion_sync()
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
        if self.common_step_counter % self.cfg.motion.resample_motions_for_envs_interval == 0:
            logger.info("Resampling motions for envs")
            logger.info(f"common_step_counter: {self.common_step_counter}")
            self.resample_motion()

    def _motion_sync(self):
        num_motions = self._motion_lib.num_motions()
        motion_ids = np.arange(self.num_envs, dtype=np.int)
        motion_ids = torch.from_numpy(np.mod(motion_ids, num_motions))
        motion_times = torch.tensor([self._hack_motion_time] * self.num_envs, dtype=torch.float32, device=self.device)
        motion_res = self._get_state_from_motionlib_cache_trimesh(motion_ids, motion_times)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, smpl_params, limb_weights, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
            motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
            motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._set_env_state(env_ids=env_ids, root_pos=root_pos, root_rot=root_rot, dof_pos=dof_pos, root_vel=root_vel, root_ang_vel=root_ang_vel, dof_vel=dof_vel)
        self._reset_env_tensors(env_ids)
        motion_dur = self._motion_lib._motion_lengths[0]
        self._hack_motion_time = np.fmod(self._hack_motion_time + self.motion_dt.cpu().numpy(), motion_dur.cpu().numpy())

    def _init_buffers(self):
        super()._init_buffers()
        self.ref_motion_cache = {}
        self._load_motion()
        self.marker_coords = torch.zeros(self.num_envs, self.num_dofs + 3, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.motion_ids = torch.arange(self.num_envs).to(self.device)
        self.motion_start_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.motion_len = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self._resample_motion_times(env_ids)
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))

    def _load_motion(self):
        cfg_motion: Motion = self.cfg.motion
        motion_path = cfg_motion.motion_file
        skeleton_path = cfg_motion.skeleton_file
        self._motion_lib = MotionLibH1(
            motion_file=motion_path, device=self.device, 
            masterfoot_conifg=None, fix_height=False,
            multi_thread=False, mjcf_file=skeleton_path, 
            sim_timestep=cfg_motion.dt if cfg_motion.dt is not None else self.dt,
        )
        sk_tree = SkeletonTree.from_mjcf(skeleton_path)
        if cfg_motion.test_keys is not None:
            self.motion_data_ids = []
            for idx, keys in enumerate(self._motion_lib._motion_data_keys):
                if keys in cfg_motion.test_keys:
                    self.motion_data_ids.append(idx)
        else:
            self.motion_data_ids = np.arange(len(self._motion_lib._motion_data_keys))
        logger.info(f"Loading {len(self.motion_data_ids)} motions from {motion_path} with skeleton {skeleton_path}")
        self.skeleton_trees = [sk_tree] * self.num_envs
        if self.cfg.env.test:
            self.motion_start_idx = 0
            self._motion_lib.load_motions(
                skeleton_trees=self.skeleton_trees, gender_betas=[torch.zeros(17)] * self.num_envs, 
                limb_weights=[np.zeros(10)] * self.num_envs, 
                random_sample=False, start_idx=self.motion_data_ids[self.motion_start_idx]
            )
        else:
            self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=[torch.zeros(17)] * self.num_envs, limb_weights=[np.zeros(10)] * self.num_envs, random_sample=True)
        self.motion_dt = self._motion_lib._motion_dt
        if cfg_motion.sync:
            self._hack_motion_time = 0.0

    def resample_motion(self):
        if self.cfg.env.test:
            self._motion_lib.load_motions(
                skeleton_trees=self.skeleton_trees, gender_betas=[torch.zeros(17)] * self.num_envs, 
                limb_weights=[np.zeros(10)] * self.num_envs, 
                random_sample=False, start_idx=self.motion_data_ids[self.motion_start_idx]
            )
        else:
            self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=[torch.zeros(17)] * self.num_envs, limb_weights=[np.zeros(10)] * self.num_envs, random_sample=True)
        env_ids = torch.arange(self.num_envs).to(self.device)
        self.reset_idx(env_ids)

    def _resample_motion_times(self, env_ids):
        if len(env_ids) == 0:
            return
        self.motion_len[env_ids] = self._motion_lib.get_motion_length(self.motion_ids[env_ids])
        if self.cfg.env.test:
            self.motion_start_times[env_ids] = 0
        else:
            self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])
        offset = self.env_origins
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)

    def _get_state_from_motionlib_cache_trimesh(self, motion_ids, motion_times, offset=None):
        if offset is None  or not "motion_ids" in self.ref_motion_cache or self.ref_motion_cache['offset'] is None or len(self.ref_motion_cache['motion_ids']) != len(motion_ids) or len(self.ref_motion_cache['offset']) != len(offset) \
            or  (self.ref_motion_cache['motion_ids'] - motion_ids).abs().sum() + (self.ref_motion_cache['motion_times'] - motion_times).abs().sum() + (self.ref_motion_cache['offset'] - offset).abs().sum() > 0 :
            self.ref_motion_cache['motion_ids'] = motion_ids.clone()
            self.ref_motion_cache['motion_times'] = motion_times.clone()
            self.ref_motion_cache['offset'] = offset.clone() if not offset is None else None
        else:
            return self.ref_motion_cache
        motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times, offset=offset)
        self.ref_motion_cache.update(motion_res)
        return self.ref_motion_cache

    def handle_viewer_action_event(self, evt):
        super().handle_viewer_action_event(evt)
        if evt.action == "prev_motion" and evt.value > 0:
            self.motion_start_idx = (self.motion_start_idx - 1) % len(self.motion_data_ids)
            self.resample_motion()
        elif evt.action == "next_motion" and evt.value > 0:
            self.motion_start_idx = (self.motion_start_idx + 1) % len(self.motion_data_ids)
            self.resample_motion()

    def _setup_viewer(self):
        super()._setup_viewer()
        if not self.headless:
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "prev_motion")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "next_motion")

    def _draw_debug_vis(self):
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        for env_id in range(self.num_envs):
            for pos_id, pos_joint in enumerate(self.marker_coords[env_id]):
                color_inner = (0.3, 0.3, 0.3) if not self.cfg.motion.visualize_config.customize_color \
                                                else self.cfg.motion.visualize_config.marker_joint_colors[pos_id % len(self.cfg.motion.visualize_config.marker_joint_colors)]
                color_inner = tuple(color_inner)
                sphere_geom_marker = gymutil.WireframeSphereGeometry(0.05, 20, 20, None, color=color_inner)
                sphere_pose = gymapi.Transform(gymapi.Vec3(pos_joint[0], pos_joint[1], pos_joint[2]), r=None)
                gymutil.draw_lines(sphere_geom_marker, self.gym, self.viewer, self.envs[env_id], sphere_pose)

    #------------ reward functions----------------

    def _reward_amp(self):
        # This function should return the discriminator reward for each environment
        # It is expected that self.amp_rewards is set externally (by the runner/algorithm)
        # If not set, return zeros
        if hasattr(self, 'amp_rewards') and self.amp_rewards is not None:
            return self.amp_rewards
        else:
            return torch.zeros_like(self.rew_buf)

    def set_amp_rewards(self, amp_rewards):
        """
        Set the AMP (discriminator) rewards for the current environment step.
        This should be called by the training loop or algorithm after discriminator evaluation.
        Args:
            amp_rewards (torch.Tensor): Tensor of shape (num_envs,) with AMP rewards for each environment.
        """
        self.amp_rewards = amp_rewards
        
    def compute_reward(self):
        """Override compute_reward to implement AMP reward blending"""
        # Call parent compute_reward first to get task-specific rewards
        super().compute_reward()
        
        # Store task rewards before modification
        task_rewards = self.rew_buf.clone()
        
        # Get AMP (discriminator) rewards if available
        if hasattr(self, 'amp_rewards') and self.amp_rewards is not None:
            # Add safety checks for AMP rewards
            if torch.any(torch.isnan(self.amp_rewards)) or torch.any(torch.isinf(self.amp_rewards)):
                print("Warning: NaN or Inf detected in AMP rewards, using task rewards only")
                return
            
            # Scale discriminator rewards
            amp_reward_scale = getattr(self.cfg.rewards.scales, 'amp', 0.5)
            scaled_disc_rewards = amp_reward_scale * self.amp_rewards
            
            # Clamp scaled discriminator rewards to prevent extreme values
            scaled_disc_rewards = torch.clamp(scaled_disc_rewards, min=-10.0, max=10.0)
            
            # Get task reward lerp parameter
            task_reward_lerp = getattr(self.cfg.rewards, 'task_reward_lerp', 0.5)
            
            # Linear interpolation between discriminator and task rewards
            # r = (1.0 - task_reward_lerp) * disc_r + task_reward_lerp * task_r
            blended_rewards = (1.0 - task_reward_lerp) * scaled_disc_rewards + task_reward_lerp * task_rewards
            
            # Store discriminator rewards for external access
            self.disc_rewards = scaled_disc_rewards
            
            # Update the reward buffer with blended rewards
            self.rew_buf = blended_rewards
            
            # Final safety clamp on total rewards
            self.rew_buf = torch.clamp(self.rew_buf, min=-100.0, max=100.0)
        else:
            # If no AMP rewards available, use task rewards only
            self.disc_rewards = torch.zeros_like(task_rewards)
            pass

    def get_discriminator_rewards(self):
        """Return the current discriminator rewards for external use"""
        return self.disc_rewards.clone()

    def set_discriminator_rewards(self, disc_rewards):
        """Set discriminator rewards from external source"""
        if disc_rewards.shape != self.disc_rewards.shape:
            raise ValueError(f"Expected discriminator rewards shape {self.disc_rewards.shape}, got {disc_rewards.shape}")
        self.disc_rewards = disc_rewards.to(self.device)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the AMP observations.
            [NOTE]: Must be adapted when changing the AMP observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # Create noise vector with AMP observation dimensions (105)
        noise_vec = torch.zeros(self.amp_obs_dim, dtype=torch.float, device=self.device, requires_grad=False)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        
        # AMP observation structure (105 dims):
        # Current pose: height(1) + rotation(6) + joints(19) = 26
        # Previous pose: height(1) + rotation(6) + joints(19) = 26  
        # Current velocity: lin_vel(3) + ang_vel(3) + joint_vel(19) = 25
        # Previous velocity: lin_vel(3) + ang_vel(3) + joint_vel(19) = 25
        # Gravity: (3)
        # Total: 26 + 26 + 25 + 25 + 3 = 105
        
        # Current pose
        noise_vec[0:1] = 0.0  # root height - no noise
        noise_vec[1:7] = 0.0  # root rotation (6D) - no noise  
        noise_vec[7:26] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos  # joint positions (19)
        
        # Previous pose  
        noise_vec[26:27] = 0.0  # prev root height - no noise
        noise_vec[27:33] = 0.0  # prev root rotation (6D) - no noise
        noise_vec[33:52] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos  # prev joint positions (19)
        
        # Current velocity
        noise_vec[52:55] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel  # root linear velocity (3)
        noise_vec[55:58] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel  # root angular velocity (3)
        noise_vec[58:77] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel  # joint velocities (19)
        
        # Previous velocity
        noise_vec[77:80] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel  # prev root linear velocity (3)
        noise_vec[80:83] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel  # prev root angular velocity (3)
        noise_vec[83:102] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel  # prev joint velocities (19)
        
        # Gravity
        noise_vec[102:105] = noise_scales.gravity * noise_level  # gravity (3)
        
        return noise_vec
