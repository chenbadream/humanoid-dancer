import numpy as np
import torch
from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import to_torch
from loguru import logger

from .h1_mimic import H1Mimic
from legged_gym.utils import torch_utils
from legged_gym.motions.motion_lib_h1 import MotionLibH1

class H1PHC(H1Mimic):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # PHC specific parameters
        self._task_reward_w = cfg.get('task_reward_w', 0.5)
        self._disc_reward_w = cfg.get('disc_reward_w', 0.5)
        self._disc_coef = cfg.get('disc_coef', 0.1)
        self._disc_logit_reg = cfg.get('disc_logit_reg', 0.01)
        self._disc_grad_penalty = cfg.get('disc_grad_penalty', 0.1)
        self._disc_weight_decay = cfg.get('disc_weight_decay', 0.0)
        self._disc_reward_scale = cfg.get('disc_reward_scale', 1.0)
        
        # Initialize AMP buffers
        self._init_amp_buffers()
        
    def _init_amp_buffers(self):
        """Initialize buffers for AMP training"""
        self._amp_obs_demo_buffer = None  # Will be initialized by the agent
        self._amp_replay_buffer = None    # Will be initialized by the agent
        self._amp_minibatch_size = self.cfg.get('amp_minibatch_size', 32)
        
    def compute_observations(self):
        """Override to include AMP observations"""
        super().compute_observations()
        
        # Add AMP observations
        amp_obs = self._get_amp_observations()
        self.amp_obs_buf = amp_obs
        
    def _get_amp_observations(self):
        """Get observations for AMP training"""
        # Get current motion state
        motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times
        motion_res = self._get_state_from_motionlib_cache_trimesh(
            self.motion_ids, motion_times, offset=self.env_origins)
            
        # Extract relevant features for AMP
        amp_obs = torch.cat([
            self.dof_pos,
            self.dof_vel,
            self.base_lin_vel,
            self.base_ang_vel,
            motion_res['dof_pos'],
            motion_res['dof_vel'],
            motion_res['root_vel'],
            motion_res['root_ang_vel']
        ], dim=-1)
        
        return amp_obs
        
    def compute_reward(self):
        """Override to include AMP rewards"""
        # Get base rewards
        super().compute_reward()
        
        # Add AMP rewards if available
        if hasattr(self, 'amp_rewards'):
            amp_reward = self.amp_rewards['disc_rewards']
            self.rew_buf = self._task_reward_w * self.rew_buf + \
                          self._disc_reward_w * amp_reward
                          
    def get_amp_observations(self):
        """Get AMP observations for the agent"""
        return self.amp_obs_buf
        
    def get_amp_demo_observations(self):
        """Get demo observations for AMP training"""
        if self._amp_obs_demo_buffer is None:
            return None
        return self._amp_obs_demo_buffer.sample(self._amp_minibatch_size)['amp_obs']
        
    def get_amp_replay_observations(self):
        """Get replay observations for AMP training"""
        if self._amp_replay_buffer is None:
            return None
        return self._amp_replay_buffer.sample(self._amp_minibatch_size)['amp_obs']
        
    def store_amp_observations(self, amp_obs):
        """Store observations in replay buffer"""
        if self._amp_replay_buffer is not None:
            self._amp_replay_buffer.store({'amp_obs': amp_obs})
            
    def set_amp_buffers(self, demo_buffer, replay_buffer):
        """Set AMP buffers from the agent"""
        self._amp_obs_demo_buffer = demo_buffer
        self._amp_replay_buffer = replay_buffer 