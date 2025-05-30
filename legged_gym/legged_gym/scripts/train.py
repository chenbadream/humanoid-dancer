from dataclasses import dataclass, field
from typing import Optional, Union
import os

import tyro

import isaacgym
from isaacgym import gymapi

from legged_gym.envs import *
from legged_gym.utils import task_registry
from legged_gym.envs.h1.h1_amp_config import H1AMPCfg, H1AMPCfgPPO
from rsl_rl.modules.discriminator import Discriminator
from rsl_rl.modules.buffers import ReplayBuffer, DemoBuffer
import pickle
import numpy as np
import torch

@dataclass  
class Args:
    env_cfg: Union[h1_config.H1Cfg, h1_mimic_config.H1MimicCfg, H1AMPCfg] = field(default_factory=h1_config.H1Cfg)
    train_cfg: Union[h1_config.H1PPOCfg, h1_mimic_config.H1MimicPPOCfg, H1AMPCfgPPO] = field(default_factory=h1_config.H1PPOCfg)
    # Resume training from a checkpoint
    resume: bool = False
    # Name of the experiment to run or load. Overrides config file if provided.
    experiment_name: Optional[str] = None
    # Name of the run. Overrides config file if provided.
    run_name: Optional[str] = None
    # Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.
    load_run: Optional[Union[str, int]] = None
    # Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.
    checkpoint: Optional[int] = None
    
    # Force display off at all times
    headless: bool = True
    # Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)
    rl_device: str = "cuda:0"
    # Number of environments to create. Overrides config file if provided.
    num_envs: Optional[int] = None
    # Random seed. Overrides config file if provided.
    seed: Optional[int] = None
    # Maximum number of training iterations. Overrides config file if provided.
    max_iterations: Optional[int] = None
    
    # Physics Device in PyTorch-like syntax
    sim_device: str = "cuda:0"
    # Graphics Device ID
    sim_device_id: int = 0
    use_gpu: bool = True
    use_gpu_pipeline: bool = True
    
    # Don't need to change this
    subscenes = 0
    slices = 0
    num_threads = 0
    physics_engine = None
    
    def __post_init__(self):
        self.physics_engine = gymapi.SIM_PHYSX
        
        if self.num_envs is not None:
            self.env_cfg.env.num_envs = self.num_envs
        if self.seed is not None:
            self.train_cfg.seed = self.seed
        self.env_cfg.seed = self.train_cfg.seed
        if self.max_iterations is not None:
            self.train_cfg.runner.max_iterations = self.max_iterations
        if self.resume:
            self.train_cfg.runner.resume = self.resume
        if self.experiment_name is not None:
            self.train_cfg.runner.experiment_name = self.experiment_name
        if self.run_name is not None:
            self.train_cfg.runner.run_name = self.run_name
        if self.load_run is not None:
            self.train_cfg.runner.load_run = self.load_run
        if self.checkpoint is not None:
            self.train_cfg.runner.checkpoint = self.checkpoint
        
def generate_amp_demo_obs(args, env_cfg, device):
    """Generate AMP demo observations for discriminator training.
    
    Following DeepMimic approach: Generate AMP observations from consecutive motion frames.
    Each AMP observation contains pose and velocity data from two consecutive timesteps.
    """
    from legged_gym.motions.motion_lib_h1 import MotionLibH1
    from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
    import torch
    import numpy as np
    import math
    
    motion_file = getattr(env_cfg.motion, 'motion_file', None)
    skeleton_file = getattr(env_cfg.motion, 'skeleton_file', None)
    if motion_file is None or skeleton_file is None:
        raise RuntimeError('Motion file or skeleton file not specified in config.')
    
    # Instantiate motion lib
    motion_lib = MotionLibH1(
        motion_file=motion_file,
        device=device,
        masterfoot_conifg=None,
        fix_height=False,
        multi_thread=False,
        mjcf_file=skeleton_file,
        sim_timestep=env_cfg.motion.dt if env_cfg.motion.dt is not None else env_cfg.sim.dt,
    )
    sk_tree = SkeletonTree.from_mjcf(skeleton_file)
    skeleton_trees = [sk_tree]
    gender_betas = [torch.zeros(17)]
    limb_weights = [np.zeros(10)]
    motion_lib.load_motions(skeleton_trees=skeleton_trees, gender_betas=gender_betas, limb_weights=limb_weights, random_sample=False)
    
    # Get obs scales and default dof pos from env config
    obs_scales = getattr(env_cfg, 'obs_scales', None)
    if obs_scales is None:
        obs_scales = type('obj', (), {'ang_vel': 1.0, 'dof_pos': 1.0, 'dof_vel': 1.0, 'lin_vel': 1.0})()
    default_dof_pos = getattr(env_cfg.init_state, 'default_dof_pos', None)
    if default_dof_pos is None:
        # fallback: zeros
        default_dof_pos = np.zeros(motion_lib.dof_pos.shape[-1])
    default_dof_pos = torch.tensor(default_dof_pos, dtype=torch.float32, device=device)
    
    def build_amp_obs_from_motion_states(state_t, state_t1, obs_scales, default_dof_pos):
        """Build AMP observation from two consecutive motion states matching H1AMP._compute_amp_observations format."""
        # Import torch_utils to use the same functions as h1_amp.py
        from legged_gym.utils import torch_utils
        
        # Current state (t)
        curr_root_pos = state_t['root_pos']  # [1, 3]
        curr_base_quat = state_t['rb_rot'][:, 0]  # [1, 4] - root rotation 
        curr_base_lin_vel = state_t['root_vel']  # [1, 3]
        curr_base_ang_vel = state_t['root_ang_vel']  # [1, 3]
        curr_dof_pos = state_t['dof_pos']  # [1, 19]
        curr_dof_vel = state_t['dof_vel']  # [1, 19]
        
        # Previous state (t-1)
        prev_root_pos = state_t1['root_pos']  # [1, 3]
        prev_base_quat = state_t1['rb_rot'][:, 0]  # [1, 4] - root rotation
        prev_base_lin_vel = state_t1['root_vel']  # [1, 3]
        prev_base_ang_vel = state_t1['root_ang_vel']  # [1, 3]
        prev_dof_pos = state_t1['dof_pos']  # [1, 19]
        prev_dof_vel = state_t1['dof_vel']  # [1, 19]
        
        # Current pose features (using heading-relative coordinates like h1_amp.py)
        curr_root_h = curr_root_pos[:, 2:3]  # Root height
        curr_heading_rot = torch_utils.calc_heading_quat(curr_base_quat)
        curr_heading_rot_inv = torch_utils.calc_heading_quat_inv(curr_base_quat)
        
        # Transform current orientation to heading-relative
        curr_local_rot = torch_utils.quat_mul(curr_heading_rot_inv, curr_base_quat)
        curr_rot_tan_norm = torch_utils.quat_to_tan_norm(curr_local_rot).view(1, -1)
        
        # Current joint positions (relative to default)
        curr_joint_pos = curr_dof_pos - default_dof_pos.unsqueeze(0)
        
        # Previous pose features
        prev_root_h = prev_root_pos[:, 2:3]  # Root height
        prev_heading_rot = torch_utils.calc_heading_quat(prev_base_quat)
        prev_heading_rot_inv = torch_utils.calc_heading_quat_inv(prev_base_quat)
        
        # Transform previous orientation to heading-relative
        prev_local_rot = torch_utils.quat_mul(prev_heading_rot_inv, prev_base_quat)
        prev_rot_tan_norm = torch_utils.quat_to_tan_norm(prev_local_rot).view(1, -1)
        
        # Previous joint positions (relative to default)
        prev_joint_pos = prev_dof_pos - default_dof_pos.unsqueeze(0)
        
        # Current velocity features (transform to local coordinates)
        curr_local_lin_vel = torch_utils.quat_rotate_inverse(curr_heading_rot, curr_base_lin_vel)
        curr_local_ang_vel = torch_utils.quat_rotate_inverse(curr_heading_rot, curr_base_ang_vel)
        
        # Previous velocity features (transform to local coordinates)
        prev_local_lin_vel = torch_utils.quat_rotate_inverse(prev_heading_rot, prev_base_lin_vel)
        prev_local_ang_vel = torch_utils.quat_rotate_inverse(prev_heading_rot, prev_base_ang_vel)
        
        # Gravity vector (pointing down)
        gravity = torch.tensor([[0.0, 0.0, -1.0]], device=curr_root_pos.device)
        
        # Concatenate all features to build 105-dimensional AMP observation
        # Following exact same structure as H1AMP._compute_amp_observations
        amp_obs = torch.cat([
            # Current pose: root height (1) + root orientation (6) + joint positions (19) = 26
            curr_root_h,                                         # 1
            curr_rot_tan_norm,                                  # 6 (2*3 for tan_norm representation)
            curr_joint_pos * obs_scales.dof_pos,               # 19
            
            # Previous pose: root height (1) + root orientation (6) + joint positions (19) = 26  
            prev_root_h,                                         # 1
            prev_rot_tan_norm,                                  # 6
            prev_joint_pos * obs_scales.dof_pos,               # 19
            
            # Current velocity: root linear vel (3) + root angular vel (3) + joint velocities (19) = 25
            curr_local_lin_vel * obs_scales.lin_vel,           # 3
            curr_local_ang_vel * obs_scales.ang_vel,           # 3
            curr_dof_vel * obs_scales.dof_vel,                 # 19
            
            # Previous velocity: root linear vel (3) + root angular vel (3) + joint velocities (19) = 25
            prev_local_lin_vel * obs_scales.lin_vel,           # 3
            prev_local_ang_vel * obs_scales.ang_vel,           # 3
            prev_dof_vel * obs_scales.dof_vel,                 # 19
            
            # Additional features
            gravity,                                            # 3
        ], dim=-1)
        
        return amp_obs
    
    # Generate AMP observations from motion data  
    demo_obs = []
    num_motions = motion_lib.num_motions()
    for motion_id in range(num_motions):
        motion_len = int(motion_lib.get_motion_num_steps(torch.tensor([motion_id], device=device))[0].item())
        # Start from frame 1 since we need consecutive pairs
        for frame in range(1, motion_len):  
            motion_ids = torch.tensor([motion_id], device=device)
            dt = float(motion_lib._motion_dt[motion_id].cpu().numpy())
            
            # Get current state (t)
            motion_times_t = torch.tensor([frame * dt], device=device)
            state_t = motion_lib.get_motion_state(motion_ids, motion_times_t)
            
            # Get previous state (t-1)
            motion_times_t1 = torch.tensor([(frame - 1) * dt], device=device)
            state_t1 = motion_lib.get_motion_state(motion_ids, motion_times_t1)
            
            # Build AMP observation from consecutive states
            amp_obs = build_amp_obs_from_motion_states(state_t, state_t1, obs_scales, default_dof_pos)
            demo_obs.append(amp_obs.squeeze(0).cpu().numpy())
    
    demo_obs = np.stack(demo_obs, axis=0)
    return torch.tensor(demo_obs, dtype=torch.float32, device=device)

def train(args: Args):
    env, env_cfg = task_registry.make_env(args=args, env_cfg=args.env_cfg)
    obs_dim = env.num_obs
    discriminator = Discriminator(input_dim=obs_dim)  # AMP observations are 105-dim (updated from 119)

    # Generate AMP demonstration data using the motion library  
    # These are now constructed as 105-dim AMP observations
    demo_data = generate_amp_demo_obs(args, env_cfg, args.rl_device)
    demo_buffer = DemoBuffer(demo_data)
    
    # Replay buffer will store AMP observations (105-dim) generated by the environment
    replay_buffer = ReplayBuffer(obs_dim, capacity=1000000)  # AMP obs size updated
    
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, args=args, train_cfg=args.train_cfg, discriminator=discriminator, demo_buffer=demo_buffer, replay_buffer=replay_buffer)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=False)
    
if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)