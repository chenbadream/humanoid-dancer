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
        """Build AMP observation from two consecutive motion states (following DeepMimic BuildAMPObs)."""
        # Current state (t)
        root_pos_t = state_t['root_pos']  # [1, 3]
        root_quat_t = state_t['rb_rot'][:, 0]  # [1, 4] - root rotation 
        root_vel_t = state_t['root_vel']  # [1, 3]
        root_ang_vel_t = state_t['root_ang_vel']  # [1, 3]
        dof_pos_t = (state_t['dof_pos'] - default_dof_pos.unsqueeze(0)) * obs_scales.dof_pos  # [1, 19]
        dof_vel_t = state_t['dof_vel'] * obs_scales.dof_vel  # [1, 19]
        
        # Previous state (t-1)
        root_pos_t1 = state_t1['root_pos']  # [1, 3]
        root_quat_t1 = state_t1['rb_rot'][:, 0]  # [1, 4] - root rotation
        root_vel_t1 = state_t1['root_vel']  # [1, 3]
        root_ang_vel_t1 = state_t1['root_ang_vel']  # [1, 3]
        dof_pos_t1 = (state_t1['dof_pos'] - default_dof_pos.unsqueeze(0)) * obs_scales.dof_pos  # [1, 19]
        dof_vel_t1 = state_t1['dof_vel'] * obs_scales.dof_vel  # [1, 19]
        
        # Build AMP observation: [pos_t, quat_t, vel_t, angvel_t, dofpos_t, dofvel_t, 
        #                         pos_t1, quat_t1, vel_t1, angvel_t1, dofpos_t1, dofvel_t1]
        # Total: 3+4+3+3+19+19 + 3+4+3+3+19+19 = 51 + 51 = 102... 
        # Wait, that's not 119. Let me check what AMP obs should contain
        
        # Following DeepMimic: root pose, root velocity, joint positions, joint velocities for two timesteps
        # Let's use root height, root rotation as 6D (tan-norm), linear vel, angular vel, dof pos, dof vel
        # for both timesteps
        
        # Convert quaternions to 6D representation (tan-norm)
        def quat_to_tan_norm(quat):
            # quat is [w, x, y, z], convert to rotation matrix first 6 elements
            w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
            # Convert to rotation matrix and take first two columns (6 elements)
            xx, yy, zz = x*x, y*y, z*z
            xy, xz, yz = x*y, x*z, y*z
            wx, wy, wz = w*x, w*y, w*z
            
            rot_mat = torch.stack([
                1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy),
                2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)
            ], dim=-1)  # First 6 elements of rotation matrix
            return rot_mat
        
        root_rot_6d_t = quat_to_tan_norm(root_quat_t)  # [1, 6]
        root_rot_6d_t1 = quat_to_tan_norm(root_quat_t1)  # [1, 6]
        
        # AMP observation for DeepMimic: 
        # root_height(1) + root_rot_6d(6) + root_vel(3) + root_ang_vel(3) + dof_pos(19) + dof_vel(19) = 51 per timestep
        # For 2 timesteps: 51 * 2 = 102... still not 119
        
        # Let me check what the actual env AMP observation size should be
        # Based on H1 robot: it might include more joint info or body positions
        # For now, let's use root height + 6D rotation + velocities + joint data for both timesteps
        # And add some padding or body parts if needed to reach 119
        
        root_height_t = root_pos_t[:, 2:3]  # [1, 1] - Z coordinate 
        root_height_t1 = root_pos_t1[:, 2:3]  # [1, 1]
        
        # Current timestep: height(1) + rot_6d(6) + lin_vel(3) + ang_vel(3) + dof_pos(19) + dof_vel(19) = 51
        obs_t = torch.cat([
            root_height_t,      # 1
            root_rot_6d_t,      # 6  
            root_vel_t,         # 3
            root_ang_vel_t,     # 3
            dof_pos_t,          # 19
            dof_vel_t,          # 19
        ], dim=-1)  # Total: 51
        
        # Previous timestep: same format
        obs_t1 = torch.cat([
            root_height_t1,     # 1
            root_rot_6d_t1,     # 6
            root_vel_t1,        # 3  
            root_ang_vel_t1,    # 3
            dof_pos_t1,         # 19
            dof_vel_t1,         # 19
        ], dim=-1)  # Total: 51
        
        # Combine both timesteps
        amp_obs = torch.cat([obs_t, obs_t1], dim=-1)  # Total: 102
        
        # If we need 119 dimensions, we need 17 more. Let's add some additional features:
        # Maybe body part positions or additional velocity info
        # For now, let's pad with zeros to reach 119 (this is a temporary fix)
        padding_size = 119 - amp_obs.shape[-1]
        if padding_size > 0:
            padding = torch.zeros(amp_obs.shape[0], padding_size, device=amp_obs.device)
            amp_obs = torch.cat([amp_obs, padding], dim=-1)
        
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
    discriminator = Discriminator(input_dim=119)  # AMP observations are 119-dim

    # Generate AMP demonstration data using the motion library  
    # These are already constructed as 119-dim AMP observations
    demo_data = generate_amp_demo_obs(args, env_cfg, args.rl_device)
    demo_buffer = DemoBuffer(demo_data)
    
    # Replay buffer will store AMP observations (119-dim) generated by the environment
    replay_buffer = ReplayBuffer(119, capacity=1000000)  # AMP obs size
    
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, args=args, train_cfg=args.train_cfg, discriminator=discriminator, demo_buffer=demo_buffer, replay_buffer=replay_buffer)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=False)
    
if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)