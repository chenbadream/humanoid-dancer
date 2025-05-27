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
    commands_scale = getattr(env_cfg, 'commands_scale', 1.0)
    default_dof_pos = getattr(env_cfg.init_state, 'default_dof_pos', None)
    if default_dof_pos is None:
        # fallback: zeros
        default_dof_pos = np.zeros(motion_lib.dof_pos.shape[-1])
    default_dof_pos = torch.tensor(default_dof_pos, dtype=torch.float32, device=device)
    obs_dim = 119
    
    # For each motion, for each frame, generate AMP obs
    demo_obs = []
    num_motions = motion_lib.num_motions()
    for motion_id in range(num_motions):
        motion_len = int(motion_lib.get_motion_num_steps(torch.tensor([motion_id], device=device))[0].item())
        for frame in range(motion_len):
            motion_ids = torch.tensor([motion_id], device=device)
            motion_times = torch.tensor([frame * float(motion_lib._motion_dt[motion_id].cpu().numpy())], device=device)
            motion_state = motion_lib.get_motion_state(motion_ids, motion_times)
            # Use reference state as current state for demo
            base_ang_vel = motion_state['root_ang_vel']
            projected_gravity = torch.tensor([[0, 0, -1]], dtype=torch.float32, device=device)  # assuming upright
            commands = torch.zeros((1, 3), dtype=torch.float32, device=device)
            dof_pos = motion_state['dof_pos']
            dof_vel = motion_state['dof_vel']
            actions = torch.zeros_like(dof_pos)
            phase = frame / max(1, motion_len-1)
            sin_phase = torch.tensor([[math.sin(2 * math.pi * phase)]], dtype=torch.float32, device=device)
            cos_phase = torch.tensor([[math.cos(2 * math.pi * phase)]], dtype=torch.float32, device=device)
            obs = torch.cat([
                base_ang_vel * obs_scales.ang_vel,
                projected_gravity,
                commands * commands_scale,
                (dof_pos - default_dof_pos) * obs_scales.dof_pos,
                dof_vel * obs_scales.dof_vel,
                actions,
                sin_phase,
                cos_phase
            ], dim=-1)
            demo_obs.append(obs.squeeze(0).cpu().numpy())
    demo_obs = np.stack(demo_obs, axis=0)
    return torch.tensor(demo_obs, dtype=torch.float32, device=device)

def train(args: Args):
    env, env_cfg = task_registry.make_env(args=args, env_cfg=args.env_cfg)
    obs_dim = env.num_obs
    discriminator = Discriminator(input_dim=121)

    # Generate demonstration data using the motion library and AMP observation logic
    demo_data = generate_amp_demo_obs(args, env_cfg, args.rl_device)
    demo_buffer = DemoBuffer(demo_data)
    replay_buffer = ReplayBuffer(68, capacity=1000000)  # 68-dim raw agent obs
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, args=args, train_cfg=args.train_cfg, discriminator=discriminator, demo_buffer=demo_buffer, replay_buffer=replay_buffer)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=False)
    
if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)