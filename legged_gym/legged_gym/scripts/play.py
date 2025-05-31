import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import tyro
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

from legged_gym.scripts.train import Args
from legged_gym.envs.h1.h1_amp_config import H1AMPCfg, H1AMPCfgPPO
from rsl_rl.modules.discriminator import Discriminator
from rsl_rl.modules.buffers import ReplayBuffer, DemoBuffer


def play(args: Args):
    env_cfg, train_cfg = args.env_cfg, args.train_cfg
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.ranges.lin_vel_x = [0.75, 0.75]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0.0, 0.0]

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(args=args, env_cfg=env_cfg)
    obs_dim = env.num_obs  # Policy observations (119-dim for H1AMP)
    obs = env.get_observations()
    
    # Check if we're using AMP algorithm and create necessary components
    discriminator = None
    demo_buffer = None
    replay_buffer = None
    
    if train_cfg.runner.algorithm_class_name == 'AMP':
        # For AMP environments, discriminator uses AMP observation dimensions, not policy observations
        if hasattr(env, 'amp_obs_dim'):
            amp_obs_dim = env.amp_obs_dim  # AMP observations (105-dim for H1AMP)
        else:
            amp_obs_dim = obs_dim  # Fallback for non-AMP environments
            
        # Create discriminator for AMP inference (same as training)
        discriminator = Discriminator(input_dim=amp_obs_dim)  # Use AMP observation dimensions
        
        # Create empty buffers for inference (not used during play but required for AMP initialization)
        demo_buffer = DemoBuffer(torch.zeros(1, amp_obs_dim))  # Dummy data with correct dimensions
        replay_buffer = ReplayBuffer(amp_obs_dim, capacity=1000)  # Small capacity for inference
    
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, args=args, train_cfg=train_cfg, 
        discriminator=discriminator, demo_buffer=demo_buffer, replay_buffer=replay_buffer
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

if __name__ == '__main__':
    args = tyro.cli(Args)
    play(args)
