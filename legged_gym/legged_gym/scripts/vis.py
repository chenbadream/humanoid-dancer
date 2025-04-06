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


def play(args: Args):
    args.headless = False
    env_cfg, train_cfg = args.env_cfg, args.train_cfg
    # override some parameters for testing
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 1000
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(args=args, env_cfg=env_cfg)

    for i in range(int(env.max_episode_length)):
        actions = torch.zeros((env.num_envs, env_cfg.env.num_actions), device=env.device)
        obs, _, rews, dones, infos = env.step(actions)

if __name__ == '__main__':
    args = tyro.cli(Args)
    play(args)
