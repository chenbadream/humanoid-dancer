from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union

from legged_gym.envs.h1 import h1_config
from legged_gym.envs.base import legged_robot_config

@dataclass
class Motion:
    motion_file: str = 'resources/motions/h1/stable_punch.pkl'
    skeleton_file: str = 'resources/robots/h1/xml/h1.xml'
    resample_motions_for_envs_interval_s: float = 1000.
    terminate_by_1time_motion: bool = True
    dt: Optional[float] = None
    sync: bool = False
    test_keys: Optional[List[str]] = None

@dataclass
class Env(h1_config.Env):
    num_privileged_obs: Optional[int] = None
    num_observations: int = 119

@dataclass
class Rewards(h1_config.Rewards):
    only_positive_rewards: bool = False
    # Add AMP-specific reward parameters if needed

@dataclass
class H1AMPCfg:
    env: Env = field(default_factory=Env)
    terrain: legged_robot_config.Terrain = field(default_factory=legged_robot_config.Terrain)
    commands: legged_robot_config.Commands = field(default_factory=legged_robot_config.Commands)
    init_state: h1_config.InitState = field(default_factory=h1_config.InitState)
    control: h1_config.Control = field(default_factory=h1_config.Control)
    asset: h1_config.Asset = field(default_factory=h1_config.Asset)
    domain_rand: h1_config.DomainRand = field(default_factory=h1_config.DomainRand)
    rewards: Rewards = field(default_factory=Rewards)
    normalization: legged_robot_config.Normalization = field(default_factory=legged_robot_config.Normalization)
    noise: legged_robot_config.Noise = field(default_factory=legged_robot_config.Noise)
    viewer: legged_robot_config.Viewer = field(default_factory=legged_robot_config.Viewer)
    sim: legged_robot_config.Sim = field(default_factory=legged_robot_config.Sim)
    motion: Motion = field(default_factory=Motion)
    seed: int = field(init=False)
    name = 'h1_amp'

@dataclass  
class Policy(h1_config.Policy):
    init_noise_std: float = 1.
    # Add AMP-specific policy parameters if needed

@dataclass
class Runner(h1_config.Runner):
    experiment_name: str = 'h1_amp'
    algorithm_class_name: str = 'AMP'  # Use AMP algorithm instead of PPO

@dataclass
class H1AMPCfgPPO:
    seed: int = 1
    runner_class_name: str = 'OnPolicyRunner'
    policy: Policy = field(default_factory=Policy)
    algorithm: h1_config.Algorithm = field(default_factory=h1_config.Algorithm)
    runner: Runner = field(default_factory=Runner)
