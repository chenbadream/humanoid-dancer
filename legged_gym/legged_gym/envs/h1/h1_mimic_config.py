from dataclasses import dataclass, field

from legged_gym.envs.h1 import h1_config
from legged_gym.envs.base import legged_robot_config

@dataclass
class H1MimicCfg:
    env: h1_config.Env = field(default_factory=h1_config.Env)
    terrain: legged_robot_config.Terrain = field(default_factory=legged_robot_config.Terrain)
    commands: legged_robot_config.Commands = field(default_factory=legged_robot_config.Commands)
    init_state: h1_config.InitState = field(default_factory=h1_config.InitState)
    control: h1_config.Control = field(default_factory=h1_config.Control)
    asset: h1_config.Asset = field(default_factory=h1_config.Asset)
    domain_rand: h1_config.DomainRand = field(default_factory=h1_config.DomainRand)
    rewards: h1_config.Rewards = field(default_factory=h1_config.Rewards)
    normalization: legged_robot_config.Normalization = field(default_factory=legged_robot_config.Normalization)
    noise: legged_robot_config.Noise = field(default_factory=legged_robot_config.Noise)
    viewer: legged_robot_config.Viewer = field(default_factory=legged_robot_config.Viewer)
    sim: legged_robot_config.Sim = field(default_factory=legged_robot_config.Sim)
    
    seed: int = field(init=False)
    name = 'h1_mimic'
    
@dataclass
class Runner( h1_config.Runner ):
    experiment_name: str = 'h1_mimic'

@dataclass
class H1MimicPPOCfg:
    seed: int = 1
    runner_class_name: str = 'OnPolicyRunner'
    policy: h1_config.Policy = field(default_factory=h1_config.Policy)
    algorithm: h1_config.Algorithm = field(default_factory=h1_config.Algorithm)
    runner: Runner = field(default_factory=Runner)