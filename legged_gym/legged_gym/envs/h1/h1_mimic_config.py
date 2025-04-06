from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union

from legged_gym.envs.h1 import h1_config
from legged_gym.envs.base import legged_robot_config

@dataclass
class VisaulizeConfig:
    customize_color: bool = True
    marker_joint_colors: List[List[float]] = field(default_factory=lambda: [
        [0.157, 0.231, 0.361], # pelvis
        [0.157, 0.231, 0.361], # left_hip_yaw_joint
        [0.157, 0.231, 0.361], # left_hip_roll_joint
        [0.157, 0.231, 0.361], # left_hip_pitch_joint
        [0.157, 0.231, 0.361], # left_knee_joint
        [0.157, 0.231, 0.361], # left_ankle_joint
        [0.157, 0.231, 0.361], # right_hip_yaw_joint
        [0.157, 0.231, 0.361], # right_hip_roll_joint
        [0.157, 0.231, 0.361], # right_hip_pitch_joint
        [0.157, 0.231, 0.361], # right_knee_joint
        [0.157, 0.231, 0.361], # right_ankle_joint
        [0.765, 0.298, 0.498], # torso_joint
        [0.765, 0.298, 0.498], # torso_joint
        [0.765, 0.298, 0.498], # torso_joint
        [0.765, 0.298, 0.498], # torso_joint
        [0.765, 0.298, 0.498], # torso_joint
        [0.765, 0.298, 0.498], # torso_joint
        [0.765, 0.298, 0.498], # torso_joint
        [0.765, 0.298, 0.498], # torso_joint
        [0.765, 0.298, 0.498], # torso_joint
        [0.765, 0.298, 0.498], # torso_joint
        [0.765, 0.298, 0.498], # torso_joint
    ])
    
@dataclass
class Motion:
    motion_file: str = 'resources/motions/h1/stable_punch.pkl'
    skeleton_file: str = 'resources/robots/h1/xml/h1.xml'
    
    dt: Optional[float] = None
    
    sync: bool = False
    
    test_keys: Optional[List[str]] = None
    visualize_config: VisaulizeConfig = field(default_factory=VisaulizeConfig)

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
    
    motion: Motion = field(default_factory=Motion)
    
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