from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union

from legged_gym.envs.h1 import h1_config
from legged_gym.envs.base import legged_robot_config
from .h1_mimic_config import Rewards as MimicRewards

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
    resample_motions_for_envs_interval_s: float = 1000.
    terminate_by_1time_motion: bool = True
    dt: Optional[float] = None
    sync: bool = False
    test_keys: Optional[List[str]] = None
    visualize_config: VisaulizeConfig = field(default_factory=VisaulizeConfig)

@dataclass
class Env(h1_config.Env):
    num_privileged_obs: Optional[int] = None
    num_observations: int = 119  # Policy network uses 119-dimensional observations (same as H1Mimic)

@dataclass
class Rewards(MimicRewards):
    only_positive_rewards: bool = False
    # AMP-specific parameters - prioritize accuracy first
    task_reward_lerp: float = 0.95  # Increased to 95% task, 5% discriminator for better accuracy
    
    # Tighter tracking tolerances for better accuracy
    tracking_joint_pos_sigma: float = 0.25  # Even tighter for better accuracy
    tracking_joint_vel_sigma: float = 6     # Tighter velocity tracking
    tracking_body_rot_sigma: float = 0.06   # Tighter rotation tracking
    tracking_body_vel_sigma: float = 6      # Tighter velocity tracking
    tracking_body_ang_vel_sigma: float = 6  # Tighter angular velocity tracking
    
    # Override scales to add AMP-specific reward term
    scales: Dict[str, float] = field(default_factory=lambda: {
        # Reduce penalty scales to make task rewards less negative
        'torques': -0.000005,  # Reduced penalty
        'torque_limits': -1.,  # Reduced penalty
        'dof_acc': -0.000005,  # Reduced penalty
        'dof_vel': -0.002,  # Reduced penalty
        'lower_action_rate': -1.5,  # Reduced penalty
        'upper_action_rate': -0.3,  # Reduced penalty
        'dof_pos_limits': -50.0,  # Reduced penalty
        'termination': -100.0,  # Reduced penalty
        'feet_contact_forces': -0.4,  # Reduced penalty
        'stumble': -500.0,  # Reduced penalty
        'feet_air_time_tracking': 1000,  # Keep positive rewards
        'slippage': -15.0,  # Reduced penalty
        'feet_ori': -25.0,  # Reduced penalty
        'in_the_air': -100,  # Reduced penalty
        'orientation': -100.0,  # Reduced penalty
        'alive': 2.0,  # Increased positive reward
        'feet_max_height_for_this_air': -1250,  # Reduced penalty
        # Boost motion tracking rewards significantly for better accuracy
        'tracking_selected_joint_position': 32 * 10,  # Increased from 32*8 to 32*10 for even stronger tracking
        'tracking_selected_joint_vel': 25,  # Increased from 20 to 25
        'tracking_root_rotation': 30.0,  # Increased from 25.0 to 30.0
        'tracking_root_vel': 8.0 * 10,  # Increased from 8.0*8 to 8.0*10
        'tracking_root_ang_vel': 8.0 * 10,  # Increased from 8.0*8 to 8.0*10
        # AMP-specific reward term (from discriminator) - minimal for accuracy first
        'amp': 0.1,  # Further reduced from 0.5 to 0.1 to strongly prioritize motion tracking
    })

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
    init_noise_std: float = 1.0
    # Add AMP-specific policy parameters if needed

@dataclass
class Algorithm(h1_config.Algorithm):
    # PPO parameters
    entropy_coef: float = 0.01
    # AMP-specific parameters
    disc_coef: float = 1.0  # discriminator loss coefficient
    disc_learning_rate: float = 1e-5  # DeepMimic: DiscStepSize: 0.00001

@dataclass
class Runner(h1_config.Runner):
    experiment_name: str = 'h1_amp'
    algorithm_class_name: str = 'AMP'  # Use AMP algorithm instead of PPO

@dataclass
class H1AMPCfgPPO:
    seed: int = 1
    runner_class_name: str = 'OnPolicyRunner'
    policy: Policy = field(default_factory=Policy)
    algorithm: Algorithm = field(default_factory=Algorithm)
    runner: Runner = field(default_factory=Runner)
