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
    
    resample_motions_for_envs_interval_s: float = 1000.
    terminate_by_1time_motion: bool = True
    dt: Optional[float] = None
    
    sync: bool = False
    
    test_keys: Optional[List[str]] = None
    visualize_config: VisaulizeConfig = field(default_factory=VisaulizeConfig)

@dataclass
class Env(h1_config.Env):
    num_privileged_obs: Optional[int] = None
    num_observations: int = 119
    
@dataclass
class Rewards(h1_config.Rewards):
    only_positive_rewards: bool = False
    max_contact_force: float = 500.0
    tracking_joint_pos_sigma: float = 0.5
    tracking_joint_vel_sigma: float = 10
    tracking_body_rot_sigma: float = 0.1
    tracking_body_vel_sigma: float = 10
    tracking_body_ang_vel_sigma: float = 10
    
    tracking_joint_pos_selection: Dict[str, float] = field(default_factory=lambda: {
        # upper body
        'torso_joint': 2.0,
        'left_shoulder_pitch_joint': 2.0,
        'left_shoulder_roll_joint': 2.0,
        'left_shoulder_yaw_joint': 2.0,
        'left_elbow_joint': 2.0,
        'right_shoulder_pitch_joint': 2.0,
        'right_shoulder_roll_joint': 2.0,
        'right_shoulder_yaw_joint': 2.0,
        'right_elbow_joint': 2.0,
        # lower body
        'left_hip_pitch_joint': 2.0,
        'left_hip_roll_joint': 0.5,
        'left_hip_yaw_joint': 0.5,
        'left_knee_joint': 0.5,
        'left_ankle_joint': 0.5,
        'right_hip_pitch_joint': 2.0,
        'right_hip_roll_joint': 0.5,
        'right_hip_yaw_joint': 0.5,
        'right_knee_joint': 0.5,
        'right_ankle_joint': 0.5
    })
    
    scales: Dict[str, float] = field(default_factory=lambda: {
        'torques': -0.00001,
        'torque_limits': -2.,
        'dof_acc': -0.000011,
        'dof_vel': -0.004,
        'lower_action_rate': -3.0,
        'upper_action_rate': -0.625,
        'dof_pos_limits': -100.0 * 1.25,
        'termination': -200 * 1.25,
        'feet_contact_forces': -0.75,
        'stumble': -1000.0 * 1.25,
        'feet_air_time_tracking': 1000,
        'slippage': -30.0 * 1.25,
        'feet_ori': -50.0 * 1.25,
        'in_the_air': -200,
        'orientation': -200.0,
        'feet_max_height_for_this_air': -2500,
        'tracking_selected_joint_position': 32 * 3,
        'tracking_selected_joint_vel': 16 * 3,
        'tracking_root_rotation': 20.0 * 3,
        'tracking_root_vel': 8.0 * 3,
        'tracking_root_ang_vel': 8.0 * 3,
    })
    
@dataclass
class H1MimicCfg:
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
    name = 'h1_mimic'

@dataclass  
class Policy(h1_config.Policy):
    init_noise_std: float = 1.
    
@dataclass
class Runner( h1_config.Runner ):
    experiment_name: str = 'h1_mimic'

@dataclass
class H1MimicPPOCfg:
    seed: int = 1
    runner_class_name: str = 'OnPolicyRunner'
    policy: Policy = field(default_factory=Policy)
    algorithm: h1_config.Algorithm = field(default_factory=h1_config.Algorithm)
    runner: Runner = field(default_factory=Runner)