from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class Env:
    num_envs: int = 4096
    num_observations: int = 48
    num_privileged_obs: Optional[int] = None
    num_actions: int = 12
    env_spacing: float = 3.0
    send_timeouts: bool = True
    episode_length_s: float = 20.0
    test: bool = False

@dataclass
class Terrain:
    mesh_type: str = 'plane'
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    border_size: float = 25
    curriculum: bool = True
    static_friction: float = 1.0
    dynamic_friction: float = 1.0
    restitution: float = 0.0
    measure_heights: bool = True
    measured_points_x: List[float] = field(default_factory=lambda: [-0.8 + 0.1 * i for i in range(17)])
    measured_points_y: List[float] = field(default_factory=lambda: [-0.5 + 0.1 * i for i in range(11)])
    selected: bool = False
    terrain_kwargs: Optional[Dict] = None
    max_init_terrain_level: int = 5
    terrain_length: float = 8.0
    terrain_width: float = 8.0
    num_rows: int = 10
    num_cols: int = 20
    terrain_proportions: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.35, 0.25, 0.2])
    slope_treshold: float = 0.75

@dataclass
class Commands:
    curriculum: bool = False
    max_curriculum: float = 1.0
    num_commands: int = 4
    resampling_time: float = 10.0
    heading_command: bool = True

    @dataclass
    class Ranges:
        lin_vel_x: List[float] = field(default_factory=lambda: [-1.0, 1.0])
        lin_vel_y: List[float] = field(default_factory=lambda: [-1.0, 1.0])
        ang_vel_yaw: List[float] = field(default_factory=lambda: [-1.0, 1.0])
        heading: List[float] = field(default_factory=lambda: [-3.14, 3.14])

    ranges: Ranges = field(default_factory=Ranges)

@dataclass
class InitState:
    pos: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    rot: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    lin_vel: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    ang_vel: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    default_joint_angles: Dict[str, float] = field(default_factory=lambda: {"joint_a": 0.0, "joint_b": 0.0})

@dataclass
class Control:
    control_type: str = 'P'
    stiffness: Dict[str, float] = field(default_factory=lambda: {'joint_a': 10.0, 'joint_b': 15.0})
    damping: Dict[str, float] = field(default_factory=lambda: {'joint_a': 1.0, 'joint_b': 1.5})
    action_scale: float = 0.5
    decimation: int = 4

@dataclass
class Asset:
    file: str = ""
    name: str = "legged_robot"
    foot_name: str = "None"
    penalize_contacts_on: List[str] = field(default_factory=list)
    terminate_after_contacts_on: List[str] = field(default_factory=list)
    disable_gravity: bool = False
    collapse_fixed_joints: bool = True
    fix_base_link: bool = False
    default_dof_drive_mode: int = 3
    self_collisions: int = 0
    replace_cylinder_with_capsule: bool = True
    flip_visual_attachments: bool = True
    density: float = 0.001
    angular_damping: float = 0.0
    linear_damping: float = 0.0
    max_angular_velocity: float = 1000.0
    max_linear_velocity: float = 1000.0
    armature: float = 0.0
    thickness: float = 0.01

@dataclass
class DomainRand:
    randomize_friction: bool = True
    friction_range: List[float] = field(default_factory=lambda: [0.5, 1.25])
    randomize_base_mass: bool = False
    added_mass_range: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    push_robots: bool = True
    push_interval_s: float = 15.0
    max_push_vel_xy: float = 1.0

@dataclass
class Rewards:
    scales: Dict[str, float] = field(default_factory=lambda: {
        "termination": -0.0,
        "tracking_lin_vel": 1.0,
        "tracking_ang_vel": 0.5,
        "lin_vel_z": -2.0,
        "ang_vel_xy": -0.05,
        "orientation": -0.0,
        "torques": -0.00001,
        "dof_vel": -0.0,
        "dof_acc": -2.5e-7,
        "base_height": -0.0,
        "feet_air_time": 1.0,
        "collision": -1.0,
        "feet_stumble": -0.0,
        "action_rate": -0.01,
        "stand_still": -0.0,
    })
    only_positive_rewards: bool = True
    tracking_sigma: float = 0.25
    soft_dof_pos_limit: float = 1.0
    soft_dof_vel_limit: float = 1.0
    soft_torque_limit: float = 1.0
    base_height_target: float = 1.0
    max_contact_force: float = 100.0

@dataclass
class Normalization:
    @dataclass
    class ObsScales:
        lin_vel: float = 2.0
        ang_vel: float = 0.25
        dof_pos: float = 1.0
        dof_vel: float = 0.05
        height_measurements: float = 5.0

    obs_scales: ObsScales = field(default_factory=ObsScales)
    clip_observations: float = 100.0
    clip_actions: float = 100.0

@dataclass
class Noise:
    @dataclass
    class NoiseScales:
        dof_pos: float = 0.01
        dof_vel: float = 1.5
        lin_vel: float = 0.1
        ang_vel: float = 0.2
        gravity: float = 0.05
        height_measurements: float = 0.1

    add_noise: bool = True
    noise_level: float = 1.0
    noise_scales: NoiseScales = field(default_factory=NoiseScales)

@dataclass
class Viewer:
    ref_env: int = 0
    pos: List[float] = field(default_factory=lambda: [10, 0, 6])
    lookat: List[float] = field(default_factory=lambda: [11.0, 5, 3.0])

@dataclass
class Sim:
    dt: float = 0.005
    substeps: int = 1
    gravity: List[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    up_axis: int = 1

    @dataclass
    class Physx:
        num_threads: int = 10
        solver_type: int = 1
        num_position_iterations: int = 4
        num_velocity_iterations: int = 0
        contact_offset: float = 0.01
        rest_offset: float = 0.0
        bounce_threshold_velocity: float = 0.5
        max_depenetration_velocity: float = 1.0
        max_gpu_contact_pairs: int = 2 ** 23
        default_buffer_size_multiplier: int = 5
        contact_collection: int = 2

    physx: Physx = field(default_factory=Physx)

@dataclass
class LeggedRobotCfg:
    env: Env = field(default_factory=Env)
    terrain: Terrain = field(default_factory=Terrain)
    commands: Commands = field(default_factory=Commands)
    init_state: InitState = field(default_factory=InitState)
    control: Control = field(default_factory=Control)
    asset: Asset = field(default_factory=Asset)
    domain_rand: DomainRand = field(default_factory=DomainRand)
    rewards: Rewards = field(default_factory=Rewards)
    normalization: Normalization = field(default_factory=Normalization)
    noise: Noise = field(default_factory=Noise)
    viewer: Viewer = field(default_factory=Viewer)
    sim: Sim = field(default_factory=Sim)
    
    seed: int = field(init=False)


@dataclass
class Policy:
    init_noise_std: float = 1.0
    actor_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = 'elu'

@dataclass
class Algorithm:
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1e-3
    schedule: str = 'adaptive'
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0

@dataclass
class Runner:
    policy_class_name: str = 'ActorCritic'
    algorithm_class_name: str = 'PPO'
    num_steps_per_env: int = 24
    max_iterations: int = 1500
    save_interval: int = 50
    experiment_name: str = 'test'
    run_name: str = ''
    resume: bool = False
    load_run: int = -1
    checkpoint: int = -1
    resume_path: Optional[str] = None

@dataclass
class LeggedRobotCfgPPO:
    seed: int = 1
    runner_class_name: str = 'OnPolicyRunner'
    policy: Policy = field(default_factory=Policy)
    algorithm: Algorithm = field(default_factory=Algorithm)
    runner: Runner = field(default_factory=Runner)