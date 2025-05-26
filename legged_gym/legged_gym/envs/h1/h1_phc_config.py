from dataclasses import dataclass
from typing import List, Optional

@dataclass
class PHCConfig:
    # AMP parameters
    task_reward_w: float = 0.5
    disc_reward_w: float = 0.5
    disc_coef: float = 0.1
    disc_logit_reg: float = 0.01
    disc_grad_penalty: float = 0.1
    disc_weight_decay: float = 0.0
    disc_reward_scale: float = 1.0
    
    # Buffer parameters
    amp_minibatch_size: int = 32
    amp_obs_demo_buffer_size: int = 10000
    amp_replay_buffer_size: int = 10000
    amp_replay_keep_prob: float = 0.8
    
    # Training parameters
    normalize_amp_input: bool = True
    norm_disc_reward: bool = True
    
    # Motion parameters
    motion_file: str = "path/to/motion/file.npz"
    skeleton_file: str = "path/to/skeleton/file.xml"
    test_keys: Optional[List[str]] = None
    dt: Optional[float] = None
    sync: bool = False
    terminate_by_1time_motion: bool = True
    resample_motions_for_envs_interval_s: float = 10.0 