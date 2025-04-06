from dataclasses import dataclass, field
from typing import Optional, Union

import tyro

import isaacgym
from isaacgym import gymapi

from legged_gym.envs import *
from legged_gym.utils import task_registry

@dataclass  
class Args:
    # Resume training or start testing from a checkpoint. Overrides config file if provided.
    task: str = "h1_lower"
    env_cfg: H1Cfg = field(default_factory=H1Cfg)
    train_cfg: H1PPOCfg = field(default_factory=H1PPOCfg)
    # Resume training from a checkpoint
    resume: bool = False
    # Name of the experiment to run or load. Overrides config file if provided.
    experiment_name: Optional[str] = None
    # Name of the run. Overrides config file if provided.
    run_name: Optional[str] = None
    # Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.
    load_run: Optional[int] = None
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
        
def train(args: Args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=args.env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=args.train_cfg)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=False)
    
if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)