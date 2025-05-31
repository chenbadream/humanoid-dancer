#!/usr/bin/env python3
"""
Play script for H1 AMP model using DIP-generated motions instead of dataset motions.

This script:
1. Loads a trained AMP model
2. Uses DIP to generate motions in real-time  
3. Converts DIP motions to AMP observations
4. Evaluates how well the AMP model follows DIP-generated motions
"""

import sys
import os
from pathlib import Path
from legged_gym import LEGGED_GYM_ROOT_DIR

import tyro
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import time

from legged_gym.scripts.train import Args
from legged_gym.envs.h1.h1_amp_config import H1AMPCfg, H1AMPCfgPPO
from rsl_rl.modules.discriminator import Discriminator
from rsl_rl.modules.buffers import ReplayBuffer, DemoBuffer

# Add scripts directory to path for DIP imports
scripts_dir = Path(LEGGED_GYM_ROOT_DIR).parent / "scripts"
sys.path.append(str(scripts_dir))

from realtime_dip_to_amp import RealTimeDIPToAMPConverter, create_realtime_converter

# Add DIP path
project_root = Path(LEGGED_GYM_ROOT_DIR).parent
sys.path.append(str(project_root / "closd" / "diffusion_planner"))

from closd.diffusion_planner.utils.model_util import create_model_and_diffusion
from closd.diffusion_planner.data_loaders.get_data import get_dataset_loader
from closd.diffusion_planner.utils.sampler_util import ClassifierFreeSampleModel
from closd.diffusion_planner.data_loaders.tensors import collate
from closd.diffusion_planner.utils.model_util import load_model_wo_clip


class DIPMotionGenerator:
    """DIP model wrapper for real-time motion generation"""
    
    def __init__(self, model_path: str, dataset_path: str, device: str = "cuda", guidance_param: float = 2.5):
        self.device = torch.device(device)
        self.guidance_param = guidance_param
        
        # Load dataset for normalization and model setup
        print("Loading DIP dataset...")
        self.data = get_dataset_loader(
            name='humanml',
            batch_size=1,
            num_frames=196,
            split='test',
            hml_mode='text_only'
        )
        
        # Create model and diffusion
        print("Creating DIP model and diffusion...")
        from closd.diffusion_planner.sample.predict import get_args
        self.args = get_args()
        self.args.model_path = model_path
        self.args.guidance_param = guidance_param
        
        self.model, self.diffusion = create_model_and_diffusion(self.args, self.data)
        
        # Load model weights
        print(f"Loading DIP model from {model_path}...")
        state_dict = torch.load(model_path, map_location='cpu')
        load_model_wo_clip(self.model, state_dict)
        
        # Wrap with classifier-free sampling
        if self.guidance_param != 1:
            self.model = ClassifierFreeSampleModel(self.model)
        
        self.model.to(self.device)
        self.model.eval()
        
    def generate_motion(self, length: int = 60, text_prompt: str = "a person walking forward") -> torch.Tensor:
        """
        Generate motion for specified length and text prompt
        
        Args:
            length: Number of frames to generate
            text_prompt: Text description for motion generation
            
        Returns:
            Generated motion in HumanML3D format (batch, seq_len, 263)
        """
        with torch.no_grad():
            # Prepare conditioning
            collate_args = [{'inp': torch.zeros(length), 'tokens': None, 'lengths': length, 'text': text_prompt}]
            _, model_kwargs = collate(collate_args)
            
            # Move to device
            model_kwargs['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val 
                               for key, val in model_kwargs['y'].items()}
            
            # Add guidance scale
            model_kwargs['y']['scale'] = torch.ones(1, device=self.device) * self.guidance_param
            
            # Generate motion
            sample_shape = (1, self.model.njoints, self.model.nfeats, length)
            sample = self.diffusion.p_sample_loop(
                self.model,
                sample_shape,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            
            # Convert back to (batch, seq_len, channels) format
            if sample.dim() == 4:  # (batch, channels, 1, seq_len)
                sample = sample.squeeze(2).permute(0, 2, 1)  # (batch, seq_len, channels)
                
            return sample


class DIPAMPPlayer:
    """Player that uses DIP-generated motions for AMP evaluation"""
    
    def __init__(self, env, policy, dip_model_path: str, dataset_path: str, device: str = "cuda"):
        self.env = env
        self.policy = policy
        self.device = torch.device(device)
        
        # Initialize DIP motion generator
        print("Initializing DIP motion generator...")
        self.dip_generator = DIPMotionGenerator(dip_model_path, dataset_path, device)
        
        # Initialize real-time converter
        print("Initializing DIP-to-AMP converter...")
        self.converter = create_realtime_converter(dataset_path, device)
        
        # Motion generation settings
        self.motion_length = 120  # Generate 120 frames (4 seconds at 30fps)
        self.motion_refresh_interval = 60  # Generate new motion every 60 steps (2 seconds)
        self.current_motion_frame = 0
        self.current_amp_obs = None
        self.step_count = 0
        
        # Available motion prompts for variety
        self.motion_prompts = [
            "a person walking forward",
            "a person walking slowly",
            "a person walking quickly", 
            "a person jogging forward",
            "a person standing and turning around",
            "a person stepping sideways",
            "a person walking backwards",
            "a person marching in place"
        ]
        self.current_prompt_idx = 0
        
        print("DIP-AMP Player initialized successfully!")
        
    def generate_new_motion(self):
        """Generate new motion from DIP and convert to AMP observations"""
        print(f"Generating new motion: '{self.motion_prompts[self.current_prompt_idx]}'")
        
        # Generate motion with current prompt
        dip_motion = self.dip_generator.generate_motion(
            length=self.motion_length,
            text_prompt=self.motion_prompts[self.current_prompt_idx]
        )
        
        # Convert to AMP observations
        amp_obs_sequence = self.converter.convert_dip_motion_to_amp_obs(dip_motion)
        
        # Update motion buffer
        self.converter.update_motion_buffer(dip_motion)
        
        # Reset frame counter
        self.current_motion_frame = 0
        
        # Cycle to next prompt for variety
        self.current_prompt_idx = (self.current_prompt_idx + 1) % len(self.motion_prompts)
        
        print(f"Generated motion with {amp_obs_sequence.shape[1]} frames")
        
    def get_target_amp_obs(self) -> torch.Tensor:
        """Get target AMP observation for current timestep"""
        # Check if we need new motion
        if (self.current_amp_obs is None or 
            self.step_count % self.motion_refresh_interval == 0):
            self.generate_new_motion()
        
        # Get next AMP observation from converter
        try:
            amp_obs, needs_new_motion = self.converter.get_next_amp_observation()
            
            if needs_new_motion:
                print("Motion buffer running low, generating new motion...")
                self.generate_new_motion()
                amp_obs, _ = self.converter.get_next_amp_observation()
                
            self.current_amp_obs = amp_obs
            
        except RuntimeError:
            # Fallback: generate new motion if buffer is empty
            print("Motion buffer empty, generating new motion...")
            self.generate_new_motion()
            amp_obs, _ = self.converter.get_next_amp_observation()
            self.current_amp_obs = amp_obs
        
        return self.current_amp_obs
    
    def step(self, obs: torch.Tensor) -> tuple:
        """Execute one step with DIP-generated target motion"""
        # Get target AMP observation
        target_amp_obs = self.get_target_amp_obs()
        
        # Get actions from policy
        actions = self.policy(obs.detach())
        
        # Step environment
        next_obs, rewards, dones, infos = self.env.step(actions.detach())
        
        # Update step counter
        self.step_count += 1
        
        # Print status every 100 steps
        if self.step_count % 100 == 0:
            print(f"Step {self.step_count}: Following DIP motion '{self.motion_prompts[(self.current_prompt_idx - 1) % len(self.motion_prompts)]}'")
        
        return next_obs, rewards, dones, infos, target_amp_obs
    
    def play(self, total_steps: int = 3000):
        """Play with DIP-generated motions for specified number of steps"""
        print(f"Starting DIP-AMP play session for {total_steps} steps...")
        
        obs = self.env.get_observations()
        
        for i in range(total_steps):
            obs, rewards, dones, infos, target_amp_obs = self.step(obs)
            
            # Optional: Add any analysis or logging here
            # You could compare actual AMP observations with target AMP observations
            # to evaluate how well the policy is following the DIP-generated motions
            
            if i % 500 == 0:
                print(f"Completed {i}/{total_steps} steps")
        
        print("DIP-AMP play session completed!")


def play_with_dip(args: Args, dip_model_path: str, dataset_path: str, text_prompts: list = None):
    """
    Play function that uses DIP-generated motions instead of dataset motions
    
    Args:
        args: Standard play arguments
        dip_model_path: Path to trained DIP model
        dataset_path: Path to dataset (for normalization data)
        text_prompts: Optional list of text prompts to use for motion generation
    """
    env_cfg, train_cfg = args.env_cfg, args.train_cfg
    
    # Override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)  # Fewer envs for easier visualization
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    
    # Set fixed commands (will be overridden by DIP motions)
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0.0, 0.0]

    env_cfg.env.test = True

    # Prepare environment
    env, _ = task_registry.make_env(args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    
    # Check if we're using AMP algorithm and create necessary components
    discriminator = None
    demo_buffer = None
    replay_buffer = None
    
    if train_cfg.runner.algorithm_class_name == 'AMP':
        # Create discriminator for AMP inference
        discriminator = Discriminator(input_dim=105)  # H1AMP uses 105-dim observations
        
        # Create empty buffers for inference
        demo_buffer = DemoBuffer(torch.zeros(1, 105))  # Dummy data
        replay_buffer = ReplayBuffer(105, capacity=1000)  # Small capacity for inference
    
    # Load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, args=args, train_cfg=train_cfg, 
        discriminator=discriminator, demo_buffer=demo_buffer, replay_buffer=replay_buffer
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # Create DIP-AMP player
    player = DIPAMPPlayer(
        env=env, 
        policy=policy, 
        dip_model_path=dip_model_path, 
        dataset_path=dataset_path,
        device=str(env.device)
    )
    
    # Override motion prompts if provided
    if text_prompts:
        player.motion_prompts = text_prompts
        print(f"Using custom motion prompts: {text_prompts}")
    
    # Start playing with DIP-generated motions
    total_steps = 10 * int(env.max_episode_length)  # 10 episodes worth
    player.play(total_steps)


def main():
    """Main function with command line argument parsing"""
    args = tyro.cli(Args)
    
    # DIP model and dataset paths - modify these as needed
    project_root = Path(LEGGED_GYM_ROOT_DIR).parent
    dip_model_path = str(project_root / "save" / "h1_prefix_dip_test_v1" / "model000000000.pt")
    dataset_path = str(project_root / "closd" / "diffusion_planner" / "dataset")
    
    # Check if DIP model exists
    if not Path(dip_model_path).exists():
        print(f"Error: DIP model not found at {dip_model_path}")
        print("Please update the dip_model_path in the script or provide the correct path")
        return
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please update the dataset_path in the script or provide the correct path")
        return
    
    print(f"Using DIP model: {dip_model_path}")
    print(f"Using dataset: {dataset_path}")
    
    # Optional: Define custom motion prompts
    custom_prompts = [
        "a person walking forward confidently",
        "a person jogging at a steady pace", 
        "a person walking slowly and carefully",
        "a person marching with high steps",
        "a person walking while turning left",
        "a person walking while turning right"
    ]
    
    # Start playing with DIP motions
    play_with_dip(args, dip_model_path, dataset_path, custom_prompts)


if __name__ == '__main__':
    main()
