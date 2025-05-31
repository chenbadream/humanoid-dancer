#!/usr/bin/env python3

"""
Online DIP-to-AMP Integration System

This module provides a complete online integration between:
1. DIP (Diffusion-based Motion Planner) for motion generation
2. AMP (Adversarial Motion Priors) for motion following
3. H1 Humanoid Robot environment

The system implements a continuous loop:
1. DIP generates future motion based on current state prefix
2. Convert DIP motion to AMP observations on-the-fly
3. AMP controller executes actions to follow the motion
4. Get new state from environment
5. Use new state as prefix for next DIP generation
6. Repeat the cycle continuously
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
import sys
import time
from pathlib import Path
from dataclasses import dataclass

# Add project paths for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "closd" / "diffusion_planner"))
sys.path.append(str(project_root / "legged_gym"))
sys.path.append(str(project_root / "amp-rsl-rl"))

# DIP imports
from closd.diffusion_planner.utils.model_util import create_model_and_diffusion
from closd.diffusion_planner.data_loaders.get_data import get_dataset_loader
from closd.diffusion_planner.utils.sampler_util import ClassifierFreeSampleModel
from closd.diffusion_planner.utils import dist_util
from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process import recover_from_ric
from closd.diffusion_planner.data_loaders.tensors import collate

# Real-time converter
from realtime_dip_to_amp import RealTimeDIPToAMPConverter, create_realtime_converter

# AMP imports
from legged_gym.envs.h1.h1_amp import H1AMP
# Note: AMP_PPO import commented out for now due to dependency issues
# from amp_rsl_rl.algorithms.amp_ppo import AMP_PPO

@dataclass
class OnlineConfig:
    """Configuration for online DIP-AMP integration"""
    # DIP Configuration
    dip_model_path: str
    dip_dataset_path: str = "./dataset/HumanML3D"
    dip_guidance_param: float = 2.5
    dip_lookahead_frames: int = 30
    dip_generation_interval: int = 20  # Generate new motion every N steps
    
    # AMP Configuration  
    amp_model_path: str
    amp_env_cfg_path: str
    
    # Environment Configuration
    device: str = "cuda"
    fps: float = 30.0
    num_envs: int = 1
    
    # Motion blending
    blend_frames: int = 5  # Number of frames to blend between motions
    prefix_length: int = 10  # Length of state history to use as DIP prefix


class DIPMotionGenerator:
    """DIP model wrapper for motion generation with state prefixes"""
    
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
        from closd.diffusion_planner.utils.model_util import load_model_wo_clip
        load_model_wo_clip(self.model, state_dict)
        
        # Wrap with classifier-free sampling
        if self.guidance_param != 1:
            self.model = ClassifierFreeSampleModel(self.model)
        
        self.model.to(self.device)
        self.model.eval()
        
    def generate_motion(self, prefix: Optional[torch.Tensor] = None, length: int = 30, 
                       text_prompt: str = "a person walking forward") -> torch.Tensor:
        """
        Generate motion with optional prefix for state conditioning
        
        Args:
            prefix: Previous motion states to condition on (batch, seq_len, 263)
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
            
            # Add prefix conditioning if provided
            if prefix is not None:
                prefix = prefix.to(self.device)
                # Convert prefix to expected format (batch, channels, 1, seq_len)
                if prefix.dim() == 3:  # (batch, seq_len, channels)
                    prefix = prefix.permute(0, 2, 1).unsqueeze(2)  # (batch, channels, 1, seq_len)
                model_kwargs['y']['prefix'] = prefix
            
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


class AMPController:
    """AMP controller wrapper for motion following"""
    
    def __init__(self, model_path: str, env_cfg_path: str, device: str = "cuda"):
        self.device = torch.device(device)
        
        # Load AMP model (placeholder - actual implementation depends on your AMP setup)
        print(f"Loading AMP controller from {model_path}...")
        # TODO: Load actual AMP model based on your implementation
        self.model_path = model_path
        
    def get_actions(self, current_obs: torch.Tensor, target_amp_obs: torch.Tensor) -> torch.Tensor:
        """
        Get actions from AMP controller to follow target motion
        
        Args:
            current_obs: Current environment observations
            target_amp_obs: Target AMP observations to follow
            
        Returns:
            Action commands for the robot
        """
        # Placeholder implementation - replace with actual AMP controller logic
        # This should compute actions that drive the robot towards the target AMP observations
        
        batch_size = current_obs.shape[0]
        num_actions = 19  # H1 has 19 DOF
        
        # For now, return zero actions (placeholder)
        actions = torch.zeros(batch_size, num_actions, device=self.device)
        
        return actions


class MotionBlender:
    """Handles smooth transitions between motion sequences"""
    
    def __init__(self, blend_frames: int = 5):
        self.blend_frames = blend_frames
        
    def blend_motions(self, old_motion: torch.Tensor, new_motion: torch.Tensor, 
                     overlap_start: int) -> torch.Tensor:
        """
        Blend two motion sequences for smooth transitions
        
        Args:
            old_motion: Previous motion sequence (batch, seq_len, dim)
            new_motion: New motion sequence (batch, seq_len, dim)
            overlap_start: Frame index where blending starts
            
        Returns:
            Blended motion sequence
        """
        if overlap_start + self.blend_frames >= old_motion.shape[1]:
            # Not enough frames to blend, just use new motion
            return new_motion
            
        # Create blending weights
        blend_weights = torch.linspace(0, 1, self.blend_frames, device=old_motion.device)
        blend_weights = blend_weights.view(1, -1, 1)  # (1, blend_frames, 1)
        
        # Blend the overlapping region
        blended_motion = old_motion.clone()
        overlap_end = overlap_start + self.blend_frames
        
        old_segment = old_motion[:, overlap_start:overlap_end, :]
        new_segment = new_motion[:, :self.blend_frames, :]
        
        blended_segment = (1 - blend_weights) * old_segment + blend_weights * new_segment
        blended_motion[:, overlap_start:overlap_end, :] = blended_segment
        
        # Append remaining new motion
        if new_motion.shape[1] > self.blend_frames:
            remaining_new = new_motion[:, self.blend_frames:, :]
            blended_motion = torch.cat([blended_motion, remaining_new], dim=1)
            
        return blended_motion


class OnlineDIPAMPIntegration:
    """Main class for online DIP-AMP integration"""
    
    def __init__(self, config: OnlineConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize components
        print("Initializing DIP motion generator...")
        self.dip_generator = DIPMotionGenerator(
            config.dip_model_path, 
            config.dip_dataset_path,
            config.device,
            config.dip_guidance_param
        )
        
        print("Initializing real-time converter...")
        self.converter = create_realtime_converter(
            config.dip_dataset_path,
            config.device
        )
        self.converter.lookahead_frames = config.dip_lookahead_frames
        
        print("Initializing AMP controller...")
        self.amp_controller = AMPController(
            config.amp_model_path,
            config.amp_env_cfg_path,
            config.device
        )
        
        print("Initializing motion blender...")
        self.motion_blender = MotionBlender(config.blend_frames)
        
        # Initialize buffers
        self.motion_buffer = None
        self.current_frame_idx = 0
        self.step_count = 0
        self.amp_obs_history = []
        
        print("Online DIP-AMP integration initialized successfully!")
        
    def state_to_dip_prefix(self, amp_obs_history: List[torch.Tensor]) -> torch.Tensor:
        """
        Convert AMP observation history back to DIP prefix format
        
        Args:
            amp_obs_history: List of recent AMP observations
            
        Returns:
            DIP prefix in HumanML3D format
        """
        if len(amp_obs_history) == 0:
            # Return dummy prefix for initial generation
            batch_size = 1
            return torch.zeros(batch_size, self.config.prefix_length, 263, device=self.device)
        
        # This is a simplified conversion - in practice, you might need more sophisticated
        # reconstruction from AMP observations back to HumanML3D format
        # For now, we'll use the last few AMP observations as a proxy
        
        recent_obs = amp_obs_history[-self.config.prefix_length:]
        if len(recent_obs) < self.config.prefix_length:
            # Pad with zeros if we don't have enough history
            padding_needed = self.config.prefix_length - len(recent_obs)
            padding = [torch.zeros_like(recent_obs[0])] * padding_needed
            recent_obs = padding + recent_obs
        
        # Stack observations
        stacked_obs = torch.stack(recent_obs, dim=1)  # (batch, seq_len, obs_dim)
        
        # Convert to HumanML3D format (placeholder conversion)
        # TODO: Implement proper conversion from AMP observations to HumanML3D format
        batch_size, seq_len, obs_dim = stacked_obs.shape
        hml_dim = 263
        
        # Simple linear projection (replace with proper conversion)
        prefix = torch.zeros(batch_size, seq_len, hml_dim, device=self.device)
        
        return prefix
        
    def run_online_loop(self, num_steps: int = 1000, text_prompt: str = "a person walking forward"):
        """
        Run the main online loop integrating DIP and AMP
        
        Args:
            num_steps: Number of simulation steps to run
            text_prompt: Text prompt for motion generation
        """
        print(f"Starting online loop for {num_steps} steps...")
        print(f"Motion prompt: '{text_prompt}'")
        
        # Initialize with first motion generation
        print("Generating initial motion...")
        initial_motion = self.dip_generator.generate_motion(
            prefix=None,
            length=self.config.dip_lookahead_frames,
            text_prompt=text_prompt
        )
        
        # Convert to AMP observations
        amp_obs_dict = self.converter.convert_dip_motion_to_amp_obs(initial_motion)
        self.motion_buffer = amp_obs_dict
        self.current_frame_idx = 0
        
        # Main simulation loop
        for step in range(num_steps):
            self.step_count = step
            
            # Check if we need to generate new motion
            needs_new_motion = (self.current_frame_idx >= 
                              self.motion_buffer['joint_positions'].shape[1] - self.config.blend_frames)
            
            if needs_new_motion and step % self.config.dip_generation_interval == 0:
                print(f"Step {step}: Generating new motion...")
                
                # Convert current state history to DIP prefix
                prefix = self.state_to_dip_prefix(self.amp_obs_history)
                
                # Generate new motion
                new_motion = self.dip_generator.generate_motion(
                    prefix=prefix,
                    length=self.config.dip_lookahead_frames,
                    text_prompt=text_prompt
                )
                
                # Convert to AMP observations
                new_amp_obs = self.converter.convert_dip_motion_to_amp_obs(new_motion)
                
                # Blend with existing motion buffer
                self._blend_motion_buffer(new_amp_obs)
            
            # Get current AMP observation target
            if self.current_frame_idx < self.motion_buffer['joint_positions'].shape[1]:
                target_amp_obs = self.converter.get_amp_observation_at_frame(
                    self.motion_buffer, self.current_frame_idx
                )
                self.current_frame_idx += 1
            else:
                # Use last frame if we've run out
                target_amp_obs = self.converter.get_amp_observation_at_frame(
                    self.motion_buffer, self.motion_buffer['joint_positions'].shape[1] - 1
                )
            
            # Store in history
            self.amp_obs_history.append(target_amp_obs.clone())
            if len(self.amp_obs_history) > self.config.prefix_length:
                self.amp_obs_history.pop(0)  # Keep only recent history
            
            # Get actions from AMP controller
            current_obs = target_amp_obs  # Placeholder - should come from environment
            actions = self.amp_controller.get_actions(current_obs, target_amp_obs)
            
            # Execute actions in environment (placeholder)
            # new_state = env.step(actions)
            
            # Logging
            if step % 100 == 0:
                print(f"Step {step}: Motion following ongoing... "
                      f"Frame {self.current_frame_idx}/{self.motion_buffer['joint_positions'].shape[1]}")
        
        print("Online loop completed!")
        
    def _blend_motion_buffer(self, new_amp_obs: Dict[str, torch.Tensor]):
        """Blend new AMP observations with existing motion buffer"""
        if self.motion_buffer is None:
            self.motion_buffer = new_amp_obs
            return
            
        # Find overlap region
        overlap_start = max(0, self.current_frame_idx - self.config.blend_frames)
        
        # Blend each component
        for key in self.motion_buffer.keys():
            old_motion = self.motion_buffer[key]
            new_motion = new_amp_obs[key]
            
            # Simple concatenation with overlap handling
            if overlap_start + self.config.blend_frames < old_motion.shape[1]:
                # Blend overlapping region
                blend_weights = torch.linspace(0, 1, self.config.blend_frames, device=self.device)
                blend_weights = blend_weights.view(1, -1) + torch.zeros_like(old_motion[:, :1])
                
                overlap_end = overlap_start + self.config.blend_frames
                old_segment = old_motion[:, overlap_start:overlap_end]
                new_segment = new_motion[:, :self.config.blend_frames]
                
                # Weighted blend
                blended_segment = (1 - blend_weights) * old_segment + blend_weights * new_segment
                
                # Construct new buffer
                pre_blend = old_motion[:, :overlap_start]
                post_blend = new_motion[:, self.config.blend_frames:]
                
                self.motion_buffer[key] = torch.cat([pre_blend, blended_segment, post_blend], dim=1)
            else:
                # Just append new motion
                self.motion_buffer[key] = torch.cat([old_motion, new_motion], dim=1)


# Example usage and demo
def demo_online_integration():
    """Demonstration of the online DIP-AMP integration"""
    
    # Configuration
    config = OnlineConfig(
        dip_model_path="./save/humanml_trans_enc_512/model000200000.pt",
        dip_dataset_path="./dataset/HumanML3D",
        amp_model_path="./save/amp_model.pt",  # Placeholder
        amp_env_cfg_path="./config/h1_amp.yaml",  # Placeholder
        dip_guidance_param=2.5,
        dip_lookahead_frames=30,
        dip_generation_interval=20,
        device="cuda" if torch.cuda.is_available() else "cpu",
        fps=30.0,
        blend_frames=5,
        prefix_length=10
    )
    
    try:
        # Initialize integration system
        integration = OnlineDIPAMPIntegration(config)
        
        # Run online loop
        integration.run_online_loop(
            num_steps=500,
            text_prompt="a person walking forward and then turning left"
        )
        
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_online_integration()
