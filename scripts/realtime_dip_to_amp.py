"""
Real-time DIP to AMP Motion Conversion System

This module provides real-time conversion of DIP-generated motions to AMP observations
for online motion following. The system maintains a continuous loop where:
1. DIP generates future motion based on current state prefix
2. Motion is converted to AMP observation format
3. AMP controller follows the motion
4. New state becomes prefix for next generation cycle
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any
import sys
from pathlib import Path

# Add project paths for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "closd" / "diffusion_planner"))

from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process import recover_from_ric, recover_root_rot_pos
from closd.diffusion_planner.data_loaders.humanml.common.quaternion import qrot, qinv
from closd.diffusion_planner.data_loaders.humanml_utils import HML_JOINT_NAMES


class RealTimeDIPToAMPConverter:
    """
    Real-time converter that transforms DIP-generated motion sequences into 
    AMP-compatible observations for online motion following.
    """
    
    def __init__(
        self, 
        mean: torch.Tensor, 
        std: torch.Tensor,
        device: str = "cuda",
        joints_num: int = 22,
        hml_type: Optional[str] = None,
        lookahead_frames: int = 30,  # How many future frames to generate
        fps: float = 30.0
    ):
        """
        Initialize the real-time converter.
        
        Args:
            mean: HumanML3D normalization mean
            std: HumanML3D normalization std
            device: Torch device
            joints_num: Number of joints (22 for HumanML3D)
            hml_type: HumanML3D type (None, 'global', 'global_root')
            lookahead_frames: Number of future frames to generate each cycle
            fps: Frame rate for motion
        """
        self.device = torch.device(device)
        self.mean = mean.to(self.device)
        self.std = std.to(self.device)
        self.joints_num = joints_num
        self.hml_type = hml_type
        self.lookahead_frames = lookahead_frames
        self.dt = 1.0 / fps
        
        # Motion buffer for smooth transitions
        self.motion_buffer = []
        self.current_frame_idx = 0
        
        # Joint mapping (excluding root/pelvis)
        self.joint_names = HML_JOINT_NAMES[1:joints_num]  # Skip pelvis
        self.num_dofs = len(self.joint_names)
        
    def denormalize_hml_vector(self, motion: torch.Tensor) -> torch.Tensor:
        """Denormalize HumanML3D vector representation."""
        if motion.dim() == 2:
            motion = motion.unsqueeze(0)  # Add batch dimension
        
        # Ensure broadcasting works correctly
        mean = self.mean.unsqueeze(0).unsqueeze(0) if self.mean.dim() == 1 else self.mean
        std = self.std.unsqueeze(0).unsqueeze(0) if self.std.dim() == 1 else self.std
        
        return motion * std + mean
    
    def extract_amp_observations(self, joint_positions: torch.Tensor) -> torch.Tensor:
        """
        Extract H1AMP-compatible 105-dimensional observations from 3D joint positions.
        
        Args:
            joint_positions: Shape (batch, seq_len, num_joints, 3)
            
        Returns:
            AMP observations: (batch, seq_len, 105)
        """
        batch_size, seq_len, num_joints, _ = joint_positions.shape
        
        # Extract root information (pelvis is joint 0)
        root_pos = joint_positions[:, :, 0, :]  # (batch, seq_len, 3)
        root_height = root_pos[:, :, 2:3]  # (batch, seq_len, 1) - Z height
        
        # Extract joint positions relative to root (excluding root itself)
        joint_pos_rel = joint_positions[:, :, 1:, :] - root_pos.unsqueeze(2)  # (batch, seq_len, 21, 3)
        joint_pos_flat = joint_pos_rel.reshape(batch_size, seq_len, -1)  # (batch, seq_len, 63)
        
        # Only use first 19 DOF to match H1 robot configuration
        if joint_pos_flat.shape[-1] > 19*3:
            joint_pos_flat = joint_pos_flat[:, :, :19*3]  # (batch, seq_len, 57) -> limit to 19 joints
        elif joint_pos_flat.shape[-1] < 19*3:
            # Pad with zeros if we have fewer joints
            padding = torch.zeros(batch_size, seq_len, 19*3 - joint_pos_flat.shape[-1], device=self.device)
            joint_pos_flat = torch.cat([joint_pos_flat, padding], dim=-1)
        
        # Reshape to get individual joint positions for H1 format
        joint_positions_19 = joint_pos_flat.reshape(batch_size, seq_len, 19, 3)  # First 19 joints only
        joint_pos_relative = joint_positions_19.reshape(batch_size, seq_len, 57)  # Flatten to 19*3=57
        
        # Compute root orientation (tangent-normal representation as in H1AMP)
        root_orientation_quat = self.extract_root_orientation(joint_positions)  # (batch, seq_len, 4)
        root_orientation_tan_norm = self.quat_to_tan_norm(root_orientation_quat)  # (batch, seq_len, 6)
        
        # Compute velocities
        root_lin_vel = torch.zeros_like(root_pos)
        root_ang_vel = torch.zeros_like(root_pos) 
        joint_vel = torch.zeros_like(joint_pos_relative)
        
        if seq_len > 1:
            # Root linear velocity
            root_lin_vel[:, 1:] = (root_pos[:, 1:] - root_pos[:, :-1]) / self.dt
            root_lin_vel[:, 0] = root_lin_vel[:, 1]
            
            # Root angular velocity (simplified from orientation changes)
            quat_diff = self.quat_diff(root_orientation_quat[:, :-1], root_orientation_quat[:, 1:])
            root_ang_vel_computed = self.quat_to_angular_velocity(quat_diff, self.dt)
            root_ang_vel[:, 1:] = root_ang_vel_computed
            root_ang_vel[:, 0] = root_ang_vel[:, 1]
            
            # Joint velocities
            joint_vel[:, 1:] = (joint_pos_relative[:, 1:] - joint_pos_relative[:, :-1]) / self.dt
            joint_vel[:, 0] = joint_vel[:, 1]
        
        # Transform to local frame
        root_lin_vel_local = self.world_to_local_velocity(root_lin_vel, root_orientation_quat)
        root_ang_vel_local = self.world_to_local_velocity(root_ang_vel, root_orientation_quat)
        
        # Projected gravity (assume standard gravity direction)
        projected_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(batch_size, seq_len, 3)
        
        # Previous states (for consecutive timestep approach)
        root_height_prev = torch.cat([root_height[:, :1], root_height[:, :-1]], dim=1)
        root_orientation_prev = torch.cat([root_orientation_tan_norm[:, :1], root_orientation_tan_norm[:, :-1]], dim=1)
        joint_pos_prev = torch.cat([joint_pos_relative[:, :1], joint_pos_relative[:, :-1]], dim=1) 
        root_lin_vel_prev = torch.cat([root_lin_vel_local[:, :1], root_lin_vel_local[:, :-1]], dim=1)
        root_ang_vel_prev = torch.cat([root_ang_vel_local[:, :1], root_ang_vel_local[:, :-1]], dim=1)
        joint_vel_prev = torch.cat([joint_vel[:, :1], joint_vel[:, :-1]], dim=1)
        
        # Build 105-dimensional AMP observation following H1AMP structure
        amp_obs = torch.cat([
            # Current pose: root height (1) + root orientation (6) + joint positions (19*3=57) = 64
            # But H1AMP uses joint positions as 19 values, not 19*3. Let me fix this:
            root_height,                              # 1
            root_orientation_tan_norm,               # 6 
            joint_pos_relative[:, :, :19],           # 19 (first component of each joint)
            
            # Previous pose: root height (1) + root orientation (6) + joint positions (19) = 26
            root_height_prev,                        # 1
            root_orientation_prev,                   # 6
            joint_pos_prev[:, :, :19],              # 19
            
            # Current velocity: root linear vel (3) + root angular vel (3) + joint velocities (19) = 25
            root_lin_vel_local,                      # 3
            root_ang_vel_local,                      # 3 
            joint_vel[:, :, :19],                   # 19
            
            # Previous velocity: root linear vel (3) + root angular vel (3) + joint velocities (19) = 25
            root_lin_vel_prev,                       # 3
            root_ang_vel_prev,                       # 3
            joint_vel_prev[:, :, :19],              # 19
            
            # Additional features
            projected_gravity,                       # 3
        ], dim=-1)
        
        return amp_obs
    
    def extract_root_orientation(self, joint_positions: torch.Tensor) -> torch.Tensor:
        """
        Extract root orientation quaternions from joint positions.
        Uses hip alignment to determine facing direction.
        
        Args:
            joint_positions: Shape (batch, seq_len, num_joints, 3)
            
        Returns:
            Root quaternions in wxyz format: (batch, seq_len, 4)
        """
        batch_size, seq_len, num_joints, _ = joint_positions.shape
        
        if num_joints >= 3:  # Ensure we have hip joints
            # Assuming joints 1 and 2 are left_hip and right_hip
            left_hip = joint_positions[:, :, 1, :]   # (batch, seq_len, 3)
            right_hip = joint_positions[:, :, 2, :]  # (batch, seq_len, 3)
            
            # Compute right direction (from left to right hip)
            right_dir = right_hip - left_hip
            right_dir = torch.nn.functional.normalize(right_dir, dim=-1)
            
            # Up direction (world y-axis)
            up_dir = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand_as(right_dir)
            
            # Forward direction (cross product of up and right)
            forward_dir = torch.cross(up_dir, right_dir, dim=-1)
            forward_dir = torch.nn.functional.normalize(forward_dir, dim=-1)
            
            # Recompute up direction for orthogonality
            up_dir = torch.cross(right_dir, forward_dir, dim=-1)
            
            # Convert to quaternions (simplified - using yaw rotation only)
            yaw_angles = torch.atan2(forward_dir[:, :, 0], forward_dir[:, :, 2])
            
            # Create quaternions from yaw angles (wxyz format)
            quaternions = torch.zeros(batch_size, seq_len, 4, device=self.device)
            quaternions[:, :, 0] = torch.cos(yaw_angles / 2)  # w
            quaternions[:, :, 3] = torch.sin(yaw_angles / 2)  # z
            
        else:
            # Fallback: identity quaternions
            quaternions = torch.zeros(batch_size, seq_len, 4, device=self.device)
            quaternions[:, :, 0] = 1.0  # w component
        
        return quaternions
    
    def world_to_local_velocity(self, world_vel: torch.Tensor, orientation: torch.Tensor) -> torch.Tensor:
        """
        Transform world-frame velocity to local (body) frame.
        
        Args:
            world_vel: World frame velocity (batch, seq_len, 3)
            orientation: Body orientation quaternion wxyz (batch, seq_len, 4)
            
        Returns:
            Local frame velocity (batch, seq_len, 3)
        """
        # Convert quaternion to rotation and apply inverse rotation
        # This is a simplified implementation - for full accuracy, use proper quaternion math
        
        # Extract yaw angle from quaternion (assuming only yaw rotation)
        yaw = 2 * torch.atan2(orientation[:, :, 3], orientation[:, :, 0])
        
        # Create rotation matrices for yaw
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        
        # Apply inverse rotation (transpose of rotation matrix)
        local_vel = torch.zeros_like(world_vel)
        local_vel[:, :, 0] = cos_yaw * world_vel[:, :, 0] + sin_yaw * world_vel[:, :, 2]  # x
        local_vel[:, :, 1] = world_vel[:, :, 1]  # y (unchanged)
        local_vel[:, :, 2] = -sin_yaw * world_vel[:, :, 0] + cos_yaw * world_vel[:, :, 2]  # z
        
        return local_vel
    
    def convert_dip_motion_to_amp_obs(self, dip_motion: torch.Tensor, blend_with_current: bool = False) -> torch.Tensor:
        """
        Convert DIP-generated motion to AMP observations.
        
        Args:
            dip_motion: HumanML3D vector representation 
                       Shape: (seq_len, 263) or (batch, seq_len, 263) or (batch, 263, 1, seq_len)
            blend_with_current: Whether to blend with current motion buffer
                       
        Returns:
            AMP observations tensor (batch, seq_len, 105)
        """
        # Handle different input shapes
        if dip_motion.dim() == 4:  # (batch, 263, 1, seq_len)
            dip_motion = dip_motion.squeeze(2).transpose(1, 2)  # -> (batch, seq_len, 263)
        elif dip_motion.dim() == 2:  # (seq_len, 263)
            dip_motion = dip_motion.unsqueeze(0)  # -> (1, seq_len, 263)
        
        dip_motion = dip_motion.to(self.device)
        
        # Step 1: Denormalize HumanML3D vectors
        denormalized_motion = self.denormalize_hml_vector(dip_motion)
        
        # Step 2: Convert to 3D joint positions
        joint_positions = recover_from_ric(denormalized_motion, self.joints_num, self.hml_type)
        # joint_positions shape: (batch, seq_len, num_joints, 3)
        
        # Step 3: Extract AMP observations (105-dimensional)
        amp_obs = self.extract_amp_observations(joint_positions)
        
        return amp_obs
    
    def get_amp_observation_at_frame(self, amp_obs: torch.Tensor, frame_idx: int) -> torch.Tensor:
        """
        Get AMP observation vector for a specific frame.
        
        Args:
            amp_obs: AMP observations tensor (batch, seq_len, 105)
            frame_idx: Frame index to extract
            
        Returns:
            AMP observation vector for specified frame (batch, 105)
        """
        if frame_idx >= amp_obs.shape[1]:
            frame_idx = amp_obs.shape[1] - 1
            
        return amp_obs[:, frame_idx, :]
    
    def update_motion_buffer(self, new_motion: torch.Tensor, overlap_frames: int = 5):
        """
        Update motion buffer with new DIP-generated motion, handling smooth transitions.
        
        Args:
            new_motion: New motion from DIP (seq_len, 263) or (batch, seq_len, 263)
            overlap_frames: Number of frames to blend for smooth transitions
        """
        # Convert new motion to AMP observations
        new_amp_obs = self.convert_dip_motion_to_amp_obs(new_motion)
        
        if self.motion_buffer is None:
            # First motion - just add it
            self.motion_buffer = new_amp_obs
            self.current_frame_idx = 0
        else:
            # Blend with existing motion for smooth transition
            if self.current_frame_idx + overlap_frames < self.motion_buffer.shape[1]:
                # Create blended transition
                blend_start = self.current_frame_idx
                blend_end = min(blend_start + overlap_frames, self.motion_buffer.shape[1])
                new_blend_end = min(overlap_frames, new_amp_obs.shape[1])
                
                # Linear blending weights
                blend_length = blend_end - blend_start
                if blend_length > 0 and new_blend_end > 0:
                    weights = torch.linspace(0, 1, min(blend_length, new_blend_end), device=self.device)
                    weights = weights.view(1, -1, 1)  # (1, blend_frames, 1)
                    
                    # Blend overlapping region
                    old_segment = self.motion_buffer[:, blend_start:blend_start+len(weights.squeeze()), :]
                    new_segment = new_amp_obs[:, :len(weights.squeeze()), :]
                    
                    blended_segment = (1 - weights) * old_segment + weights * new_segment
                    
                    # Construct new buffer
                    pre_blend = self.motion_buffer[:, :blend_start, :]
                    post_new = new_amp_obs[:, len(weights.squeeze()):, :]
                    
                    self.motion_buffer = torch.cat([pre_blend, blended_segment, post_new], dim=1)
                else:
                    # Just append new motion
                    self.motion_buffer = torch.cat([self.motion_buffer, new_amp_obs], dim=1)
            else:
                # Replace the buffer if we're near the end
                self.motion_buffer = new_amp_obs
                self.current_frame_idx = 0
    
    def get_next_amp_observation(self) -> Tuple[torch.Tensor, bool]:
        """
        Get the next AMP observation from the motion buffer.
        
        Returns:
            Tuple of (amp_observation, needs_new_motion)
            needs_new_motion is True when buffer is running low
        """
        if self.motion_buffer is None:
            raise RuntimeError("Motion buffer is empty. Call update_motion_buffer first.")
        
        seq_len = self.motion_buffer.shape[1]
        
        # Check if we need new motion
        needs_new_motion = (self.current_frame_idx >= seq_len - 10)  # Request new motion when 10 frames left
        
        # Get current observation
        if self.current_frame_idx < seq_len:
            amp_obs = self.get_amp_observation_at_frame(self.motion_buffer, self.current_frame_idx)
            self.current_frame_idx += 1
        else:
            # Use last frame if we've run out
            amp_obs = self.get_amp_observation_at_frame(self.motion_buffer, seq_len - 1)
            needs_new_motion = True
        
        return amp_obs, needs_new_motion
    
    def get_current_state_as_prefix(self, current_obs: torch.Tensor, history_length: int = 10) -> torch.Tensor:
        """
        Convert current AMP observation back to HumanML3D format for use as DIP prefix.
        
        Args:
            current_obs: Current AMP observation vector
            history_length: Length of history to use as prefix
            
        Returns:
            HumanML3D vector for use as DIP prefix
        """
        # This is a simplified conversion - in practice you might need more sophisticated 
        # state reconstruction from AMP observations to HumanML3D format
        
        # Extract joint positions and velocities from AMP observation
        # AMP obs format: [joint_pos, joint_vel, root_lin_vel, root_ang_vel]
        num_dofs = len(self.joint_names)
        
        joint_pos = current_obs[:, :num_dofs]                        # (batch, num_dofs)
        joint_vel = current_obs[:, num_dofs:2*num_dofs]             # (batch, num_dofs)
        root_lin_vel = current_obs[:, 2*num_dofs:2*num_dofs+3]      # (batch, 3)
        root_ang_vel = current_obs[:, 2*num_dofs+3:2*num_dofs+6]    # (batch, 3)
        
        # Convert back to HumanML3D format (this is a placeholder - needs proper implementation)
        # For now, return a dummy prefix
        batch_size = current_obs.shape[0]
        prefix_frames = min(history_length, 10)  # Use last 10 frames as prefix
        hml_vector_dim = 263  # HumanML3D vector dimension
        
        # Create placeholder prefix (in practice, you'd convert from AMP obs to HML format)
        dummy_prefix = torch.zeros(batch_size, prefix_frames, hml_vector_dim, device=self.device)
        
        return dummy_prefix
    
    def quat_to_tan_norm(self, quaternions: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternions to tangent-normal representation (6D)
        
        Args:
            quaternions: (batch, seq_len, 4) in wxyz format
            
        Returns:
            Tangent-normal representation: (batch, seq_len, 6)
        """
        # Extract w, x, y, z components
        w, x, y, z = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
        
        # Convert to rotation matrix first 2 columns (tangent-normal representation)
        # First column of rotation matrix
        col1_x = 1 - 2 * (y*y + z*z)
        col1_y = 2 * (x*y + w*z)
        col1_z = 2 * (x*z - w*y)
        
        # Second column of rotation matrix  
        col2_x = 2 * (x*y - w*z)
        col2_y = 1 - 2 * (x*x + z*z) 
        col2_z = 2 * (y*z + w*x)
        
        # Stack to form 6D representation
        tan_norm = torch.stack([col1_x, col1_y, col1_z, col2_x, col2_y, col2_z], dim=-1)
        
        return tan_norm
    
    def quat_diff(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Compute quaternion difference q1^-1 * q2
        
        Args:
            q1, q2: Quaternions in wxyz format
            
        Returns:
            Quaternion difference
        """
        # Conjugate of q1
        q1_conj = torch.cat([q1[..., :1], -q1[..., 1:]], dim=-1)
        
        # Quaternion multiplication q1_conj * q2
        w1, x1, y1, z1 = q1_conj[..., 0], q1_conj[..., 1], q1_conj[..., 2], q1_conj[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return torch.stack([w, x, y, z], dim=-1)
    
    def quat_to_angular_velocity(self, quat_diff: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Convert quaternion difference to angular velocity
        
        Args:
            quat_diff: Quaternion difference
            dt: Time step
            
        Returns:
            Angular velocity (batch, seq_len-1, 3)
        """
        # Convert to axis-angle
        w = quat_diff[..., 0]
        xyz = quat_diff[..., 1:]
        
        # Compute angle
        angle = 2 * torch.atan2(torch.norm(xyz, dim=-1), torch.abs(w))
        
        # Compute axis (handle singularity)
        norm_xyz = torch.norm(xyz, dim=-1, keepdim=True)
        axis = torch.where(norm_xyz > 1e-6, xyz / norm_xyz, torch.zeros_like(xyz))
        
        # Angular velocity
        angular_vel = axis * angle.unsqueeze(-1) / dt
        
        return angular_vel

    # ...existing code...
    
def create_realtime_converter(dataset_path: str, device: str = "cuda") -> RealTimeDIPToAMPConverter:
    """
    Factory function to create a real-time converter with loaded normalization data.
    
    Args:
        dataset_path: Path to directory containing mean/std files
        device: Torch device
        
    Returns:
        Configured RealTimeDIPToAMPConverter
    """
    dataset_dir = Path(dataset_path)
    
    # Load normalization data
    mean_path = dataset_dir / "t2m_mean.npy"
    std_path = dataset_dir / "t2m_std.npy"
    
    if not mean_path.exists() or not std_path.exists():
        raise FileNotFoundError(f"Could not find normalization files in {dataset_dir}")
    
    mean = torch.from_numpy(np.load(mean_path)).float()
    std = torch.from_numpy(np.load(std_path)).float()
    
    return RealTimeDIPToAMPConverter(mean, std, device=device)


# Example usage for online motion following
class OnlineMotionFollower:
    """
    Example class showing how to use the real-time converter for online motion following.
    """
    
    def __init__(self, converter: RealTimeDIPToAMPConverter, dip_model, amp_controller):
        self.converter = converter
        self.dip_model = dip_model
        self.amp_controller = amp_controller
        
    def run_online_following(self, initial_state: torch.Tensor, num_steps: int = 1000):
        """
        Run online motion following loop.
        
        Args:
            initial_state: Initial environment state
            num_steps: Number of simulation steps to run
        """
        current_state = initial_state
        
        for step in range(num_steps):
            # Check if we need new motion from DIP
            try:
                amp_obs, needs_new_motion = self.converter.get_next_amp_observation()
            except RuntimeError:
                needs_new_motion = True
            
            if needs_new_motion:
                # Get current state as prefix for DIP
                prefix = self.converter.get_current_state_as_prefix(current_state)
                
                # Generate new motion with DIP
                new_motion = self.dip_model.generate(prefix=prefix, length=self.converter.lookahead_frames)
                
                # Update motion buffer
                self.converter.update_motion_buffer(new_motion)
                
                # Get first observation from new motion
                amp_obs, _ = self.converter.get_next_amp_observation()
            
            # Use AMP controller to get actions
            actions = self.amp_controller.get_actions(current_state, amp_obs)
            
            # Execute actions in environment and get new state
            # current_state = env.step(actions)  # This would be your environment step
            
            # For demonstration, just use a dummy update
            current_state = amp_obs  # In practice, this would come from environment
            
            if step % 100 == 0:
                print(f"Step {step}: Motion following ongoing...")
