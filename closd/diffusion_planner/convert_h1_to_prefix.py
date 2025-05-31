#!/usr/bin/env python3
"""
Convert H1 AMASS dataset to HumanML3D format for prefix-conditioned diffusion planner training.

This script converts the H1 humanoid motion data from AMASS format to the 263-dimensional
feature vectors expected by the HumanML3D diffusion planner, but for prefix conditioning
instead of text conditioning.
"""

import numpy as np
import joblib
import os
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import argparse
from pathlib import Path
import json

# Import motion processing utilities from HumanML3D
import sys
sys.path.append('/home/disk2/cba/humanoid-dancer')
from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process_torch import extract_features_t2m
from closd.diffusion_planner.data_loaders.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain

class H1ToPrefixConverter:
    def __init__(self, h1_dataset_path, output_dir):
        """
        Initialize the converter for prefix conditioning.
        
        Args:
            h1_dataset_path: Path to the H1 AMASS dataset pickle file
            output_dir: Directory to save the converted prefix-conditioned data
        """
        self.h1_dataset_path = h1_dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the H1 dataset
        print(f"Loading H1 dataset from {h1_dataset_path}...")
        self.h1_data = joblib.load(h1_dataset_path)
        
        print(f"Loaded {len(self.h1_data)} motion sequences")
        
        # HumanML3D parameters
        self.target_fps = 20  # HumanML3D uses 20 fps
        self.min_seq_len = 40  # Minimum sequence length in frames
        self.max_seq_len = 196  # Maximum sequence length in frames
        
    def convert_smpl_to_humanml_joints(self, smpl_joints):
        """
        Convert SMPL joint positions (24 joints) to HumanML3D format (22 joints).
        
        SMPL joints (24): pelvis, left_hip, right_hip, spine1, left_knee, right_knee, spine2, 
                         left_ankle, right_ankle, spine3, left_foot, right_foot, neck, 
                         left_collar, right_collar, head, left_shoulder, right_shoulder, 
                         left_elbow, right_elbow, left_wrist, right_wrist, left_hand, right_hand
        
        HumanML3D joints (22): Remove left_hand and right_hand
        """
        # Remove the last 2 joints (left_hand, right_hand) to get 22 joints
        return smpl_joints[:, :22, :]
    
    def downsample_motion(self, motion_data, original_fps=30, target_fps=20):
        """
        Downsample motion from 30fps to 20fps to match HumanML3D.
        """
        seq_len = motion_data.shape[0]
        downsample_ratio = original_fps / target_fps
        
        # Create indices for downsampling
        indices = np.arange(0, seq_len, downsample_ratio).astype(int)
        indices = indices[indices < seq_len]
        
        return motion_data[indices]
    
    def convert_single_motion(self, motion_key, motion_data):
        """
        Convert a single H1 motion sequence to HumanML3D format.
        """
        try:
            # Extract data
            smpl_joints = motion_data['smpl_joints']  # (seq_len, 24, 3)
            fps = motion_data['fps']
            
            # Convert to 22 joints (remove hand joints)
            joints_22 = self.convert_smpl_to_humanml_joints(smpl_joints)
            
            # Downsample from 30fps to 20fps if needed
            if fps == 30:
                joints_22 = self.downsample_motion(joints_22, 30, 20)
            
            seq_len = joints_22.shape[0]
            
            # Filter by sequence length
            if seq_len < self.min_seq_len or seq_len > self.max_seq_len:
                return None, f"Sequence length {seq_len} out of range [{self.min_seq_len}, {self.max_seq_len}]"
            
            # Extract features using HumanML3D processing pipeline
            # This converts joint positions to the 263-dimensional feature vector
            joints_22_tensor = torch.from_numpy(joints_22).float().unsqueeze(0)  # Add batch dimension
            features, _ = extract_features_t2m(joints_22_tensor)
            features = features.squeeze(0).numpy()  # Remove batch dimension and convert back to numpy
            
            return features, None
            
        except Exception as e:
            return None, f"Error processing motion: {str(e)}"
    
    def convert_dataset(self):
        """
        Convert the entire H1 dataset for prefix conditioning.
        """
        print("Converting H1 dataset for prefix conditioning...")
        
        converted_motions = {}
        failed_conversions = []
        
        for motion_key, motion_data in tqdm(self.h1_data.items(), desc="Converting motions"):
            features, error = self.convert_single_motion(motion_key, motion_data)
            
            if features is not None:
                # For prefix conditioning, we don't need text - just the motion data
                converted_motions[motion_key] = {
                    'motion': features,  # (seq_len, 263) feature vectors
                    'length': len(features),
                    'motion_id': motion_key  # Use motion key as identifier
                }
            else:
                failed_conversions.append((motion_key, error))
        
        print(f"Successfully converted {len(converted_motions)} motions")
        print(f"Failed to convert {len(failed_conversions)} motions")
        
        if failed_conversions:
            print("Failed conversions:")
            for key, error in failed_conversions[:10]:  # Show first 10 failures
                print(f"  {key}: {error}")
        
        return converted_motions
    
    def save_prefix_format(self, converted_motions):
        """
        Save the converted motions for prefix conditioning.
        """
        print("Saving converted data for prefix conditioning...")
        
        # Create data directories
        motion_dir = self.output_dir / "motions"
        motion_dir.mkdir(exist_ok=True)
        
        # Save individual motion files
        motion_names = []
        motion_lengths = []
        
        for motion_key, motion_data in tqdm(converted_motions.items(), desc="Saving files"):
            # Save motion as .npy file
            motion_file = motion_dir / f"{motion_key}.npy"
            np.save(motion_file, motion_data['motion'])
            
            motion_names.append(motion_key)
            motion_lengths.append(motion_data['length'])
        
        # Create train/test/val splits
        total_motions = len(motion_names)
        train_split = int(0.8 * total_motions)
        val_split = int(0.9 * total_motions)
        
        # Shuffle with fixed seed for reproducibility
        np.random.seed(42)
        indices = np.random.permutation(total_motions)
        
        train_indices = indices[:train_split]
        val_indices = indices[train_split:val_split]
        test_indices = indices[val_split:]
        
        # Save split files
        with open(self.output_dir / "train.txt", 'w') as f:
            for idx in train_indices:
                f.write(f"{motion_names[idx]}\n")
        
        with open(self.output_dir / "val.txt", 'w') as f:
            for idx in val_indices:
                f.write(f"{motion_names[idx]}\n")
        
        with open(self.output_dir / "test.txt", 'w') as f:
            for idx in test_indices:
                f.write(f"{motion_names[idx]}\n")
        
        print(f"Saved {len(train_indices)} training, {len(val_indices)} validation, {len(test_indices)} test motions")
        
        # Save motion metadata for prefix conditioning
        metadata = {
            'motion_names': motion_names,
            'motion_lengths': motion_lengths,
            'total_motions': total_motions,
            'train_split': len(train_indices),
            'val_split': len(val_indices),
            'test_split': len(test_indices)
        }
        
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Compute and save dataset statistics
        self.compute_dataset_stats(converted_motions, motion_names)
        
        return motion_names
    
    def compute_dataset_stats(self, converted_motions, motion_names):
        """
        Compute mean and std statistics for the dataset.
        """
        print("Computing dataset statistics...")
        
        all_motions = []
        for name in motion_names:
            motion = converted_motions[name]['motion']
            all_motions.append(motion)
        
        # Concatenate all motions
        all_data = np.concatenate(all_motions, axis=0)  # (total_frames, 263)
        
        # Compute statistics
        mean = np.mean(all_data, axis=0)
        std = np.std(all_data, axis=0)
        
        # Save statistics
        np.save(self.output_dir / "Mean.npy", mean)
        np.save(self.output_dir / "Std.npy", std)
        
        print(f"Dataset statistics saved: mean shape {mean.shape}, std shape {std.shape}")
        
        # Print some basic statistics
        print(f"Feature dimensions: {len(mean)}")
        print(f"Total frames: {len(all_data)}")
        print(f"Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
        print(f"Std range: [{std.min():.4f}, {std.max():.4f}]")


def main():
    parser = argparse.ArgumentParser(description="Convert H1 AMASS dataset for prefix conditioning")
    parser.add_argument("--input", type=str, 
                       default="/home/disk2/cba/humanoid-dancer/legged_gym/resources/motions/h1/amass_phc_filtered.pkl",
                       help="Path to H1 AMASS dataset pickle file")
    parser.add_argument("--output", type=str,
                       default="/home/disk2/cba/humanoid-dancer/closd/diffusion_planner/dataset/h1_prefix",
                       help="Output directory for converted prefix-conditioned data")
    
    args = parser.parse_args()
    
    # Create converter and run conversion
    converter = H1ToPrefixConverter(args.input, args.output)
    converted_motions = converter.convert_dataset()
    
    if converted_motions:
        motion_names = converter.save_prefix_format(converted_motions)
        print(f"\nConversion complete! Dataset saved to: {args.output}")
        print(f"Total motions: {len(motion_names)}")
        print("\nThis dataset is ready for prefix conditioning.")
        print("The diffusion planner will use motion prefixes as conditioning instead of text.")
    else:
        print("No motions were successfully converted!")


if __name__ == "__main__":
    main()
