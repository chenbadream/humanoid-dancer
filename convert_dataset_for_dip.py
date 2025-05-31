#!/usr/bin/env python3
"""
DiP Dataset Converter: Convert motion pickle files to HumanML3D format for DiP training

This script converts your motion dataset from the pickle format to the HumanML3D 
263-dimensional feature vectors that can be used to train DiP.
"""

import joblib
import numpy as np
import torch
import sys
import os
from pathlib import Path
from tqdm import tqdm
import argparse

# Add CLoSD modules to path
sys.path.append('/home/chen/workspace/CLoSD-1')

from diffusion_planner.data_loaders.humanml.scripts.motion_process_torch import extract_features_t2m
from diffusion_planner.data_loaders.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain
from diffusion_planner.data_loaders.humanml_utils import HML_JOINT_NAMES, NUM_HML_JOINTS


def smpl_to_humanml3d_skeleton(smpl_joints):
    """
    Convert SMPL joint positions (24 joints) to HumanML3D format (22 joints).
    
    Args:
        smpl_joints: numpy array of shape (seq_len, 24, 3)
    
    Returns:
        humanml3d_joints: numpy array of shape (seq_len, 22, 3)
    """
    # Mapping from SMPL (24 joints) to HumanML3D (22 joints) - exclude jaw and eyes (indices 22, 23)
    SMPL_TO_HML_MAPPING = list(range(22))  # First 22 joints
    return smpl_joints[:, SMPL_TO_HML_MAPPING, :]


def process_single_motion(motion_data, motion_name, feet_threshold=0.002):
    """
    Process a single motion sequence to HumanML3D format.
    
    Args:
        motion_data: Dictionary containing motion data for one sequence
        motion_name: Name/identifier for this motion
        feet_threshold: Threshold for foot contact detection
    
    Returns:
        features: (seq_len-1, 263) HumanML3D feature vectors
        motion_info: Dictionary with motion metadata
    """
    
    # Extract data
    root_trans = motion_data['root_trans_offset']  # (seq_len, 3)
    pose_aa = motion_data['pose_aa']  # (seq_len, 22, 3)
    root_rot = motion_data['root_rot']  # (seq_len, 4) quaternion
    smpl_joints = motion_data['smpl_joints']  # (seq_len, 24, 3)
    fps = motion_data['fps']
    
    seq_len = root_trans.shape[0]
    
    # Convert SMPL joints (24) to HumanML3D format (22 joints)
    positions = smpl_to_humanml3d_skeleton(smpl_joints)
    
    # Convert to torch tensor and add batch dimension
    positions_torch = torch.from_numpy(positions).float().unsqueeze(0)  # (1, seq_len, 22, 3)
    
    try:
        # Extract features using CLoSD's function
        features, recon_data = extract_features_t2m(
            positions_torch,
            feet_thre=feet_threshold,
            n_raw_offsets=t2m_raw_offsets,
            kinematic_chain=t2m_kinematic_chain,
            face_joint_indx=[2, 1, 17, 16],  # HumanML3D face joint indices
            fid_r=[8, 11],  # right ankle, right foot
            fid_l=[7, 10],  # left ankle, left foot
            fix_ik_bug=False
        )
        
        # Remove batch dimension
        features = features.squeeze(0)  # (seq_len-1, 263)
        
        motion_info = {
            'name': motion_name,
            'original_frames': seq_len,
            'processed_frames': features.shape[0],
            'fps': fps,
            'duration': seq_len / fps,
            'feature_dims': features.shape[1]
        }
        
        return features.numpy(), motion_info
        
    except Exception as e:
        print(f"Error processing motion '{motion_name}': {e}")
        return None, None


def convert_dataset_to_dip_format(input_file, output_dir, min_frames=10, max_frames=None, 
                                  feet_threshold=0.002, split_ratio=0.9):
    """
    Convert entire dataset to DiP training format.
    
    Args:
        input_file: Path to input pickle file
        output_dir: Directory to save processed data
        min_frames: Minimum number of frames to include a motion
        max_frames: Maximum number of frames (None for no limit)
        feet_threshold: Threshold for foot contact detection
        split_ratio: Ratio for train/test split
    """
    
    print(f"Loading dataset from {input_file}")
    data = joblib.load(input_file)
    
    if not isinstance(data, dict):
        raise ValueError("Expected dataset to be a dictionary")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(data)} motion sequences")
    
    processed_motions = []
    motion_info_list = []
    failed_motions = []
    
    # Process each motion sequence
    for motion_name, motion_data in tqdm(data.items(), desc="Processing motions"):
        
        # Check minimum frame requirement
        seq_len = motion_data['root_trans_offset'].shape[0]
        if seq_len < min_frames:
            print(f"Skipping '{motion_name}': too few frames ({seq_len} < {min_frames})")
            continue
            
        # Check maximum frame requirement
        if max_frames is not None and seq_len > max_frames:
            print(f"Skipping '{motion_name}': too many frames ({seq_len} > {max_frames})")
            continue
        
        # Process the motion
        features, motion_info = process_single_motion(motion_data, motion_name, feet_threshold)
        
        if features is not None:
            processed_motions.append(features)
            motion_info_list.append(motion_info)
        else:
            failed_motions.append(motion_name)
    
    print(f"\nProcessing complete:")
    print(f"  Successfully processed: {len(processed_motions)} motions")
    print(f"  Failed: {len(failed_motions)} motions")
    
    if len(processed_motions) == 0:
        print("No motions were successfully processed!")
        return
    
    # Split into train and test sets
    total_motions = len(processed_motions)
    train_size = int(total_motions * split_ratio)
    
    train_motions = processed_motions[:train_size]
    train_info = motion_info_list[:train_size]
    
    test_motions = processed_motions[train_size:]
    test_info = motion_info_list[train_size:]
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_motions)} motions")
    print(f"  Testing: {len(test_motions)} motions")
    
    # Save training data
    train_file = output_dir / "train_features.npy"
    train_features_flat = np.concatenate(train_motions, axis=0)  # Concatenate all sequences
    np.save(train_file, train_features_flat)
    print(f"Saved training features: {train_features_flat.shape} -> {train_file}")
    
    # Save test data
    if len(test_motions) > 0:
        test_file = output_dir / "test_features.npy"
        test_features_flat = np.concatenate(test_motions, axis=0)
        np.save(test_file, test_features_flat)
        print(f"Saved test features: {test_features_flat.shape} -> {test_file}")
    
    # Save individual motion files (for more detailed analysis)
    motions_dir = output_dir / "individual_motions"
    motions_dir.mkdir(exist_ok=True)
    
    for i, (features, info) in enumerate(zip(processed_motions, motion_info_list)):
        motion_file = motions_dir / f"motion_{i:06d}.npz"
        np.savez_compressed(motion_file, 
                          features=features,
                          name=info['name'],
                          original_frames=info['original_frames'],
                          processed_frames=info['processed_frames'],
                          fps=info['fps'],
                          duration=info['duration'])
    
    print(f"Saved {len(processed_motions)} individual motion files to {motions_dir}")
    
    # Save metadata
    metadata = {
        'total_motions': len(processed_motions),
        'train_motions': len(train_motions),
        'test_motions': len(test_motions),
        'failed_motions': failed_motions,
        'train_info': train_info,
        'test_info': test_info,
        'feature_dims': 263,
        'fps': 30,
        'processing_params': {
            'min_frames': min_frames,
            'max_frames': max_frames,
            'feet_threshold': feet_threshold,
            'split_ratio': split_ratio
        }
    }
    
    metadata_file = output_dir / "dataset_metadata.pkl"
    joblib.dump(metadata, metadata_file)
    print(f"Saved metadata to {metadata_file}")
    
    # Print summary statistics
    print(f"\nDataset Statistics:")
    all_frame_counts = [info['processed_frames'] for info in motion_info_list]
    all_durations = [info['duration'] for info in motion_info_list]
    
    print(f"  Frame counts - Min: {min(all_frame_counts)}, Max: {max(all_frame_counts)}, Mean: {np.mean(all_frame_counts):.1f}")
    print(f"  Durations - Min: {min(all_durations):.2f}s, Max: {max(all_durations):.2f}s, Mean: {np.mean(all_durations):.2f}s")
    print(f"  Total training frames: {train_features_flat.shape[0]}")
    if len(test_motions) > 0:
        print(f"  Total test frames: {test_features_flat.shape[0]}")
    
    return output_dir


def verify_feature_dimensions(features):
    """Verify that features have the correct HumanML3D format."""
    expected_breakdown = {
        'root_angular_velocity': 1,
        'root_linear_velocity': 2, 
        'root_height': 1,
        'joint_positions_ric': 21 * 3,  # (joints_num-1) * 3 = 63
        'joint_rotations_6d': 21 * 6,   # (joints_num-1) * 6 = 126  
        'joint_velocities': 22 * 3,     # joints_num * 3 = 66
        'foot_contacts': 4              # left_ankle, left_foot, right_ankle, right_foot
    }
    
    expected_total = sum(expected_breakdown.values())
    actual_dims = features.shape[1] if len(features.shape) > 1 else features.shape[0]
    
    print(f"\nFeature Verification:")
    print(f"  Expected dimensions: {expected_total}")
    print(f"  Actual dimensions: {actual_dims}")
    
    if actual_dims == expected_total:
        print("  ✓ Feature dimensions match HumanML3D format!")
        return True
    else:
        print("  ⚠ Warning: Feature dimensions don't match expected format!")
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert motion dataset to DiP training format')
    parser.add_argument('--input', type=str, required=True,
                        help='Input pickle file path')
    parser.add_argument('--output', type=str, default='./processed_dataset',
                        help='Output directory for processed data')
    parser.add_argument('--min_frames', type=int, default=10,
                        help='Minimum number of frames to include a motion')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum number of frames (None for no limit)')
    parser.add_argument('--feet_threshold', type=float, default=0.002,
                        help='Threshold for foot contact detection')
    parser.add_argument('--split_ratio', type=float, default=0.9,
                        help='Ratio for train/test split')
    parser.add_argument('--verify', action='store_true',
                        help='Verify output features format')
    
    args = parser.parse_args()
    
    # Convert the dataset
    output_dir = convert_dataset_to_dip_format(
        input_file=args.input,
        output_dir=args.output,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        feet_threshold=args.feet_threshold,
        split_ratio=args.split_ratio
    )
    
    # Verify the output if requested
    if args.verify and output_dir:
        print(f"\nVerifying output features...")
        train_file = output_dir / "train_features.npy"
        if train_file.exists():
            features = np.load(train_file)
            verify_feature_dimensions(features)
        else:
            print("Training features file not found for verification")


if __name__ == "__main__":
    main()
