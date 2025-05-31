#!/usr/bin/env python3
"""
Prepare dataset for DiP training

This script takes your processed motion features and organizes them in the format
expected by the DiP training pipeline.
"""

import numpy as np
import os
from pathlib import Path
import joblib
from tqdm import tqdm
import shutil

def calculate_normalization_stats(features_list):
    """Calculate mean and std for normalization."""
    all_features = np.concatenate(features_list, axis=0)
    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0)
    
    # Prevent division by zero
    std = np.clip(std, a_min=1e-8, a_max=None)
    
    return mean, std

def create_data_splits(motion_files, train_ratio=0.9):
    """Create train/test splits and generate split files."""
    total_motions = len(motion_files)
    train_size = int(total_motions * train_ratio)
    
    # Shuffle the motion files for random splits
    import random
    random.shuffle(motion_files)
    
    train_files = motion_files[:train_size]
    test_files = motion_files[train_size:]
    
    # Create name lists for split files
    train_names = [f"motion_{i:06d}" for i in range(len(train_files))]
    test_names = [f"motion_{i:06d}" for i in range(len(train_files), len(motion_files))]
    
    return train_files, test_files, train_names, test_names

def save_motion_data_in_hml_format(motion_files, output_dir, names):
    """Save individual motion files in H1 expected format."""
    motion_dir = output_dir / "new_joint_vecs"
    motion_dir.mkdir(exist_ok=True)
    
    cached_data = {}
    
    for motion_file, name in tqdm(zip(motion_files, names), desc="Processing motions"):
        # Load motion data
        data = np.load(motion_file)
        features = data['features']
        
        # Save as .npy file with expected naming
        motion_path = motion_dir / f"{name}.npy"
        np.save(motion_path, features)
        
        # Prepare cached data entry with blank text
        cached_data[name] = {
            'motion': features,
            'length': len(features),
            'text': [{'caption': " ", 'tokens': []}]  # Blank text data
        }
    
    return cached_data

def create_cached_dataset(train_data, test_data, train_names, test_names, output_dir):
    """Create cached dataset files expected by the H1 loader."""
    cache_dir = output_dir / "data" / "h1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create training cache
    train_cache = {
        'name_list': train_names,
        'length_list': [len(train_data[name]['motion']) for name in train_names],
        'data_dict': train_data
    }
    
    test_cache = {
        'name_list': test_names,
        'length_list': [len(test_data[name]['motion']) for name in test_names],
        'data_dict': test_data
    }
    
    # Save cache files with expected naming
    np.save(cache_dir / "h1_train.npy", train_cache)
    np.save(cache_dir / "h1_test.npy", test_cache)
    
    print(f"Saved cached data to {cache_dir}")
    return cache_dir

def prepare_dip_dataset(processed_dir, output_dir, train_ratio=0.9):
    """
    Prepare your processed dataset for DiP training.
    
    Args:
        processed_dir: Directory containing your processed motion data
        output_dir: Output directory for DiP-compatible dataset
        train_ratio: Ratio for train/test split
    """
    
    processed_path = Path(processed_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Preparing DiP dataset from {processed_dir}")
    print(f"Output directory: {output_dir}")
    
    # Find all individual motion files
    motions_dir = processed_path / "individual_motions"
    if not motions_dir.exists():
        raise FileNotFoundError(f"Motion files directory not found: {motions_dir}")
    
    motion_files = list(motions_dir.glob("motion_*.npz"))
    print(f"Found {len(motion_files)} motion files")
    
    if len(motion_files) == 0:
        raise ValueError("No motion files found in the directory")
    
    # Load all features for normalization
    print("Loading features for normalization calculation...")
    all_features = []
    for motion_file in tqdm(motion_files, desc="Loading features"):
        data = np.load(motion_file)
        features = data['features']
        all_features.append(features)
    
    # Calculate normalization statistics
    print("Calculating normalization statistics...")
    mean, std = calculate_normalization_stats(all_features)
    
    print(f"Feature dimensions: {mean.shape[0]}")
    print(f"Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"Std range: [{std.min():.4f}, {std.max():.4f}]")
    
    # Create train/test splits
    print("Creating data splits...")
    train_files, test_files, train_names, test_names = create_data_splits(motion_files, train_ratio)
    
    print(f"Train: {len(train_files)} motions")
    print(f"Test: {len(test_files)} motions")
    
    # Create HumanML3D directory structure
    h1_dir = output_path / "H1"
    h1_dir.mkdir(exist_ok=True)
    
    # Save motion files in expected format
    print("Saving motion files...")
    train_data = save_motion_data_in_hml_format(train_files, h1_dir, train_names)
    test_data = save_motion_data_in_hml_format(test_files, h1_dir, test_names) if test_files else {}
    
    # Create split files
    print("Creating split files...")
    with open(h1_dir / "train.txt", 'w') as f:
        for name in train_names:
            f.write(f"{name}\n")
    
    if test_names:
        with open(h1_dir / "test.txt", 'w') as f:
            for name in test_names:
                f.write(f"{name}\n")
    
    # Create cached dataset
    print("Creating cached dataset...")
    cache_dir = create_cached_dataset(train_data, test_data, train_names, test_names, output_path)
    
    # Save normalization statistics
    print("Saving normalization statistics...")
    np.save(cache_dir / "Mean.npy", mean)
    np.save(cache_dir / "Std.npy", std)
    
    # Create dummy text data (if needed for compatibility)
    texts_dir = h1_dir / "texts"
    texts_dir.mkdir(exist_ok=True)
    
    for name in train_names + test_names:
        with open(texts_dir / f"{name}.txt", 'w') as f:
            f.write(" #motion sequence#0.0#0.0\n")  # Blank text with required format
    
    print(f"\nDataset preparation complete!")
    print(f"Dataset structure:")
    print(f"  {output_path}/")
    print(f"    H1/")
    print(f"      new_joint_vecs/")
    print(f"      texts/")
    print(f"      train.txt")
    print(f"      test.txt")
    print(f"    data/")
    print(f"      h1/")
    print(f"        Mean.npy")
    print(f"        Std.npy")
    print(f"        h1_train.npy")
    print(f"        h1_test.npy")
    
    # Create a summary
    summary = {
        'total_motions': len(motion_files),
        'train_motions': len(train_files),
        'test_motions': len(test_files),
        'feature_dims': mean.shape[0],
        'mean_stats': {'min': float(mean.min()), 'max': float(mean.max()), 'mean': float(mean.mean())},
        'std_stats': {'min': float(std.min()), 'max': float(std.max()), 'mean': float(std.mean())},
        'train_names': train_names[:10],  # Sample names
        'test_names': test_names[:10] if test_names else []
    }
    
    joblib.dump(summary, output_path / "preparation_summary.pkl")
    print(f"Saved preparation summary to {output_path}/preparation_summary.pkl")
    
    return output_path

def verify_dataset_structure(dataset_dir):
    """Verify that the dataset structure is correct for DiP training."""
    dataset_path = Path(dataset_dir)
    
    required_files = [
        "H1/train.txt",
        "H1/test.txt", 
        "data/h1/Mean.npy",
        "data/h1/Std.npy",
        "data/h1/h1_train.npy",
        "data/h1/h1_test.npy"
    ]
    
    print("Verifying dataset structure...")
    all_good = True
    
    for file_path in required_files:
        full_path = dataset_path / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (missing)")
            all_good = False
    
    # Check motion files
    motion_dir = dataset_path / "H1" / "new_joint_vecs"
    if motion_dir.exists():
        motion_files = list(motion_dir.glob("*.npy"))
        print(f"  ✓ {len(motion_files)} motion files in new_joint_vecs/")
    else:
        print(f"  ✗ Motion directory missing")
        all_good = False
    
    if all_good:
        print("✓ Dataset structure verification passed!")
    else:
        print("✗ Dataset structure verification failed!")
    
    return all_good

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare processed motion dataset for DiP training')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory with processed motion data')
    parser.add_argument('--output', type=str, default='./h1_dataset',
                        help='Output directory for DiP-compatible dataset')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Ratio for train/test split')
    parser.add_argument('--verify', action='store_true',
                        help='Verify dataset structure after preparation')
    
    args = parser.parse_args()
    
    # Prepare the dataset
    output_dir = prepare_dip_dataset(
        processed_dir=args.input,
        output_dir=args.output,
        train_ratio=args.train_ratio
    )
    
    # Verify if requested
    if args.verify:
        verify_dataset_structure(output_dir)
