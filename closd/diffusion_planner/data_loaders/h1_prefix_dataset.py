#!/usr/bin/env python3
"""
H1 Prefix Dataset class for diffusion planner training with prefix conditioning.

This implements a dataset loader for H1 humanoid motions that uses motion prefixes
as conditioning instead of text descriptions.
"""

import numpy as np
import os
import random
import torch
from torch.utils import data
from pathlib import Path
import json

# Import utilities from HumanML3D
from .humanml.utils.word_vectorizer import WordVectorizer
import codecs as cs


class H1PrefixDataset(data.Dataset):
    """
    Dataset class for H1 motions with prefix conditioning.
    
    This dataset loads H1 motion sequences and supports prefix conditioning
    where a portion of the motion is used as conditioning for generating the rest.
    """
    
    def __init__(self, opt, mean, std, split_file):
        """
        Initialize H1 prefix dataset.
        
        Args:
            opt: Dataset options/configuration
            mean: Dataset mean for normalization
            std: Dataset std for normalization
            split_file: File containing motion names for this split (train/val/test)
        """
        self.opt = opt
        self.max_length = 20  # Default max length for compatibility
        if hasattr(opt, 'fixed_len') and opt.fixed_len > 0:
            self.max_length = opt.fixed_len
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.min_motion_len = 40  # Minimum motion length
        
        # Load motion names from split file
        motion_names = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                motion_names.append(line.strip())
        
        print(f"Loading {len(motion_names)} motions from {split_file}")
        
        # Load motion data
        data_dict = {}
        length_list = []
        
        for name in motion_names:
            try:
                motion_file = os.path.join(opt.motion_dir, name + '.npy')
                if os.path.exists(motion_file):
                    motion = np.load(motion_file)
                    
                    # Filter by length
                    if len(motion) >= self.min_motion_len and len(motion) <= self.max_motion_length:
                        data_dict[name] = {
                            'motion': motion,
                            'length': len(motion),
                            'motion_id': name
                        }
                        length_list.append(len(motion))
                    else:
                        print(f"Skipping {name}: length {len(motion)} out of range")
                else:
                    print(f"Motion file not found: {motion_file}")
            except Exception as e:
                print(f"Error loading {name}: {e}")
        
        print(f"Successfully loaded {len(data_dict)} motions")
        
        # Sort by length
        name_list, length_list = zip(*sorted(zip(list(data_dict.keys()), length_list), key=lambda x: x[1]))
        
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = list(name_list)
        self.reset_max_len(self.max_length)
    
    def reset_max_len(self, length):
        """Reset maximum length and update pointer."""
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length
    
    def inv_transform(self, data):
        """Inverse transform (denormalize) data."""
        return data * self.std + self.mean
    
    def __len__(self):
        return len(self.data_dict) - self.pointer
    
    def __getitem__(self, item):
        """
        Get a motion sample for prefix conditioning.
        
        Returns motion data that can be split into prefix and target portions
        during training.
        """
        idx = self.pointer + item
        key = self.name_list[idx]
        data = self.data_dict[key]
        motion, m_length, motion_id = data['motion'], data['length'], data['motion_id']
        
        # Handle fixed length
        if hasattr(self.opt, 'fixed_len') and self.opt.fixed_len > 0:
            m_length = self.opt.fixed_len
        else:
            # Crop the motions to multiples of unit_length for consistency
            if hasattr(self.opt, 'unit_length') and self.opt.unit_length > 0:
                unit_length = self.opt.unit_length
                if unit_length < 10:
                    coin = np.random.choice(['single', 'single', 'double'])
                else:
                    coin = 'single'
                
                if coin == 'double':
                    m_length = (m_length // unit_length - 1) * unit_length
                elif coin == 'single':
                    m_length = (m_length // unit_length) * unit_length
        
        # Random crop
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        
        # Z Normalization
        motion = (motion - self.mean) / self.std
        
        # Pad if necessary
        if m_length < self.max_motion_length:
            motion = np.concatenate([
                motion,
                np.zeros((self.max_motion_length - m_length, motion.shape[1]))
            ], axis=0)
        
        # Return format compatible with HumanML3D dataset
        # For prefix conditioning, we don't need text embeddings or tokens
        # The collate function will handle splitting into prefix and target
        dummy_word_embeddings = np.zeros((1, 300))  # Dummy word embeddings
        dummy_pos_one_hots = np.zeros((1, 15))  # Dummy POS tags
        dummy_caption = motion_id  # Use motion ID as caption
        dummy_sent_len = 1
        dummy_tokens = motion_id  # Use motion ID as tokens
        
        return (dummy_word_embeddings, dummy_pos_one_hots, dummy_caption, 
                dummy_sent_len, motion, m_length, dummy_tokens, key)


def collate_fn(batch):
    """Collate function for H1 prefix dataset."""
    batch.sort(key=lambda x: x[3], reverse=True)
    return batch


class H1PrefixDatasetWrapper:
    """
    Wrapper class that mimics the HumanML3D dataset structure
    but uses prefix conditioning instead of text conditioning.
    """
    
    def __init__(self, opt, mean, std, split_file):
        self.dataset = H1PrefixDataset(opt, mean, std, split_file)
        self.opt = opt
        
        # Dummy word vectorizer for compatibility
        self.w_vectorizer = None
        
        # Dataset statistics
        self.mean = mean
        self.std = std
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
        return self.dataset[item]
    
    def inv_transform(self, data):
        """Inverse transform (denormalize) data."""
        return self.dataset.inv_transform(data)


# Factory function to create H1 prefix dataset
def get_h1_prefix_dataset(split, num_frames, mode='train', abs_path='.', 
                         fixed_len=0, device=None, autoregressive=False, 
                         return_keys=False, cache_path=None):
    """
    Factory function to create H1 prefix dataset.
    
    Args:
        split: 'train', 'val', or 'test'
        num_frames: Number of frames (not used for prefix conditioning)
        mode: Dataset mode
        abs_path: Absolute path to dataset
        fixed_len: Fixed sequence length
        device: Device to load data on
        autoregressive: Whether to use autoregressive mode
        return_keys: Whether to return motion keys
        cache_path: Path to cache directory
    """
    
    # Dataset configuration
    class H1PrefixOpt:
        def __init__(self):
            self.dataset_name = 'h1_prefix'
            self.motion_dir = os.path.join(abs_path, 'dataset/h1_prefix/motions')
            self.max_motion_length = 196
            self.unit_length = 4
            self.fixed_len = fixed_len
            self.return_keys = return_keys
    
    opt = H1PrefixOpt()
    
    # Load dataset statistics
    stats_dir = os.path.join(abs_path, 'dataset/h1_prefix')
    mean = np.load(os.path.join(stats_dir, 'Mean.npy'))
    std = np.load(os.path.join(stats_dir, 'Std.npy'))
    
    # Split file
    split_file = os.path.join(stats_dir, f'{split}.txt')
    
    return H1PrefixDatasetWrapper(opt, mean, std, split_file)


def main():
    """Test the H1 prefix dataset."""
    print("Testing H1 Prefix Dataset...")
    
    # Test dataset creation
    dataset = get_h1_prefix_dataset(
        split='train',
        num_frames=120,
        abs_path='/home/disk2/cba/humanoid-dancer/closd/diffusion_planner'
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample shape: {sample[4].shape}")  # motion shape
        print(f"Motion length: {sample[5]}")
        print(f"Motion key: {sample[7]}")


if __name__ == "__main__":
    main()
