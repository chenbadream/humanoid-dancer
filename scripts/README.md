# DIP to AMP Motion Conversion

This directory contains scripts to convert DIP-generated motion back to the original dataset format for AMP training.

## Overview

The DIP (Diffusion Planner) pipeline generates motions in HumanML3D vector representation (263-dimensional for 22 joints). To use these generated motions for AMP training, they need to be converted back to the format expected by the AMP system.

## Conversion Pipeline

The conversion process involves several key steps:

1. **Denormalization**: Apply inverse normalization using mean/std statistics
2. **Vector to Positions**: Convert HumanML3D vector representation to 3D joint positions using `recover_from_ric()`
3. **Root Extraction**: Extract root position and orientation from joint positions
4. **Joint Processing**: Format joint positions and compute velocities
5. **AMP Format**: Package data in the format expected by AMP training

## Files

- `convert_dip_to_amp_simple.py` - Core conversion functions with detailed documentation
- `convert_dip_to_amp.py` - Full-featured command-line conversion tool
- `convert_example.py` - Example script showing how to use the conversion in your workspace
- `README.md` - This documentation file

## Quick Start

### Method 1: Using the Example Script

```bash
cd /home/disk2/cba/humanoid-dancer/scripts
python convert_example.py
```

This will:
- Automatically find normalization files in your workspace
- Load a sample DIP output (or create a dummy one for testing)
- Convert it to AMP format
- Save the result to `converted_motions/example_converted_motion.npy`

### Method 2: Batch Conversion

```bash
python convert_example.py /path/to/dip/samples /path/to/output/directory
```

### Method 3: Using the Full Command-Line Tool

```bash
python convert_dip_to_amp.py \
    --input_dir /path/to/dip/samples \
    --output_dir /path/to/amp/data \
    --mean_std_path /home/disk2/cba/humanoid-dancer/closd/diffusion_planner/dataset \
    --fps 30
```

## Input Format

The scripts expect DIP samples in one of these formats:

1. **HumanML3D Vector**: Shape `(seq_len, 263)` or `(batch, seq_len, 263)`
2. **Batched**: Shape `(batch, 263, 1, seq_len)` - will be automatically reshaped
3. **File formats**: `.npy`, `.pt`, `.pth` files

## Output Format

The converted files are saved as `.npy` files containing a dictionary with:

```python
{
    'joint_positions': List[np.ndarray],  # List of joint position arrays per frame
    'joint_velocities': List[np.ndarray], # List of joint velocity arrays per frame  
    'root_position': np.ndarray,          # Root positions (seq_len, 3)
    'root_quaternion': np.ndarray,        # Root quaternions in xyzw format (seq_len, 4)
    'joints_list': List[str],             # List of joint names (excluding root)
    'fps': int                            # Frame rate
}
```

## Key Functions

### `denormalize_hml_vector(motion, mean, std)`
Applies inverse normalization to recover original scale from normalized HumanML3D vectors.

### `convert_dip_to_amp_format(dip_sample, mean, std, fps=30, joints_num=22)`
Main conversion function that handles the entire pipeline from DIP sample to AMP format.

### `extract_root_orientation_from_joints(joint_positions)`
Computes root orientation quaternions from hip joint positions using cross products.

## Data Flow

```
DIP Sample (HumanML3D Vector)
           ↓
    Denormalization  
           ↓
   recover_from_ric() 
           ↓
   3D Joint Positions
           ↓
  Root & Joint Extraction
           ↓
    AMP Format (.npy)
```

## Normalization Files

The scripts automatically look for normalization files in your workspace:

- `closd/diffusion_planner/dataset/t2m_mean.npy`
- `closd/diffusion_planner/dataset/t2m_std.npy`

Or KIT dataset equivalents:
- `closd/diffusion_planner/dataset/kit_mean.npy`  
- `closd/diffusion_planner/dataset/kit_std.npy`

## Joint Mapping

The conversion uses HumanML3D joint names by default:

```python
HML_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
]
```

The root joint (pelvis) is handled separately, and the remaining 21 joints are included in the output.

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the correct directory and the project paths are set up correctly.

2. **Shape Mismatches**: The scripts handle common input shapes, but if you get shape errors, check that your DIP samples have the expected dimensions.

3. **Missing Normalization Files**: If normalization files aren't found, the scripts will use dummy values. Make sure the paths are correct.

4. **Quaternion Issues**: Root orientation is computed from hip joint positions. If the results look incorrect, you may need to adjust the orientation computation logic.

### Debugging

Add debug prints to see intermediate shapes:

```python
print(f"Input shape: {dip_sample.shape}")
print(f"After denorm: {denormalized.shape}")  
print(f"Joint positions: {joint_positions.shape}")
print(f"Root position: {root_position.shape}")
```

## Customization

### Changing Joint Sets

To use a different set of joints, modify the `joints_list` in the conversion function:

```python
# Custom joint names
custom_joints = ['left_hip', 'right_hip', 'left_knee', 'right_knee', ...]
amp_data['joints_list'] = custom_joints
```

### Adjusting FPS

Change the target frame rate by modifying the `fps` parameter:

```python
amp_data = convert_dip_to_amp_format(sample, mean, std, fps=60)  # 60 FPS
```

### Root Orientation

The current implementation computes root orientation from hip alignment. For different approaches, modify `extract_root_orientation_from_joints()`.

## Integration with AMP Training

Once converted, the `.npy` files can be used directly with the AMP training system:

```python
from amp_rsl_rl.utils.motion_loader import MotionLoader

# Load converted motions for AMP training
motion_loader = MotionLoader(
    device="cuda",
    dataset_path_root="/path/to/converted/motions",
    dataset_names=["converted_motion_001", "converted_motion_002"],
    dataset_weights=[1.0, 1.0],
    simulation_dt=0.02,
    slow_down_factor=1
)
```

## References

- HumanML3D paper and codebase for the vector representation format
- AMP paper for the motion format requirements  
- The original DIP codebase for the conversion functions (`recover_from_ric`, etc.)
