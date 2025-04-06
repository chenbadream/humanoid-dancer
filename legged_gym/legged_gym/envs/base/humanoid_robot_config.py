from dataclasses import dataclass, field
from typing import List, Optional, Dict

from legged_gym.envs.base import legged_robot_config

@dataclass
class Asset(legged_robot_config.Asset):
    hip_roll_name: Optional[str] = None
    hip_yaw_name: Optional[str] = None
    hip_pitch_name: Optional[str] = None
    
    shoulder_roll_name: Optional[str] = None
    shoulder_yaw_name: Optional[str] = None
    shoulder_pitch_name: Optional[str] = None
    
    elbow_name: Optional[str] = None
    knee_name: Optional[str] = None
    
    wrist_roll_name: Optional[str] = None
    wrist_yaw_name: Optional[str] = None
    wrist_pitch_name: Optional[str] = None
    
    ankle_roll_name: Optional[str] = None
    ankle_yaw_name: Optional[str] = None
    ankle_pitch_name: Optional[str] = None
    
    waist_roll_name: Optional[str] = None
    waist_yaw_name: Optional[str] = None
    waist_pitch_name: Optional[str] = None
    
    