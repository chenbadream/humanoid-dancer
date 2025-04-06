import os
import torch

from loguru import logger

from isaacgym import gymapi
from isaacgym.torch_utils import *

from legged_gym import LEGGED_GYM_ROOT_DIR
from .legged_robot import LeggedRobot
from .humanoid_robot_config import Asset

class HumanoidRobot(LeggedRobot):
    def _create_envs(self):
        super()._create_envs()
        
        asset_cfg: Asset = self.cfg.asset
        
        hip_roll_names = [name for name in self.dof_names if asset_cfg.hip_roll_name in name] if asset_cfg.hip_roll_name else []
        self.hip_roll_joint_indices = torch.zeros(len(hip_roll_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(hip_roll_names)):
            self.hip_roll_joint_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], hip_roll_names[i])
        logger.info(f"hip_roll_joint_indices: {self.hip_roll_joint_indices}; hip_roll_names: {hip_roll_names}")
        
        hip_yaw_names = [name for name in self.dof_names if asset_cfg.hip_yaw_name in name] if asset_cfg.hip_yaw_name else []
        self.hip_yaw_joint_indices = torch.zeros(len(hip_yaw_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(hip_yaw_names)):
            self.hip_yaw_joint_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], hip_yaw_names[i])
        logger.info(f"hip_yaw_joint_indices: {self.hip_yaw_joint_indices}; hip_yaw_names: {hip_yaw_names}")
        
        hip_pitch_names = [name for name in self.dof_names if asset_cfg.hip_pitch_name in name] if asset_cfg.hip_pitch_name else []
        self.hip_pitch_joint_indices = torch.zeros(len(hip_pitch_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(hip_pitch_names)):
            self.hip_pitch_joint_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], hip_pitch_names[i])
        logger.info(f"hip_pitch_joint_indices: {self.hip_pitch_joint_indices}; hip_pitch_names: {hip_pitch_names}")
        
        shoulder_roll_names = [name for name in self.dof_names if asset_cfg.shoulder_roll_name in name] if asset_cfg.shoulder_roll_name else []
        self.shoulder_roll_joint_indices = torch.zeros(len(shoulder_roll_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(shoulder_roll_names)):
            self.shoulder_roll_joint_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], shoulder_roll_names[i])
        logger.info(f"shoulder_roll_joint_indices: {self.shoulder_roll_joint_indices}; shoulder_roll_names: {shoulder_roll_names}")
        
        shoulder_yaw_names = [name for name in self.dof_names if asset_cfg.shoulder_yaw_name in name] if asset_cfg.shoulder_yaw_name else []
        self.shoulder_yaw_joint_indices = torch.zeros(len(shoulder_yaw_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(shoulder_yaw_names)):
            self.shoulder_yaw_joint_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], shoulder_yaw_names[i])
        logger.info(f"shoulder_yaw_joint_indices: {self.shoulder_yaw_joint_indices}; shoulder_yaw_names: {shoulder_yaw_names}")

        shoulder_pitch_names = [name for name in self.dof_names if asset_cfg.shoulder_pitch_name in name] if asset_cfg.shoulder_pitch_name else []
        self.shoulder_pitch_joint_indices = torch.zeros(len(shoulder_pitch_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(shoulder_pitch_names)):
            self.shoulder_pitch_joint_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], shoulder_pitch_names[i])
        logger.info(f"shoulder_pitch_joint_indices: {self.shoulder_pitch_joint_indices}; shoulder_pitch_names: {shoulder_pitch_names}")
        
        elbow_names = [name for name in self.dof_names if asset_cfg.elbow_name in name] if asset_cfg.elbow_name else []
        self.elbow_joint_indices = torch.zeros(len(elbow_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(elbow_names)):
            self.elbow_joint_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], elbow_names[i])
        logger.info(f"elbow_joint_indices: {self.elbow_joint_indices}; elbow_names: {elbow_names}")
        
        knee_names = [name for name in self.dof_names if asset_cfg.knee_name in name] if asset_cfg.knee_name else []
        self.knee_joint_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_joint_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], knee_names[i])
        logger.info(f"knee_joint_indices: {self.knee_joint_indices}; knee_names: {knee_names}")
        
        wrist_roll_names = [name for name in self.dof_names if asset_cfg.wrist_roll_name in name] if asset_cfg.wrist_roll_name else []
        self.wrist_roll_joint_indices = torch.zeros(len(wrist_roll_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(wrist_roll_names)):
            self.wrist_roll_joint_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], wrist_roll_names[i])
        logger.info(f"wrist_roll_joint_indices: {self.wrist_roll_joint_indices}; wrist_roll_names: {wrist_roll_names}")
        
        wrist_yaw_names = [name for name in self.dof_names if asset_cfg.wrist_yaw_name in name] if asset_cfg.wrist_yaw_name else []
        self.wrist_yaw_joint_indices = torch.zeros(len(wrist_yaw_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(wrist_yaw_names)):
            self.wrist_yaw_joint_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], wrist_yaw_names[i])
        logger.info(f"wrist_yaw_joint_indices: {self.wrist_yaw_joint_indices}; wrist_yaw_names: {wrist_yaw_names}")
        
        wrist_pitch_names = [name for name in self.dof_names if asset_cfg.wrist_pitch_name in name] if asset_cfg.wrist_pitch_name else []
        self.wrist_pitch_joint_indices = torch.zeros(len(wrist_pitch_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(wrist_pitch_names)):
            self.wrist_pitch_joint_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], wrist_pitch_names[i])
        logger.info(f"wrist_pitch_joint_indices: {self.wrist_pitch_joint_indices}; wrist_pitch_names: {wrist_pitch_names}")
        
        ankle_roll_names = [name for name in self.dof_names if asset_cfg.ankle_roll_name in name] if asset_cfg.ankle_roll_name else []
        self.ankle_roll_joint_indices = torch.zeros(len(ankle_roll_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(ankle_roll_names)):
            self.ankle_roll_joint_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], ankle_roll_names[i])
        logger.info(f"ankle_roll_joint_indices: {self.ankle_roll_joint_indices}; ankle_roll_names: {ankle_roll_names}")
        
        ankle_yaw_names = [name for name in self.dof_names if asset_cfg.ankle_yaw_name in name] if asset_cfg.ankle_yaw_name else []
        self.ankle_yaw_joint_indices = torch.zeros(len(ankle_yaw_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(ankle_yaw_names)):
            self.ankle_yaw_joint_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], ankle_yaw_names[i])
        logger.info(f"ankle_yaw_joint_indices: {self.ankle_yaw_joint_indices}; ankle_yaw_names: {ankle_yaw_names}")
        
        ankle_pitch_names = [name for name in self.dof_names if asset_cfg.ankle_pitch_name in name] if asset_cfg.ankle_pitch_name else []
        self.ankle_pitch_joint_indices = torch.zeros(len(ankle_pitch_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(ankle_pitch_names)):
            self.ankle_pitch_joint_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], ankle_pitch_names[i])
        logger.info(f"ankle_pitch_joint_indices: {self.ankle_pitch_joint_indices}; ankle_pitch_names: {ankle_pitch_names}")
        
        waist_roll_names = [name for name in self.dof_names if asset_cfg.waist_roll_name in name] if asset_cfg.waist_roll_name else []
        self.waist_roll_joint_indices = torch.zeros(len(waist_roll_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(waist_roll_names)):
            self.waist_roll_joint_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], waist_roll_names[i])
        logger.info(f"waist_roll_joint_indices: {self.waist_roll_joint_indices}; waist_roll_names: {waist_roll_names}")
        
        waist_yaw_names = [name for name in self.dof_names if asset_cfg.waist_yaw_name in name] if asset_cfg.waist_yaw_name else []
        self.waist_yaw_joint_indices = torch.zeros(len(waist_yaw_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(waist_yaw_names)):
            self.waist_yaw_joint_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], waist_yaw_names[i])
        logger.info(f"waist_yaw_joint_indices: {self.waist_yaw_joint_indices}; waist_yaw_names: {waist_yaw_names}")
        
        waist_pitch_names = [name for name in self.dof_names if asset_cfg.waist_pitch_name in name] if asset_cfg.waist_pitch_name else []
        self.waist_pitch_joint_indices = torch.zeros(len(waist_pitch_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(waist_pitch_names)):
            self.waist_pitch_joint_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], waist_pitch_names[i])
        logger.info(f"waist_pitch_joint_indices: {self.waist_pitch_joint_indices}; waist_pitch_names: {waist_pitch_names}")
        
        