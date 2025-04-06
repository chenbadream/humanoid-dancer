import os
import torch

from loguru import logger

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

from legged_gym import LEGGED_GYM_ROOT_DIR
from .legged_robot import LeggedRobot
from .humanoid_robot_config import Asset

class HumanoidRobot(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._setup_viewer()
        
    def _setup_viewer(self):
        if not self.headless:
            # self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_L, "toggle_video_record")
            # self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SEMICOLON, "cancel_video_record")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "follow")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_C, "print_cam")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_K, "show_traj")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_J, "apply_force")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT, "prev_env")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT, "next_env")

        self.follow = True
        self.viewing_env_idx = 0
        self.show_traj = False
        self._cam_prev_char_pos = self.root_states[self.viewing_env_idx, 0:3].cpu().numpy() + np.array([2.5, 2.5, 0])
        
    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self.root_states[self.viewing_env_idx, 0:3].cpu().numpy()

        if self.viewer:
            cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
            cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        else:
            cam_pos = np.array([char_root_pos[0] + 2.5, char_root_pos[1] + 2.5, char_root_pos[2]])

        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], char_root_pos[2])
        # if np.abs(cam_pos[2] - char_root_pos[2]) > 5:
        cam_pos[2] = char_root_pos[2] + 0.5
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], char_root_pos[1] + cam_delta[1], cam_pos[2])

        # self.gym.set_camera_location(self.recorder_camera_handle, self.envs[self.viewing_env_idx], new_cam_pos, new_cam_target)

        if self.follow:
            self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return
    
    def handle_viewer_action_event(self, evt):
        super().handle_viewer_action_event(evt)
        if evt.action == "reset" and evt.value > 0:
            self.reset()
        elif evt.action == "follow" and evt.value > 0:
            self.follow = not self.follow
        elif evt.action == "print_cam" and evt.value > 0:
            cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
            cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
            # print("Print camera", cam_pos)
            logger.info(f"Print camera {cam_pos}")
        elif evt.action == "show_traj" and evt.value > 0:
            self.show_traj = not self.show_traj
            logger.info(f"show_traj: {self.show_traj}")
        elif evt.action == "apply_force" and evt.value > 0:
            forces = torch.zeros((1, self.rigid_body_states.shape[0], 3), device=self.device, dtype=torch.float)
            torques = torch.zeros((1, self.rigid_body_states.shape[0], 3), device=self.device, dtype=torch.float)
            for i in range(self.rigid_body_states.shape[0] // self.num_bodies):
                forces[:, i * self.num_bodies + 3, :] = -3500
                forces[:, i * self.num_bodies + 7, :] = -3500
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)
            
        elif evt.action == "prev_env" and evt.value > 0:
            self.viewing_env_idx = (self.viewing_env_idx - 1) % self.num_envs
            logger.info(f"\nShowing env: {self.viewing_env_idx}")
        elif evt.action == "next_env" and evt.value > 0:
            self.viewing_env_idx = (self.viewing_env_idx + 1) % self.num_envs
            logger.info(f"\nShowing env: {self.viewing_env_idx}")
            
    def render(self, sync_frame_time=True):
        if self.viewer:
            self._update_camera()

        super().render(sync_frame_time)
    
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
        
    def _set_env_state(
        self,
        env_ids,
        root_pos,
        root_rot,
        dof_pos,
        root_vel,
        root_ang_vel,
        dof_vel,
        rigid_body_pos=None,
        rigid_body_rot=None,
        rigid_body_vel=None,
        rigid_body_ang_vel=None,
    ):
        self.root_states[env_ids, 0:3] = root_pos
        self.root_states[env_ids, 3:7] = root_rot
        self.root_states[env_ids, 7:10] = root_vel
        self.root_states[env_ids, 10:13] = root_ang_vel
        
        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel

        if (not rigid_body_pos is None) and (not rigid_body_rot is None):
            # self._rigid_body_pos[env_ids] = rigid_body_pos
            # self._rigid_body_rot[env_ids] = rigid_body_rot
            # self._rigid_body_vel[env_ids] = rigid_body_vel
            # self._rigid_body_ang_vel[env_ids] = rigid_body_ang_vel

            # self._reset_rb_pos = self._rigid_body_pos[env_ids].clone()
            # self._reset_rb_rot = self._rigid_body_rot[env_ids].clone()
            # self._reset_rb_vel = self._rigid_body_vel[env_ids].clone()
            # self._reset_rb_ang_vel = self._rigid_body_ang_vel[env_ids].clone()
            pass
        
    def _reset_env_tensors(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32, device=self.device)

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        # self._terminate_buf[env_ids] = 0
        # self._contact_forces[env_ids] = 0

        return
    
    #------------ reward functions----------------
    
    def _reward_lower_action_rate(self):
        # Penalize changes in actions
        lower_dof_indices = torch.concat([
            self.hip_roll_joint_indices,
            self.hip_yaw_joint_indices,
            self.hip_pitch_joint_indices,
            self.knee_joint_indices,
            self.ankle_roll_joint_indices,
            self.ankle_yaw_joint_indices,
            self.ankle_pitch_joint_indices,
        ])
        return torch.sum(torch.square(self.last_actions[:, lower_dof_indices] - self.actions[:, lower_dof_indices]), dim=1)
    
    def _reward_upper_action_rate(self):
        # Penalize changes in actions
        upper_dof_indices = torch.concat([
            self.shoulder_roll_joint_indices,
            self.shoulder_yaw_joint_indices,
            self.shoulder_pitch_joint_indices,
            self.elbow_joint_indices,
            self.wrist_roll_joint_indices,
            self.wrist_yaw_joint_indices,
            self.wrist_pitch_joint_indices,
            self.waist_roll_joint_indices,
            self.waist_yaw_joint_indices,
            self.waist_pitch_joint_indices,
        ])
        return torch.sum(torch.square(self.last_actions[:, upper_dof_indices] - self.actions[:, upper_dof_indices]), dim=1)
    
    def _reward_slippage(self):
        foot_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
        return torch.sum(torch.norm(foot_vel, dim=-1) * (torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.), dim=1)
    
    def _reward_feet_ori(self):
        left_quat = self.rigid_body_states[:, self.feet_indices[0], 3:7]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.rigid_body_states[:, self.feet_indices[1], 3:7]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        return torch.sum(torch.square(left_gravity[:, :2]), dim=1)**0.5 + torch.sum(torch.square(right_gravity[:, :2]), dim=1)**0.5 
    
    def _reward_in_the_air(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        first_foot_contact = contact_filt[:,0]
        second_foot_contact = contact_filt[:,1]
        reward = ~(first_foot_contact | second_foot_contact)
        return reward
    
    def _reward_feet_max_height_for_this_air(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        from_air_to_contact = torch.logical_and(contact_filt, ~self.last_contacts_filt)
        self.last_contacts = contact
        self.last_contacts_filt = contact_filt

        self.feet_air_max_height = torch.max(self.feet_air_max_height, self.rigid_body_states[:, self.feet_indices, 2])
        
        rew_feet_max_height = torch.sum((torch.clamp_min(self.cfg.rewards.desired_feet_max_height_for_this_air - self.feet_air_max_height, 0)) * from_air_to_contact, dim=1) # reward only on first contact with the ground
        self.feet_air_max_height *= ~contact_filt
        return rew_feet_max_height
