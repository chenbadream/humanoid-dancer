import numpy as np
import torch
from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import to_torch
from loguru import logger
from legged_gym.motions.motion_lib_h1 import MotionLibH1
from legged_gym.utils import torch_utils
from .h1_robot import H1Robot
from .h1_amp_config import Motion
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
from .h1_amp import H1AMP

class H1DIPAMP(H1AMP):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # Set amp_obs_dim BEFORE calling super().__init__() since _get_noise_scale_vec() needs it
        self.amp_obs_dim = 105  # Updated AMP observation dimension for humanoid
        
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # AMP-specific initialization
        self.amp_obs_buf = torch.zeros(self.num_envs, self.amp_obs_dim, dtype=torch.float, device=self.device, requires_grad=False)
        
        # Store previous states for AMP observation construction
        self.prev_dof_pos = torch.zeros_like(self.dof_pos)
        self.prev_dof_vel = torch.zeros_like(self.dof_vel)
        self.prev_base_quat = torch.zeros_like(self.base_quat)
        self.prev_base_lin_vel = torch.zeros_like(self.base_lin_vel)
        self.prev_base_ang_vel = torch.zeros_like(self.base_ang_vel)
        
        # Initialize discriminator reward storage
        self.disc_rewards = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.disc_coef = getattr(cfg.rewards.scales, 'disc_coef', 0.5)  # Default coefficient for discriminator reward
        
        # DIP motion generation storage
        self.dip_motion_buffer = {}  # Buffer for DIP-generated motion sequences
        self.dip_motion_history = {}  # Buffer for past 20 frames for each environment 
        self.dip_motion_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # Frame counter for each env
        self.dip_generation_interval = 20  # Generate new motion every 20 frames
        self.dip_history_length = 20  # Store past 20 frames for DIP input
        self.dip_future_length = 40   # DIP generates 40 future frames
        
        # Initialize DIP motion buffers for each environment
        self._init_dip_buffers()
        
    def _init_dip_buffers(self):
        """Initialize DIP motion buffers for each environment"""
        for env_id in range(self.num_envs):
            # Initialize motion history buffer (past 20 frames)
            self.dip_motion_history[env_id] = {
                'root_pos': torch.zeros((self.dip_history_length, 3), device=self.device),
                'root_rot': torch.zeros((self.dip_history_length, 4), device=self.device),
                'root_vel': torch.zeros((self.dip_history_length, 3), device=self.device),
                'root_ang_vel': torch.zeros((self.dip_history_length, 3), device=self.device),
                'dof_pos': torch.zeros((self.dip_history_length, self.num_dof), device=self.device),
                'dof_vel': torch.zeros((self.dip_history_length, self.num_dof), device=self.device),
                'rg_pos': torch.zeros((self.dip_history_length, self.num_bodies, 3), device=self.device),
                'rb_rot': torch.zeros((self.dip_history_length, self.num_bodies, 4), device=self.device),
                'body_vel': torch.zeros((self.dip_history_length, self.num_bodies, 3), device=self.device),
                'body_ang_vel': torch.zeros((self.dip_history_length, self.num_bodies, 3), device=self.device),
            }
            
            # Initialize motion buffer (future 40 frames from DIP)
            self.dip_motion_buffer[env_id] = {
                'root_pos': torch.zeros((self.dip_future_length, 3), device=self.device),
                'root_rot': torch.zeros((self.dip_future_length, 4), device=self.device),
                'root_vel': torch.zeros((self.dip_future_length, 3), device=self.device),
                'root_ang_vel': torch.zeros((self.dip_future_length, 3), device=self.device),
                'dof_pos': torch.zeros((self.dip_future_length, self.num_dof), device=self.device),
                'dof_vel': torch.zeros((self.dip_future_length, self.num_dof), device=self.device),
                'rg_pos': torch.zeros((self.dip_future_length, self.num_bodies, 3), device=self.device),
                'rg_pos_t': torch.zeros((self.dip_future_length, self.num_bodies, 3), device=self.device),
                'rb_rot': torch.zeros((self.dip_future_length, self.num_bodies, 4), device=self.device),
                'rg_rot_t': torch.zeros((self.dip_future_length, self.num_bodies, 4), device=self.device),
                'body_vel': torch.zeros((self.dip_future_length, self.num_bodies, 3), device=self.device),
                'body_vel_t': torch.zeros((self.dip_future_length, self.num_bodies, 3), device=self.device),
                'body_ang_vel': torch.zeros((self.dip_future_length, self.num_bodies, 3), device=self.device),
                'body_ang_vel_t': torch.zeros((self.dip_future_length, self.num_bodies, 3), device=self.device),
                'valid': False  # Flag to indicate if buffer contains valid DIP data
            }
    
    def set_dip_model(self, dip_model):
        """Set the DIP model for motion generation"""
        self.dip_model = dip_model
    
    def update_motion_history(self, env_ids=None):
        """Update motion history with current robot state for specified environments"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        for env_id in env_ids:
            env_id_cpu = env_id.item()
            # Shift history buffer (remove oldest, add current state)
            history = self.dip_motion_history[env_id_cpu]
            
            # Shift all entries by one position
            for key in history.keys():
                history[key][:-1] = history[key][1:].clone()
            
            # Add current state as the newest entry
            history['root_pos'][-1] = self.root_states[env_id, :3]
            history['root_rot'][-1] = self.root_states[env_id, 3:7]
            history['root_vel'][-1] = self.root_states[env_id, 7:10]
            history['root_ang_vel'][-1] = self.root_states[env_id, 10:13]
            history['dof_pos'][-1] = self.dof_pos[env_id]
            history['dof_vel'][-1] = self.dof_vel[env_id]
            
            # Update body positions and rotations (simplified - you may need to compute these)
            # For now, using root state as approximation
            history['rg_pos'][-1, 0] = self.root_states[env_id, :3]  # Root body
            history['rb_rot'][-1, 0] = self.root_states[env_id, 3:7]  # Root rotation
            history['body_vel'][-1, 0] = self.root_states[env_id, 7:10]  # Root velocity
            history['body_ang_vel'][-1, 0] = self.root_states[env_id, 10:13]  # Root angular velocity
    
    def generate_dip_motion(self, env_id):
        """Generate future motion using DIP for a specific environment"""
        if not hasattr(self, 'dip_model') or self.dip_model is None:
            logger.warning("DIP model not set. Using fallback motion.")
            return self._fallback_motion_generation(env_id)
        
        try:
            # Get past 20 frames for this environment
            history = self.dip_motion_history[env_id]
            
            # Convert history to format expected by DIP model
            # This depends on your DIP model's input format
            past_motion = self._convert_history_to_dip_input(history)
            
            # Generate future 40 frames using DIP
            # This is a placeholder - implement according to your DIP model API
            future_motion = self.dip_model.generate(past_motion)
            
            # Convert DIP output to motion_res format
            motion_res = self._convert_dip_output_to_motion_res(future_motion, env_id)
            
            # Store in buffer
            self.dip_motion_buffer[env_id] = motion_res
            self.dip_motion_buffer[env_id]['valid'] = True
            
            return motion_res
            
        except Exception as e:
            logger.error(f"DIP motion generation failed for env {env_id}: {e}")
            return self._fallback_motion_generation(env_id)
    
    def _convert_history_to_dip_input(self, history):
        """Convert motion history to DIP model input format"""
        # This is a placeholder - implement according to your DIP model's expected input
        # You'll need to adapt this based on your specific DIP model
        pass
    
    def _convert_dip_output_to_motion_res(self, dip_output, env_id):
        """Convert DIP model output to motion_res format compatible with observations"""
        # This is a placeholder - implement according to your DIP model's output format
        # You'll need to adapt this based on your specific DIP model
        pass
    
    def _fallback_motion_generation(self, env_id):
        """Fallback motion generation using the original motion library"""
        # Use parent class method as fallback
        offset = self.env_origins[env_id:env_id+1]
        motion_times = (self.episode_length_buf[env_id:env_id+1] + 1) * self.dt + self.motion_start_times[env_id:env_id+1]
        motion_ids = self.motion_ids[env_id:env_id+1]
        
        return self._get_state_from_motionlib_cache_trimesh(motion_ids, motion_times, offset=offset)
        
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.cfg.motion.resample_motions_for_envs_interval = np.ceil(self.cfg.motion.resample_motions_for_envs_interval_s / self.dt)

    def check_termination(self):
        if self.cfg.motion.terminate_by_1time_motion:
            time = (self.episode_length_buf) * self.dt + self.motion_start_times
            self.time_out_by_1time_motion = time > self.motion_len
            self.time_out_buf = self.time_out_by_1time_motion
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf |= torch.any(torch.abs(self.projected_gravity[:, 0:1]) > 0.7, dim=1)
        self.reset_buf |= torch.any(torch.abs(self.projected_gravity[:, 1:2]) > 0.7, dim=1)
        self.reset_buf |= self.time_out_buf

    def compute_observations(self):
        # For H1AMP, we separate observations for policy and discriminator:
        # - Policy network gets 119-dimensional observations (same as H1Mimic)
        # - Discriminator gets 105-dimensional AMP observations
        
        # Compute 119-dimensional observations for policy network (same as H1Mimic)
        self.obs_buf = self._compute_mimic_observations()
        
        # Compute 105-dimensional AMP observations for discriminator 
        self.amp_obs_buf = self._compute_amp_observations()
        
        # Add noise if needed (only to policy observations)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        
        # Update previous states for next timestep
        self._update_previous_states()
        
    def _compute_mimic_observations(self):
        """Compute 119-dimensional observations for policy network using DIP-generated motion"""
        B = self.motion_ids.shape[0]
        
        # Update motion history with current states
        self.update_motion_history()
        
        # Check if we need to generate new DIP motion for any environment
        envs_need_generation = []
        for env_id in range(self.num_envs):
            counter = self.dip_motion_counter[env_id].item()
            
            # Generate new motion every dip_generation_interval frames or if buffer is invalid
            if (counter % self.dip_generation_interval == 0) or not self.dip_motion_buffer[env_id]['valid']:
                envs_need_generation.append(env_id)
        
        # Generate DIP motion for environments that need it
        for env_id in envs_need_generation:
            self.generate_dip_motion(env_id)
        
        # Increment motion counters
        self.dip_motion_counter += 1
        
        # Get motion_res for each environment from DIP buffer or fallback
        motion_res_list = []
        for env_id in range(self.num_envs):
            if self.dip_motion_buffer[env_id]['valid']:
                # Use DIP-generated motion
                counter = self.dip_motion_counter[env_id].item()
                frame_idx = min(counter % self.dip_generation_interval, self.dip_future_length - 1)
                
                # Extract motion data for current frame
                motion_data = {}
                for key in ['rg_pos', 'rg_pos_t', 'rb_rot', 'rg_rot_t', 'body_vel', 'body_vel_t', 
                           'body_ang_vel', 'body_ang_vel_t', 'dof_pos', 'dof_vel']:
                    if key in self.dip_motion_buffer[env_id]:
                        motion_data[key] = self.dip_motion_buffer[env_id][key][frame_idx].unsqueeze(0)
                
                motion_res_list.append(motion_data)
            else:
                # Fallback to motion library
                fallback_res = self._fallback_motion_generation(env_id)
                motion_res_list.append({k: v[0:1] if v.dim() > 1 else v.unsqueeze(0) for k, v in fallback_res.items()})
        
        # Combine all environment motion data
        motion_res = {}
        for key in motion_res_list[0].keys():
            motion_res[key] = torch.cat([motion_data[key] for motion_data in motion_res_list], dim=0)
        
        # Rest of the observation computation remains the same
        ref_body_pos = motion_res["rg_pos"] 
        ref_body_pos_extend = motion_res["rg_pos_t"]
        ref_body_vel_subset = motion_res["body_vel"] # [num_envs, num_markers, 3]
        ref_body_vel = ref_body_vel_subset
        ref_body_vel_extend = motion_res["body_vel_t"] # [num_envs, num_markers, 3]
        ref_body_rot = motion_res["rb_rot"] # [num_envs, num_markers, 4]
        ref_body_rot_extend = motion_res["rg_rot_t"] # [num_envs, num_markers, 4]
        ref_body_ang_vel = motion_res["body_ang_vel"] # [num_envs, num_markers, 3]
        ref_body_ang_vel_extend = motion_res["body_ang_vel_t"] # [num_envs, num_markers, 3]
        ref_dof_pos = motion_res["dof_pos"] # [num_envs, num_dofs]
        ref_dof_vel = motion_res["dof_vel"] # [num_envs, num_dofs]
        
        self.marker_coords[:] = ref_body_pos_extend.reshape(B, -1, 3)
        
        ref_root_vel = ref_body_vel[:, 0] # [num_envs, 3]
        ref_root_ang_vel = ref_body_ang_vel[:, 0]
        
        root_rot = self.base_quat
        root_vel = self.base_lin_vel
        root_ang_vel = self.base_ang_vel
    
        heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_rot = torch_utils.calc_heading_quat(root_rot)
        
        diff_global_body_rot = torch_utils.quat_mul(ref_body_rot[:, 0], torch_utils.quat_conjugate(root_rot))
        diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_inv_rot.view(-1, 4), diff_global_body_rot.view(-1, 4)), heading_rot.view(-1, 4))
        
        diff_global_root_vel = ref_root_vel.view(B, 1, 3) - root_vel.view(B, 1, 3)
        diff_local_root_vel = torch_utils.my_quat_rotate(heading_inv_rot.view(-1, 4), diff_global_root_vel.view(-1, 3))
        
        diff_global_root_ang_vel = ref_root_ang_vel.view(B, 1, 3) - root_ang_vel.view(B, 1, 3)
        diff_local_root_ang_vel = torch_utils.my_quat_rotate(heading_inv_rot.view(-1, 4), diff_global_root_ang_vel.view(-1, 3))
        
        dof_diff = ref_dof_pos.view(B, 1, -1) - self.dof_pos.view(B, 1, -1)
        dof_vel_diff = ref_dof_vel.view(B, 1, -1) - self.dof_vel.view(B, 1, -1)

        # Build 119-dimensional observation same as H1Mimic
        mimic_obs = torch.cat((  
                                    # self obs (3 + 3 + 3 + 19 + 19 + 19 + 3 = 69)
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale * 0, # do not use commands
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.base_lin_vel * self.obs_scales.lin_vel,
                                    
                                    # task obs (6 + 3 + 3 + 19 + 19 = 50)
                                    torch_utils.quat_to_tan_norm(diff_local_body_rot_flat).view(B, -1),
                                    diff_local_root_vel.view(B, -1) * self.obs_scales.lin_vel,
                                    diff_local_root_ang_vel.view(B, -1) * self.obs_scales.ang_vel,
                                    dof_diff.view(B, -1) * self.obs_scales.dof_pos,
                                    dof_vel_diff.view(B, -1) * self.obs_scales.dof_vel,
                                    ),dim=-1)
        
        return mimic_obs
    
    def reset_idx(self, env_ids):
        """Reset DIP buffers and counters for specified environments"""
        # Reset DIP motion counters for specified environments
        self.dip_motion_counter[env_ids] = 0
        
        # Reset DIP motion buffers for specified environments
        for env_id in env_ids:
            env_id_cpu = env_id.item()
            # Mark buffer as valid to force regeneration
            self.dip_motion_buffer[env_id_cpu]['valid'] = False
            
            # Clear motion history (will be rebuilt as simulation progresses)
            history = self.dip_motion_history[env_id_cpu]
            for key in history.keys():
                history[key].zero_()
        
        # Call parent reset
        return super().reset_idx(env_ids)
    
    def get_dip_motion_info(self):
        """Get information about DIP motion generation status"""
        valid_buffers = sum(1 for env_id in range(self.num_envs) if self.dip_motion_buffer[env_id]['valid'])
        return {
            'valid_buffers': valid_buffers,
            'total_envs': self.num_envs,
            'generation_interval': self.dip_generation_interval,
            'motion_counters': self.dip_motion_counter.cpu().numpy().tolist()
        }
