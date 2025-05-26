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
from .h1_mimic import H1Mimic

class H1AMP(H1Mimic):
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
        # This is a copy of H1Mimic, but you can add AMP-specific features here
        offset = self.env_origins
        B = self.motion_ids.shape[0]
        if self.cfg.motion.sync:
            motion_times = torch.tensor([self._hack_motion_time] * B, dtype=torch.float32, device=self.device).view(-1)
        else:
            motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset=offset)
        # ...existing code from H1Mimic for observation construction...

    def reset_idx(self, env_ids):
        self._resample_motion_times(env_ids)
        return super().reset_idx(env_ids)

    def _reset_dofs(self, env_ids):
        motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times
        offset = self.env_origins
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset=offset)
        self.dof_pos[env_ids] = motion_res['dof_pos'][env_ids]
        self.dof_vel[env_ids] = motion_res['dof_vel'][env_ids]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        if self.custom_origins:
            raise NotImplementedError("Custom origins not implemented for H1AMP")
        else:
            motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times
            offset = self.env_origins
            motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset=offset)
            self.root_states[env_ids, :3] = motion_res['root_pos'][env_ids]
            self.root_states[env_ids, 3:7] = motion_res['root_rot'][env_ids]
            self.root_states[env_ids, 7:10] = motion_res['root_vel'][env_ids]
            self.root_states[env_ids, 10:13] = motion_res['root_ang_vel'][env_ids]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        env_ids_int32 = torch.arange(self.num_envs).to(dtype=torch.int32).to(self.device)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def post_physics_step(self):
        super().post_physics_step()
        if self.cfg.motion.sync:
            self._motion_sync()
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
        if self.common_step_counter % self.cfg.motion.resample_motions_for_envs_interval == 0:
            logger.info("Resampling motions for envs")
            logger.info(f"common_step_counter: {self.common_step_counter}")
            self.resample_motion()

    def _motion_sync(self):
        num_motions = self._motion_lib.num_motions()
        motion_ids = np.arange(self.num_envs, dtype=np.int)
        motion_ids = torch.from_numpy(np.mod(motion_ids, num_motions))
        motion_times = torch.tensor([self._hack_motion_time] * self.num_envs, dtype=torch.float32, device=self.device)
        motion_res = self._get_state_from_motionlib_cache_trimesh(motion_ids, motion_times)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, smpl_params, limb_weights, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
            motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
            motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._set_env_state(env_ids=env_ids, root_pos=root_pos, root_rot=root_rot, dof_pos=dof_pos, root_vel=root_vel, root_ang_vel=root_ang_vel, dof_vel=dof_vel)
        self._reset_env_tensors(env_ids)
        motion_dur = self._motion_lib._motion_lengths[0]
        self._hack_motion_time = np.fmod(self._hack_motion_time + self.motion_dt.cpu().numpy(), motion_dur.cpu().numpy())

    def _init_buffers(self):
        super()._init_buffers()
        self.ref_motion_cache = {}
        self._load_motion()
        self.marker_coords = torch.zeros(self.num_envs, self.num_dofs + 3, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.motion_ids = torch.arange(self.num_envs).to(self.device)
        self.motion_start_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.motion_len = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self._resample_motion_times(env_ids)
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))

    def _load_motion(self):
        cfg_motion: Motion = self.cfg.motion
        motion_path = cfg_motion.motion_file
        skeleton_path = cfg_motion.skeleton_file
        self._motion_lib = MotionLibH1(
            motion_file=motion_path, device=self.device, 
            masterfoot_conifg=None, fix_height=False,
            multi_thread=False, mjcf_file=skeleton_path, 
            sim_timestep=cfg_motion.dt if cfg_motion.dt is not None else self.dt,
        )
        sk_tree = SkeletonTree.from_mjcf(skeleton_path)
        if cfg_motion.test_keys is not None:
            self.motion_data_ids = []
            for idx, keys in enumerate(self._motion_lib._motion_data_keys):
                if keys in cfg_motion.test_keys:
                    self.motion_data_ids.append(idx)
        else:
            self.motion_data_ids = np.arange(len(self._motion_lib._motion_data_keys))
        logger.info(f"Loading {len(self.motion_data_ids)} motions from {motion_path} with skeleton {skeleton_path}")
        self.skeleton_trees = [sk_tree] * self.num_envs
        if self.cfg.env.test:
            self.motion_start_idx = 0
            self._motion_lib.load_motions(
                skeleton_trees=self.skeleton_trees, gender_betas=[torch.zeros(17)] * self.num_envs, 
                limb_weights=[np.zeros(10)] * self.num_envs, 
                random_sample=False, start_idx=self.motion_data_ids[self.motion_start_idx]
            )
        else:
            self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=[torch.zeros(17)] * self.num_envs, limb_weights=[np.zeros(10)] * self.num_envs, random_sample=True)
        self.motion_dt = self._motion_lib._motion_dt
        if cfg_motion.sync:
            self._hack_motion_time = 0.0

    def resample_motion(self):
        if self.cfg.env.test:
            self._motion_lib.load_motions(
                skeleton_trees=self.skeleton_trees, gender_betas=[torch.zeros(17)] * self.num_envs, 
                limb_weights=[np.zeros(10)] * self.num_envs, 
                random_sample=False, start_idx=self.motion_data_ids[self.motion_start_idx]
            )
        else:
            self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=[torch.zeros(17)] * self.num_envs, limb_weights=[np.zeros(10)] * self.num_envs, random_sample=True)
        env_ids = torch.arange(self.num_envs).to(self.device)
        self.reset_idx(env_ids)

    def _resample_motion_times(self, env_ids):
        if len(env_ids) == 0:
            return
        self.motion_len[env_ids] = self._motion_lib.get_motion_length(self.motion_ids[env_ids])
        if self.cfg.env.test:
            self.motion_start_times[env_ids] = 0
        else:
            self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])
        offset = self.env_origins
        motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times
        motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)

    def _get_state_from_motionlib_cache_trimesh(self, motion_ids, motion_times, offset=None):
        if offset is None  or not "motion_ids" in self.ref_motion_cache or self.ref_motion_cache['offset'] is None or len(self.ref_motion_cache['motion_ids']) != len(motion_ids) or len(self.ref_motion_cache['offset']) != len(offset) \
            or  (self.ref_motion_cache['motion_ids'] - motion_ids).abs().sum() + (self.ref_motion_cache['motion_times'] - motion_times).abs().sum() + (self.ref_motion_cache['offset'] - offset).abs().sum() > 0 :
            self.ref_motion_cache['motion_ids'] = motion_ids.clone()
            self.ref_motion_cache['motion_times'] = motion_times.clone()
            self.ref_motion_cache['offset'] = offset.clone() if not offset is None else None
        else:
            return self.ref_motion_cache
        motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times, offset=offset)
        self.ref_motion_cache.update(motion_res)
        return self.ref_motion_cache

    def handle_viewer_action_event(self, evt):
        super().handle_viewer_action_event(evt)
        if evt.action == "prev_motion" and evt.value > 0:
            self.motion_start_idx = (self.motion_start_idx - 1) % len(self.motion_data_ids)
            self.resample_motion()
        elif evt.action == "next_motion" and evt.value > 0:
            self.motion_start_idx = (self.motion_start_idx + 1) % len(self.motion_data_ids)
            self.resample_motion()

    def _setup_viewer(self):
        super()._setup_viewer()
        if not self.headless:
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "prev_motion")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "next_motion")

    def _draw_debug_vis(self):
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        for env_id in range(self.num_envs):
            for pos_id, pos_joint in enumerate(self.marker_coords[env_id]):
                color_inner = (0.3, 0.3, 0.3) if not self.cfg.motion.visualize_config.customize_color \
                                                else self.cfg.motion.visualize_config.marker_joint_colors[pos_id % len(self.cfg.motion.visualize_config.marker_joint_colors)]
                color_inner = tuple(color_inner)
                sphere_geom_marker = gymutil.WireframeSphereGeometry(0.05, 20, 20, None, color=color_inner)
                sphere_pose = gymapi.Transform(gymapi.Vec3(pos_joint[0], pos_joint[1], pos_joint[2]), r=None)
                gymutil.draw_lines(sphere_geom_marker, self.gym, self.viewer, self.envs[env_id], sphere_pose)
