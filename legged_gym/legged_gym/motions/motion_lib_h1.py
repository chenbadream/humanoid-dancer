import numpy as np
import os
import yaml
from tqdm import tqdm
import os.path as osp

import joblib
import torch
import torch.multiprocessing as mp
import copy
import gc
from scipy.spatial.transform import Rotation as sRot
import random
from .motion_lib_base import MotionLibBase, DeviceCache, compute_motion_dof_vels, FixHeightMode
from .humanoid_batch import HumanoidBatch
from easydict import EasyDict

from legged_gym.utils import torch_utils

def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)



USE_CACHE = False
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy
    
    class Patch:

        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy


class MotionLibH1(MotionLibBase):

    def __init__(self, motion_file, device, fix_height=FixHeightMode.no_fix, masterfoot_conifg=None, min_length=-1, im_eval=False, multi_thread=True, extend_hand = True, extend_head = False, mjcf_file="resources/robots/h1/h1.xml", sim_timestep = 1/50):
        super().__init__(motion_file=motion_file, device=device, fix_height=fix_height, masterfoot_conifg=masterfoot_conifg, min_length=min_length, im_eval=im_eval, multi_thread=multi_thread, sim_timestep = sim_timestep)
        self.mesh_parsers = HumanoidBatch(extend_hand=extend_hand, extend_head=extend_head, mjcf_file=mjcf_file)
        return
    
    @staticmethod
    def fix_trans_height(pose_aa, trans, curr_gender_betas, mesh_parsers, fix_height_mode):
        if fix_height_mode == FixHeightMode.no_fix:
            return trans, 0
        
        with torch.no_grad():
            raise NotImplementedError("Fix height is not implemented for H1")
            return trans, diff_fix
        
    def load_motions(self, skeleton_trees, gender_betas, limb_weights, random_sample=True, start_idx=0, max_len=-1, target_heading = None):
        motions = super().load_motions(
            skeleton_trees=skeleton_trees,
            gender_betas=gender_betas,
            limb_weights=limb_weights,
            random_sample=random_sample,
            start_idx=start_idx,
            max_len=max_len,
            target_heading=target_heading
        )

        if "global_translation_extend" in motions[0].__dict__:
            self.gts_t = torch.cat([m.global_translation_extend for m in motions], dim=0).float().to(self._device)
            self.grs_t = torch.cat([m.global_rotation_extend for m in motions], dim=0).float().to(self._device)
            self.gvs_t = torch.cat([m.global_velocity_extend for m in motions], dim=0).float().to(self._device)
            self.gavs_t = torch.cat([m.global_angular_velocity_extend for m in motions], dim=0).float().to(self._device)
        
        return motions

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        # print("non_interval", frame_idx0, frame_idx1)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        return_dict = {}
        
        if "gts_t" in self.__dict__:
            rg_pos_t0 = self.gts_t[f0l]
            rg_pos_t1 = self.gts_t[f1l]
            
            rg_rot_t0 = self.grs_t[f0l]
            rg_rot_t1 = self.grs_t[f1l]
            
            body_vel_t0 = self.gvs_t[f0l]
            body_vel_t1 = self.gvs_t[f1l]
            
            body_ang_vel_t0 = self.gavs_t[f0l]
            body_ang_vel_t1 = self.gavs_t[f1l]
            if offset is None:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1  
            else:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1 + offset[..., None, :]
            rg_rot_t = torch_utils.slerp(rg_rot_t0, rg_rot_t1, blend_exp)
            body_vel_t = (1.0 - blend_exp) * body_vel_t0 + blend_exp * body_vel_t1
            body_ang_vel_t = (1.0 - blend_exp) * body_ang_vel_t0 + blend_exp * body_ang_vel_t1
            
            return_dict['rg_pos_t'] = rg_pos_t
            return_dict['rg_rot_t'] = rg_rot_t
            return_dict['body_vel_t'] = body_vel_t
            return_dict['body_ang_vel_t'] = body_ang_vel_t
            
        return_dict.update(
            super().get_motion_state(
                motion_ids=motion_ids,
                motion_times=motion_times,
                offset=offset,
            )
        )
        return return_dict
        
    @staticmethod
    def load_motion_with_skeleton(ids, motion_data_list, skeleton_trees, gender_betas, fix_height, mesh_parsers, masterfoot_config, target_heading, max_len, queue, pid):
        # ZL: loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
        np.random.seed(np.random.randint(5000)* pid)
        res = {}
        assert (len(ids) == len(motion_data_list))
        for f in range(len(motion_data_list)):
            curr_id = ids[f]  # id for this datasample
            curr_file = motion_data_list[f]
            if not isinstance(curr_file, dict) and osp.isfile(curr_file):
                key = motion_data_list[f].split("/")[-1].split(".")[0]
                curr_file = joblib.load(curr_file)[key]

            seq_len = curr_file['root_trans_offset'].shape[0]
            if max_len == -1 or seq_len < max_len:
                start, end = 0, seq_len
            else:
                start = random.randint(0, seq_len - max_len)
                end = start + max_len

            trans = to_torch(curr_file['root_trans_offset']).clone()[start:end]
            pose_aa = to_torch(curr_file['pose_aa'][start:end]).clone()
            dt = 1/curr_file['fps']

            B, J, N = pose_aa.shape
            
            ##### ZL: randomize the heading ######
            if not target_heading is None:
                start_root_rot = sRot.from_rotvec(pose_aa[0, 0])
                heading_inv_rot = sRot.from_quat(torch_utils.calc_heading_quat_inv(torch.from_numpy(start_root_rot.as_quat()[None, ])))
                heading_delta = sRot.from_quat(target_heading) * heading_inv_rot 
                pose_aa[:, 0] = torch.tensor((heading_delta * sRot.from_rotvec(pose_aa[:, 0])).as_rotvec())

                trans = torch.matmul(trans, torch.from_numpy(heading_delta.as_matrix().squeeze().T))


            # trans, trans_fix = MotionLibSMPL.fix_trans_height(pose_aa, trans, curr_gender_beta, mesh_parsers, fix_height_mode = fix_height)
            curr_motion = mesh_parsers.fk_batch(pose_aa[None, ], trans[None, ], return_full= True, dt = dt)
            curr_motion = EasyDict({k: v.squeeze() if torch.is_tensor(v) else v for k, v in curr_motion.items() })
            
            
            res[curr_id] = (curr_file, curr_motion)
            
        if not queue is None:
            queue.put(res)
        else:
            return res


    
    