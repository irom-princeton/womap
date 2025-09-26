from typing import List, Sequence, Literal
import yaml
import cv2
import numpy as np
import torch
from torchvision import transforms as tv_transforms
from pathlib import Path
import os
import json
from tqdm.auto import tqdm
from pytorch3d import transforms as py3d_transforms
from torch.nn.functional import normalize
import matplotlib.pyplot as plt
import roma

# from src.models.gsplatmodel import GSScene # TODO
from src.scenes.pybullet_scene import PyBulletScene
from src.utils.gdino import compute_text_embeddings
from src.tests.planner import Planner
from src.utils.trajectory_utils import apply_action


# Epsilon
EPS = 1e-12


class WMPlanner(Planner):
    def __init__(
        self,
        world_model,
        enable_action_damping=True,
        enable_action_initialization=True,
        reinitialize_action_reward_threshold=0.2,
        termination_reward_threshold=0.8,
        action_init_mode: Literal["topdown", "grid", "vlm"] = "grid",
        vlm_action_initializer=None,  # action initializer from the VLM
        seed: int = 100,
    ):
        super().__init__()
        
        # initialize
        self.device = world_model.device
        self.world_model = world_model
        self.img_size = self.world_model.img_resize_shape
        self.enable_action_damping = enable_action_damping
        self.enable_action_initialization = enable_action_initialization
        self.reinitialize_action_reward_threshold = reinitialize_action_reward_threshold
        self.termination_reward_threshold = termination_reward_threshold
        
        # random-action initialization mode
        self.action_init_mode = action_init_mode.lower()
        
        # action initializer from the VLM
        self.vlm_action_initializer = vlm_action_initializer
        
        # random-number generator
        self.np_rng = np.random.default_rng()
        
        assert self.world_model.mode == "inference"
        
        # initialize the shared planning parameters
        self.init_planner()
        
        # TODO: Remove
        self.action_init_counter = 0
        self.max_action_init_count = 7

    def init_planner(self):
        # # action damping for smoothness
        # self.action_damping_activated = True
        
        # damping factors
        self.set_damping_factors()
        
    def set_damping_factors(self, rot_factor=1.05, trans_factor=1.1):
        if not hasattr(self, "action_damping_factor"):
            self.action_damping_factor = {}
        
        # update the damping factors
        self.action_damping_factor["rotation"] = rot_factor
        self.action_damping_factor["translation"] = trans_factor
        
    def apply_action_damping(self, action, pred_rewards):
        # apply a damping factor
        # rotation in axis-angle representation
        # rot_mat = R.from_euler('XYZ', action[:3]).as_matrix()
        
        rot_mat = py3d_transforms.euler_angles_to_matrix(action[:3], convention="XYZ")
        rot_axang = py3d_transforms.matrix_to_axis_angle(rot_mat)
        action[:3] = (1 / torch.exp(self.action_damping_factor["rotation"] * pred_rewards)) * rot_axang
        
        # convert to rotation matrix
        rot_mat = py3d_transforms.axis_angle_to_matrix(action[:3])
        
        # convert to Euler Angles
        action[:3] = py3d_transforms.matrix_to_euler_angles(rot_mat, convention="XYZ")
        
        # translation
        action[3:] = 1 / torch.exp(self.action_damping_factor["translation"] * pred_rewards) * action[3:]
                
        return action
        
    def compute_initial_action(
        self, 
        current_pose, 
        steps: int = 5, 
        init_mode: Literal["topdown", "grid", "vlm"] = "grid",
        init_grid_ids: Sequence[int] = np.arange(4).astype(int),
        cand_actions: Sequence[torch.Tensor] = None,
    ):
        """
        Initialize the action.
        Only supports a bird's-eye-view initilization (for now).
        """
        # TODO: Will refactor this function
        
        # action init mode
        init_mode = init_mode.lower()
    
        # current pose
        c_pose = np.asarray(current_pose)
        current_pose = torch.eye(4, device=self.device).float()
        current_pose[:3] = torch.tensor(c_pose)
        
        if init_mode == "topdown":
            # desired waypoint
            des_waypoint_rot = torch.eye(3)
            
            des_waypoint_pose = torch.eye(4, device=self.device).float()
            des_waypoint_pose[:3, :3] = des_waypoint_rot
            des_waypoint_pose[:3, -1] = current_pose[:3, -1]
            des_waypoint_pose[2, -1] = 2
            des_waypoint_pose[0, -1] = 0.55
            des_waypoint_pose[1, -1] = -0.25
            
            # cache the candidate waypoints
            cand_waypoint_poses = [des_waypoint_pose]
        elif init_mode == "grid":
            # magnitude of translation along each axis in form (low, high)
            mag_trans = np.ones((3, 2))
            mag_trans[:2, 0] *= 0.05
            mag_trans[:2, 1] *= 0.1
            
            # z-component
            mag_trans[2, 0] *= 0.05
            mag_trans[2, 1] *= 0.1
            
            # randomly sample a waypoint
            rand_waypoint = self.np_rng.uniform(low=0.0, high=1.0, size=(3,))

            # translation component
            rand_trans = mag_trans[:, 0] + rand_waypoint * (mag_trans[:, 1] - mag_trans[:, 0])
            
            # TODO: Revisit
            # # randomly select a desired waypoint
            # rand_waypoint_idx = np.random.default_rng().integers(4)
            
            # all candidate waypoints
            cand_waypoint_poses = []
                
            for cand_idx in init_grid_ids:
                # direction matrix
                trans_dir_mat = np.eye(3)
                
                # random sign for the z-direction
                trans_dir_mat[-1, -1] = np.sign(-1 + self.np_rng.uniform(size=(1,)) * 2)
                
                if cand_idx == 0:
                    # random forward-right waypoint
                    pass
                elif cand_idx == 1:
                    # random forward-left waypoint
                    # direction matrix
                    trans_dir_mat[0, 0] = -1
                elif cand_idx == 2:
                    # random bottom-left waypoint
                    # direction matrix
                    trans_dir_mat[0, 0] = -1
                    trans_dir_mat[1, 1] = -1
                elif cand_idx == 3:
                    # random bottom-right waypoint
                    # direction matrix
                    trans_dir_mat[1, 1] = -1
                else:
                    raise ValueError(
                        f"A wrong grid index was provided to the action initialization function. Received argument {init_grid_ids}!"
                    )
                    
                # compute the translation vector
                trans_vec = trans_dir_mat @ rand_trans
                    
                # desired waypoint
                des_waypoint_pose = np.eye(4)
                
                # translation component
                des_waypoint_pose[:3, -1] = c_pose[:3, -1] + trans_vec
                
                # rotation component (align the viewing direction, i.e., y-axis with the translation vector)
                des_y_dir = trans_vec
                des_waypoint_pose[:3, 1] = des_y_dir
                
                # rotation component (x-axis)
                des_x_dir = np.cross(trans_vec, np.array([0, 1, 0]))
                des_x_dir = des_x_dir / (np.linalg.norm(des_x_dir) + EPS)
                des_waypoint_pose[:3, 0] = des_x_dir
                
                # rotation component (z-axis)
                des_z_dir = np.cross(des_x_dir, des_y_dir)
                des_z_dir = des_z_dir / (np.linalg.norm(des_z_dir) + EPS)
                des_waypoint_pose[:3, 2] = des_z_dir
                
                # convert to tensor
                des_waypoint_pose = torch.tensor(des_waypoint_pose, device=self.device).float()
                
                # cache the pose
                cand_waypoint_poses.append(des_waypoint_pose)
        elif init_mode == "vlm":
            # all candidate waypoints
            cand_waypoint_poses = []
            
            for c_act in cand_actions:
                # convert to tensors
                c_act = torch.tensor(c_act, device=self.device).float()
                
                # desired waypoint
                des_waypoint_pose = torch.eye(4, device=self.device).float()
                des_waypoint_pose[:3] = apply_action(current_pose, c_act)
                
                # cache the candidate waypoints
                cand_waypoint_poses.append(des_waypoint_pose)
            
        # compute the candidate actions (include zero-action)
        cand_actions = [torch.zeros(steps, 1, 6, dtype=torch.float32, device=self.device)]
        
        for waypoint_pose in cand_waypoint_poses:
            # interpolate between the poses        
            interp_steps = torch.linspace(0, 1, steps=steps + 1).to(self.device)
            interp_rotmat = roma.rotmat_slerp(current_pose[:3, :3], des_waypoint_pose[:3, :3], steps=interp_steps)
            interp_trans = current_pose[None, :3, -1] + interp_steps[:, None] * (des_waypoint_pose[:3, -1] - current_pose[:3, -1])[None, :]
            
            # compute the actions
            actions = torch.zeros(steps, 1, 6, dtype=torch.float32, device=self.device)
            
            # previous pose
            prev_pose = current_pose
            
            for st in range(steps):
                # current pose
                curr_pose_st = torch.eye(4, device=self.device)
                curr_pose_st[:3, :3] = interp_rotmat[st]
                curr_pose_st[:3, -1] = interp_trans[st]
                
                # next pose
                next_pose_st = torch.eye(4, device=self.device)
                next_pose_st[:3, :3] = interp_rotmat[st + 1]
                next_pose_st[:3, -1] = interp_trans[st + 1]
                
                # compute the action
                act_step = torch.linalg.inv(curr_pose_st)  @ next_pose_st
                
                # extract the rotation and translation components
                act_step_rot = roma.rotmat_to_euler(convention='XYZ', rotmat=act_step[:3, :3])
                act_step_trans = act_step[:3, -1]
                
                # cache the action
                actions[st] = torch.cat((act_step_trans, act_step_rot), dim=-1).unsqueeze(0)
                
            # cache the action
            cand_actions.append(actions)
                
        return cand_actions
        
    def get_action(self,
        current_obs_img,
        target_obj_embed=None,
        curr_pose=None, # TODO included temporarily
    ):
        """
        Compute an action for a given image/observation 
        """
        # apply torchvision transforms
        current_obs = self.world_model.transform(current_obs_img).unsqueeze(0).to(self.device)
        
        # encode the image
        z = self.world_model.encoder(current_obs)
        
        # predicted rewards for the current image
        pred_reward, pred_bboxes = self.world_model.rewards_predictor(
            z,
            cond=target_obj_embed,
        )
        
        # sample from the distribution of the rewards
        pred_reward_mean, pred_reward_logvar = pred_reward[..., 0:1], pred_reward[..., 1:2]
        pred_reward_std = torch.exp(pred_reward_logvar / 2.0)
        pred_reward = pred_reward_mean + torch.randn_like(pred_reward_std) * pred_reward_std
                
        # predicted rewards for the current image
        reward_predicted = self.world_model.conf_pred_output_act(pred_reward)
        
        # # check the termination-reward criterion
        # if reward_predicted >= self.termination_reward_threshold:
        #     return torch.zeros(6,).float().to(reward_predicted.device)
        
        # initialize the actions
        if self.enable_action_initialization and reward_predicted < self.reinitialize_action_reward_threshold:
            self.action_init_counter += 1
            
            # action init mode
            init_mode = self.action_init_mode.lower()
            
            # initialize the action
            if init_mode == "vlm" and self.action_init_counter < self.max_action_init_count:
                cand_init_action = self.vlm_action_initializer(
                    obs=current_obs_img,
                    pose=curr_pose,
                )
                
                # compute a sequence of actions for each proposal
                cand_init_action = self.compute_initial_action(
                    current_pose=curr_pose,
                    init_mode=init_mode,
                    cand_actions=cand_init_action)
            else:
                cand_init_action = self.compute_initial_action(
                    current_pose=curr_pose,
                    init_mode=init_mode)
        else:
            cand_init_action = [None]
            
        # list of predicted next-step rewards per candidate action
        cand_pred_next_reward = []
        
        # list of planned trajectories
        cand_plans = []
        
        for init_action in cand_init_action:     
            # compute the action
            plan = self.world_model.joint_gradient_planner(
                z,
                target_obj_embed=target_obj_embed,
                verbose_print=False,
                init_action=init_action,
            )
            
            # cache the candidate plan
            cand_plans.append(plan)
            
            # cache the predicted next-step reward
            cand_pred_next_reward.append(plan[2].item())
            
        # identify the maximum predicted next-step reward
        plan_argmax = np.argmax(cand_pred_next_reward)
            
        # extract the relevant results from the planner
        pred_action, pred_next_z, next_reward_predicted, next_predicted_bbox, next_reward_pred_std = cand_plans[plan_argmax]
        
        if self.enable_action_damping:
            # apply a damping factor
            pred_action = self.apply_action_damping(
                action=pred_action,
                pred_rewards=reward_predicted
            )
            
        return pred_action, reward_predicted, next_reward_predicted, next_reward_pred_std
    