import os
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
# from pytorch3d import transforms as py3d_transforms
from scipy.spatial.transform import Rotation as R

from PIL import Image
from abc import ABC, abstractmethod

from src.utils.gsplat.splat_utils import *
import src.utils.trajectory_utils as traj_utils


class Scene(ABC):
    def __init__(self, device, img_resize_shape):
        self.device = device
        self.img_resize_shape = img_resize_shape

    def reshape(self, obs):
        tensor = obs.permute(2, 0, 1)  # Convert to [C, H, W]
        resized_tensor = TF.resize(tensor, (self.img_resize_shape, self.img_resize_shape))
        resized_tensor = resized_tensor.permute(1, 2, 0)
        return resized_tensor

    @abstractmethod
    def _get_obs_from_pose(self, camera_pose):
        pass
    
    def get_obs_from_pose(self, camera_pose, resize=True, type='img'):
        """
        Args:
            camera_pose (array-like): The (3x4) view matrix
            resize (bool): Whether to resize the image or not
            type (str): The type of output: 'img', 'tensor', or 'numpy'.
        """
        # case-insensitive type
        type = type.lower()
        # if pose is not tensor
        if not torch.is_tensor(camera_pose):
            camera_pose = torch.tensor(camera_pose)
            
        obs_rgb = self._get_obs_from_pose(camera_pose)
        
        if resize:
            obs_rgb = self.reshape(obs_rgb)
        if type == "numpy" or type == "np":
            return obs_rgb.cpu().numpy()
        elif type == "tensor":
            return obs_rgb
        elif type == "img":
            obs_rgb = (obs_rgb.cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(obs_rgb)
        raise ValueError(f"Invalid type: {type}")


    def apply_action(self, prev_cam2world, action):
        new_cam2world = traj_utils.apply_action(prev_cam2world, action)

        return new_cam2world
    
    
    def get_action(self, prev_cam2world, current_cam2world):
        action = traj_utils.get_action(prev_cam2world, current_cam2world)
        return action
    