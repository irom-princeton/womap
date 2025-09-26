import os
import json
import yaml
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF

import pybullet as p
import pybullet_data

from PIL import Image
from omegaconf import OmegaConf

from src.utils.gsplat.splat_utils import *
from src.scenes.scene import Scene

OBJECT_PROPERTIES = OmegaConf.load("/n/fs/ap-project/GSDataGen/assets/tabletop_objects.yaml")

class PyBulletScene(Scene):
    def __init__(self, device, img_resize_shape, pb_config_path=None, scene_idx=0):
        super().__init__(device, img_resize_shape)
        # /n/fs/ap-project/GSDataGen/configs/scene/pybullet/table_simple.yaml
        # /n/fs/ap-project/GSDataGen/configs/scene/pybullet/table_scissor_banana.yaml
        
        self.cfg = yaml.load(open(pb_config_path, "r"), Loader=yaml.FullLoader)
        
        physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-10)
        
        self.plane = p.loadURDF("plane.urdf")

        # setup table
        self.table = p.loadURDF(
            self.cfg['table']['model_path'],
            self.cfg['table']['center'],
            p.getQuaternionFromEuler([0,0,np.pi/2]),
            globalScaling=self.cfg['table']['scale'],
            useFixedBase=True,
        )

        expand_scale = 2.0

        [cx, cy, cz] = self.cfg['table']['center'] # TODO table z not used
        self.tableHeight = (.6 + .05/2) * self.cfg['table']['scale']
        zmin = self.tableHeight + 0.02

        zmax = zmin + 0.5 * expand_scale
        
        self.table_width = .6 * self.cfg['table']['scale']
        self.table_length = 1.2 * self.cfg['table']['scale']
        self.table_start_x = cx - self.table_width / 2
        self.table_start_y = cy + self.table_length / 2

        # setup object models
        self.object_names = self.cfg['objects']
        
        # TODO update with omegaconf
        env_name = self.cfg['environment_name']
        
        scene_path = f"/n/fs/ap-project/data/wm_scenes/pybullet/{env_name}/scene{scene_idx}.json"
        self.object_positions = json.load(open(scene_path))["objects"]

        self.set_camera()
        self.set_objects()

    def set_camera(self):
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.cfg['camera']['fov'],
            aspect=self.cfg['camera']['aspect'],
            nearVal=self.cfg['camera']['near'],
            farVal=self.cfg['camera']['far'],
        )
        self.width = self.img_resize_shape
        self.height = self.img_resize_shape

    def cam2world_to_viewmat(self, cam2world):
        """
        cam2world: 
        """
        R_mat = cam2world[:, :3]
        T_mat = cam2world[:, 3]
        cameraEyePosition = T_mat
        cameraTargetPosition = T_mat - R_mat[:, 2]
        cameraUpVector = R_mat[:, 1]
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cameraEyePosition,
            cameraTargetPosition=cameraTargetPosition,
            cameraUpVector=cameraUpVector,
        )
        return view_matrix
    
    def _get_obs_from_pose(self, cam2world):
        view_matrix = self.cam2world_to_viewmat(cam2world)
        img = p.getCameraImage(self.img_resize_shape, self.img_resize_shape, 
                               view_matrix, 
                               self.projection_matrix, 
                               renderer=p.ER_BULLET_HARDWARE_OPENGL)
        img_rgb = np.reshape(img[2], (self.width, self.height, 4))
        img_rgb = img_rgb[:, :, :3] # remove alpha channel
        img_rgb = img_rgb / 255
        return torch.tensor(img_rgb.copy()).float()

    def set_objects(self):
        self.objects = {}
        self.object_poses = {}
        for obj in self.object_names:
            absx = self.table_start_x + self.object_positions[obj]["pose"][0] 
            absy = self.table_start_y - self.object_positions[obj]["pose"][1]
            absz = self.tableHeight + OBJECT_PROPERTIES[obj]["offset"]
            ori = self.object_positions[obj]["orientation"]
            
            self.object_poses[obj] = np.array([absx, absy, absz])
            
            path_to_urdf = Path(OBJECT_PROPERTIES[obj]["model_path"]) / "model.urdf"
            self.objects[obj] = p.loadURDF(
                str(path_to_urdf),
                self.object_poses[obj] ,
                p.getQuaternionFromEuler(ori)
            )
            
    def delete_scene(self):
        for obj in self.objects.values():
            p.removeBody(obj)
        p.removeBody(self.plane)
        p.removeBody(self.table)
        p.disconnect()