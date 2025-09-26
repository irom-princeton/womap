import os
import json
import yaml
from pathlib import Path
from omegaconf import OmegaConf

import numpy as np
import torch
import torchvision.transforms.functional as TF
from scipy.spatial.transform import Rotation as R
# from pytorch3d import transforms as py3d_transforms

from PIL import Image

from src.utils.gsplat.splat_utils import *
from src.scenes.scene import Scene

class GSScene(Scene):
    def __init__(self, device, img_resize_shape, gs_config_path=None, scene_idx=0):
        super().__init__(device, img_resize_shape)
        
        
        self.cfg = OmegaConf.load(open(gs_config_path, "r"))
        
        # TODO refactor
        self.environment_name = self.cfg.environment_name
        self.environment_type = self.cfg.environment_type
        # setup camera
        self.camera_parameters = self.cfg.camera_parameters
        # load config file
        self.model_directory = self.cfg.model_paths["directory"]
        self.model_subdirectory = f"gsplat_models/{self.environment_name}_{scene_idx}/gemsplat"
        self.time_stamp = self.cfg.model_paths["time_stamps"]
        
        self.object_names = self.cfg['objects']
        
        self.gsplat_config_path = f"{self.model_directory}/{self.model_subdirectory}/{self.time_stamp}/config.yml"
        self.annotation_path = f"{self.model_directory}/{self.environment_name}_object_annotations.json"
        self.splat_align_transforms_path = f"{self.model_directory}/map_align_transforms.json"
        
        transforms = json.load(open(self.splat_align_transforms_path))
        self.splat_to_world_transform = torch.tensor(transforms[f"scene_{scene_idx}"]["splat_to_world_transform"])

        # TODO temporary
        self.object_properties = OmegaConf.load("/n/fs/ap-project/GSDataGen/assets/fwing_0_objects.yaml")
        self.height = 224
        self.width = 224
        
        self.object_positions = json.load(open(self.annotation_path))[f"scene_{scene_idx}"]["objects"]
        self.object_poses = {}

        # self.set_vertices(self.cfg.scene_boundaries.boundary_dimensions)
        self.set_objects()

        # for gsplat scenes origin is at the center of the table
        self.scene_origin = np.array([0, 0, 0])

        self.splat = SplatModel(
            config_path=Path(self.gsplat_config_path),
            res_factor=None,
            test_mode="test",  # Options: "inference", "val", "test"
            dataset_mode="test",
            device=self.device,
        )
        
        
    def _get_obs_from_pose(self, camera_pose):
        # transform to get new cam2world
        cam_to_world_4 = torch.eye(4)
        cam_to_world_4[:3] = camera_pose
        cam_to_world = torch.matmul(self.splat_to_world_transform.inverse(), cam_to_world_4)[:3]

        cam_render = Cameras(
            fx=self.camera_parameters["fl_x"],
            fy=self.camera_parameters["fl_y"],
            cx=self.camera_parameters["cx"],
            cy=self.camera_parameters["cy"],
            camera_type=torch.tensor([[1]]),
            camera_to_worlds=cam_to_world,
            height=self.camera_parameters["h"],
            width=self.camera_parameters["w"],
        )

        rendered_rgb = self.splat.render(cameras=cam_render, pose=None)["rgb"]

        rendered_rgb = rendered_rgb.permute(2, 0, 1)  # from [H, W, C] to [C, H, W]

        # Resize to 224 x 224
        img_resized = TF.resize(rendered_rgb, [self.height, self.width], interpolation=TF.InterpolationMode.BILINEAR)

        # Convert back to [H, W, C] if needed
        rendered_rgb = img_resized.permute(1, 2, 0)  # back to [H, W, C]
        
        return rendered_rgb

        # rendered_rgb = (rendered_rgb.cpu().numpy() * 255).astype(np.uint8)

        # return Image.fromarray(rendered_rgb)
    
    def set_objects(self):
        self.objects = {}
        self.obstacles = []

        for obj in self.object_names:
            x = self.object_positions[obj]["xyz"][0]
            y = self.object_positions[obj]["xyz"][1]
            z = self.object_positions[obj]["xyz"][2]
            roll = self.object_positions[obj]["rpy"][0]
            pitch = self.object_positions[obj]["rpy"][1]
            yaw = self.object_positions[obj]["rpy"][2]

            x_rad = self.object_properties[obj]["bounds"]["x"]
            y_rad = self.object_properties[obj]["bounds"]["y"]
            z_rad = self.object_properties[obj]["bounds"]["z"]

            corners_local = np.array([
                [ sx,  sy,  sz]
                for sx in (-x_rad, x_rad)
                for sy in (-y_rad, y_rad)
                for sz in (-z_rad, z_rad)
            ])

            # Rotation matrix from rpy
            rot = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

            # Transform corners to world frame
            corners_world = (rot @ corners_local.T).T + np.array([x, y, z])

            # Get AABB in world frame
            xyz_min = np.min(corners_world, axis=0)
            xyz_max = np.max(corners_world, axis=0)

            self.obstacles.append(np.concatenate([xyz_min, xyz_max]))
            # self.obstacles.append([xmin, ymin, zmin, xmax, ymax, zmax])
            self.object_poses[obj] = np.array([x, y, z])

    def delete_scene(self):
        del self.splat