from typing import Optional, Callable
import os
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import gc

from src.utils.trajectory_utils import get_action
from src.utils.gdino import compute_text_embeddings

class ActivePerceptionDataset(Dataset):
    def __init__(
        self,
        root_path,
        transform,
        img_resize_shape=224,
        confidence_bin=None,
        sequence_length=2,
        consecutive_obs=True,
        use_weighted_rewards=False,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Args:
            root_path (Path | str): Directory containing the images, poses, and confidence scores.
            transform (callable, optional): Optional transforms to be applied to each data sample.
        """
        self.root_path = root_path
        self.transform = transform
        self.sequence_length = sequence_length
        self.consecutive_obs = consecutive_obs
        self.img_resize_shape = img_resize_shape

        # load the dataset configs
        self.load_dataset_config()
        
        self.agent_poses = []
        self.sample_trajectory_lengths = []
        self.detection_confidence_scores = []
        self.bboxes = []
        
        for sid in range(self.num_scenes):
            self.scene_agent_poses = []
            self.scene_sample_trajectory_lengths = []
            self.scene_detection_confidence_scores = []
            self.scene_bboxes = []
            for tid in range(self.num_trajectories_per_scene):
                # effective lengths of each trajectory for smapling (accountign for the sequence length)
                self.scene_sample_trajectory_lengths.append(self.trajectory_lengths_per_scene[sid][tid] - self.sequence_length + 1)
                
                # load the poses
                poses = torch.tensor(
                    np.load(f"{self.root_path}/scene{sid}/traj{tid}/cam_to_worlds.npy"), device=device
                ).float().cpu()
                self.scene_agent_poses.append(poses)
                
                if use_weighted_rewards:
                    confidence_score_path = f"{self.root_path}/scene{sid}/traj{tid}/weighted_confidence_scores.npy"
                else:
                    confidence_score_path = f"{self.root_path}/scene{sid}/traj{tid}/raw_confidence_scores.npy"
                
                confidence_scores = torch.tensor(np.load(confidence_score_path), device=device).float().cpu()[..., :len(self.target_objects)]
                
                if confidence_bin is not None:
                    # bin confidence scores
                    confidence_scores = (
                        torch.round(confidence_scores * (0.1 / confidence_bin),
                                    decimals=1)
                        / (0.1 / confidence_bin)
                    )
                    
                self.scene_detection_confidence_scores.append(confidence_scores)
            
                # bounding boxes
                bboxes = torch.tensor(
                    np.load(f"{self.root_path}/scene{sid}/traj{tid}/bounding_boxes.npy"), device=device
                ).float().cpu()[..., :len(self.target_objects), :]
                self.scene_bboxes.append(bboxes)
                
            
            self.agent_poses.append(self.scene_agent_poses)
            self.sample_trajectory_lengths.append(self.scene_sample_trajectory_lengths)
            self.detection_confidence_scores.append(self.scene_detection_confidence_scores)
            self.bboxes.append(self.scene_bboxes)

    def load_dataset_config(self):
        dataset_config_path = f"{self.root_path}/dataset_config.yaml"
        dataset_info_path = f"{self.root_path}/dataset_info.yaml" 
        with open(dataset_info_path, "r") as y_file:
            dataset_info = yaml.load(y_file, Loader=yaml.FullLoader)
            self.num_scenes = dataset_info["data"]["num_scenes"]
            self.num_trajectories_per_scene = dataset_info["data"]["num_trajectories_per_scene"]
            
            # target objects for reward
            self.target_objects = dataset_info["reward"]["all_targets"]
            self.target_objects_embeds = compute_text_embeddings(self.target_objects)
            
        with open(dataset_config_path, "r") as y_file:
            dataset_config = yaml.load(y_file, Loader=yaml.FullLoader)
            self.num_poses = dataset_config["num_poses"]
            self.trajectory_lengths_per_scene = dataset_config["trajectory_lengths"]
    
    def _index_to_trajectory(self, idx):
        # find the trajectory index
        scene_idx = 0
        traj_idx = 0
        while idx >= self.sample_trajectory_lengths[scene_idx][traj_idx]:
            idx -= self.sample_trajectory_lengths[scene_idx][traj_idx]
            traj_idx += 1
            if traj_idx == self.num_trajectories_per_scene:
                scene_idx += 1
                traj_idx = 0
        return scene_idx, traj_idx, idx
    
    def __len__(self):
        return self.num_poses - self.num_trajectories_per_scene * self.num_scenes * (self.sequence_length - 1)
    
    def __getitem__(self, idx):
        sid, tid, oid = self._index_to_trajectory(idx) # parse to trajectory and observation index

        if self.consecutive_obs:
            observation_idx_sequence = np.arange(oid, oid + self.sequence_length)
        else:
            observation_idx_sequence = np.random.randint(oid, oid + self.sequence_length, size=self.sequence_length)
        
        observation_sequence = torch.zeros(self.sequence_length, 3, self.img_resize_shape, self.img_resize_shape)
        computed_actions = np.zeros((self.sequence_length - 1, 6))
        
        computed_scores = np.zeros((self.sequence_length, len(self.target_objects)))
        bboxes = np.zeros((self.sequence_length, len(self.target_objects), 4))

        for i in range(self.sequence_length):
            obs_id = observation_idx_sequence[i]
            image = Image.open(f"{self.root_path}/scene{sid}/traj{tid}/{obs_id:06d}.png")
            image = self.transform(image)
            observation_sequence[i] = image
            
            # bounding box
            bboxes[i] = self.bboxes[sid][tid][obs_id]

            if i < self.sequence_length - 1:
                obs_id_next = observation_idx_sequence[i+1]
                computed_actions[i] = get_action(self.agent_poses[sid][tid][obs_id], self.agent_poses[sid][tid][obs_id_next])
            
            computed_scores[i] = self.detection_confidence_scores[sid][tid][obs_id]
        
        sample = {
            "observation_sequence": observation_sequence,
            "actions": computed_actions,
            "detection_confidence_scores": computed_scores,
            "bboxes": bboxes,
        }
        return sample