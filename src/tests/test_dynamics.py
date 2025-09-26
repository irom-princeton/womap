from typing import List
import yaml
import cv2
import numpy as np
import torch
from torchvision import transforms as tv_transforms
from pathlib import Path
import os
from tqdm import tqdm
from collections import deque

from src.utils.rewards import compute_text_embeddings


class WMTestDynamics:
    def __init__(
        self,
        scene,
        world_model,
        output_dir,
        trajectory_dir="/n/fs/ap-project/active_perception_world_models/test_inputs/trajectories",
        target_object=None,
    ):
        self.scene = scene
        self.device = world_model.device
        self.world_model = world_model
        self.img_size = self.world_model.img_resize_shape
        
        assert self.world_model.mode == "inference"
        self.output_dir = output_dir
        # TODO hard code
        # self.trajectory_dir = "/n/fs/ap-project/active_perception_world_models/test_inputs/trajectories"
        self.trajectory_dir = trajectory_dir
        
        # set the target object
        self.set_target_object(target_object)

    def set_target_object(self, target_object):
        """
        Compute the text embeddings for the target object
        """
        # set the target object
        self.target_object = target_object
        
        if self.target_object is not None:
            # compute the text embeddings
            self.target_obj_embed = compute_text_embeddings(text=[self.target_object]).unsqueeze(0).to(self.device)
        else:
            self.target_obj_embed = None
            
    def generate_random_actions(self,
                                horizon=10,
                                rot_range=0.01,
                                trans_range=0.05,
                                ):
        unnormalized_action = np.random.randn(horizon, 6)
        # angular scale [-0.01, 0.01]
        unnormalized_action[:, :3] = unnormalized_action[:, :3] * 2 * rot_range - rot_range
        # linear scale [-0.05, 0.05]
        unnormalized_action[:, 3:] = unnormalized_action[:, 3:] * 2 * trans_range - trans_range
        return unnormalized_action
    
    def dynamics_consistency_test_traj(self, trajectory_fnames: List):
        """
        Rollout the dynamics model on a trajectory and save the video
        """
        
        for trajectory_name in tqdm(trajectory_fnames, desc='Evaluating trajectories'):
            # add 5-digit random number to the video name
            video_name = Path(f"{self.output_dir}/dyn-consistency-{os.path.splitext(Path(trajectory_name).stem)[0]}.mp4")
            
            # create directory, if necessary
            video_name.parent.mkdir(parents=True, exist_ok=True)
            
            # convert to string
            video_name = str(video_name)
            
            # video shape (ground truth and prediction side-by-side)
            video_shape = (self.img_size * 2, self.img_size)
            
            # set up video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_name, fourcc, 10, video_shape)
            
            # text scale, thickness, and placement
            text_scale =  self.world_model.img_resize_shape / 1280 * 4
            text_thickness = 2
            x_offset = int(10 * text_scale)
            y_offset = int(30 * text_scale)
            y_increment = int(50 * text_scale)

            # load trajectory
            cam_pose_list = np.load(trajectory_name)
            num_frames = cam_pose_list.shape[0]

            # set up initial latent
            current_obs = self.scene.get_obs_from_pose(cam_pose_list[0], type="img")
            init_obs = self.world_model.transform(current_obs).unsqueeze(0).to(self.device)
            z = self.world_model.encoder(init_obs)
            
            # TODO: We can improve how we incorporate history.
            # initialize the latent state history
            z_hist = deque([z] * (self.world_model.latent_state_history_length - 1))
                     
            for i in range(num_frames):
                # get observation (not required at iteration 0)
                obs = self.scene.get_obs_from_pose(cam_pose_list[i], type="img")
                
                # decode the predicted latent state
                with torch.no_grad():
                    z_decode, _ = self.world_model.decoder(z.unsqueeze(1)) # TODO doesn't work
                    
                    # apply the inverse image transfomrs
                    z_decode = self.world_model.inv_transform(z_decode)
                    z_decode = torch.clamp(
                        z_decode,
                        min=0,
                        max=1,
                    )
                    
                    # apply the inverse image transform to the observed RGB
                    obs_img = self.world_model.inv_transform(self.world_model.transform(obs)).to(self.device)
                    obs_img = torch.clamp(
                        obs_img,
                        min=0,
                        max=1,
                    )
                    
                # latent state from the observed image
                obs_latent = self.world_model.encoder(
                    self.world_model.transform(obs).unsqueeze(0).to(self.device)
                )
                
                # compute the error in the predicted and observed latent states
                latent_diff = torch.mean(torch.abs(obs_latent - z))
                print(f"Step={i}, Latent Difference={latent_diff.item()}")

                # combined images (observed image, predicted image in the Width axis)
                img = (torch.cat((obs_img.squeeze(), z_decode.squeeze()), dim=-1)
                                    .moveaxis(-3, -1)
                                    .detach()
                                    .cpu()
                                    .numpy() * 255).astype(np.uint8)
                
                # convert color to 
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # annotate the image
                cv2.putText(
                    img, 
                    f"Step={i}", 
                    (x_offset, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), text_thickness
                    )
                cv2.putText(
                    img, 
                    f"L1 Loss={latent_diff.item():.3f}", 
                    (x_offset, y_offset + y_increment), 
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), text_thickness
                    )
                
                # write the image
                video_writer.write(img)

                if i < num_frames - 1:
                    # compute the action
                    action = self.scene.get_action(cam_pose_list[i], cam_pose_list[i+1])
                    action_tensor = torch.tensor(action).to(self.device)
                    
                    # latent state with history
                    z_curr_with_history = torch.cat(
                        (z, *z_hist),
                        dim=-2,
                    )
                            
                    # predict the next latent state
                    z = self.world_model.dynamics_predictor(z_curr_with_history, action_tensor.unsqueeze(0).unsqueeze(0))[..., :z.shape[-2], :]
                    z_mu, z_logvar = z[..., :self.world_model.dynamics_predictor.input_output_embed_dim], z[..., self.world_model.dynamics_predictor.input_output_embed_dim:]
              
                    # sample from the distribution of the next latent state
                    z_std = torch.exp(z_logvar / 2.0)
                    z = z_mu + torch.randn_like(z_std) * z_std

                    # update the latent state history
                    if self.world_model.latent_state_history_length > 1:
                        z_hist.pop() # remove from right
                        z_hist.appendleft(z) # append to left
                            
            video_writer.release()
            cv2.destroyAllWindows()
            
            # TODO: Remove (just local video playback issues)
            # output path
            output_video = f"{video_name}_temp.mp4"
            
            # encode with H264 codec
            os.system(f"ffmpeg -hide_banner -loglevel error -y -i {video_name} -vcodec libx264 {output_video}")

            # delete the temporary file
            Path(video_name).unlink()
            os.rename(f"{output_video}", f"{video_name}")
        
            print(f"Video saved as {video_name}")

    def rollout_test(self, init_position):
        """
        Rollout the dynamics model at given initial position and save the video
        """
        pass
    