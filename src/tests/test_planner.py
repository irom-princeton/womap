from typing import List
import yaml
import cv2
import numpy as np
import torch
from torchvision import transforms as tv_transforms
from pathlib import Path
import os
import json
from tqdm.auto import tqdm
# from pytorch3d import transforms as py3d_transforms
from torch.nn.functional import normalize
import matplotlib.pyplot as plt
import roma

# from src.models.gsplatmodel import GSScene # TODO
from src.scenes.pybullet_scene import PyBulletScene
from src.utils.gdino import compute_reward, compute_text_embeddings


# Epsilon
EPS = 1e-12


class WMTestPlanner:
    def __init__(
        self,
        scene,
        world_model,
        output_dir,
        trajectory_dir="/n/fs/ap-project/active_perception_world_models/test_inputs/trajectories",        
        target_object=None,
        initialize_action=False,
        initialize_action_reward_threshold=0.1,
    ):
        self.scene = scene
        self.device = world_model.device
        self.world_model = world_model
        self.img_size = self.world_model.img_resize_shape
        self.initialize_action = initialize_action
        self.initialize_action_reward_threshold = initialize_action_reward_threshold
        
        assert self.world_model.mode == "inference"
        self.output_dir = output_dir
        # TODO hard code
        # self.trajectory_dir = "/n/fs/ap-project/active_perception_world_models/test_inputs/trajectories"
        self.trajectory_dir = trajectory_dir
        
        # set the target object
        self.set_target_object(target_object)
        
        # initialize the shared planning parameters
        self.init_planner()
                
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
            
    def init_planner(self):
        # action damping for smoothness
        self.action_damping_activated = False
        
        # damping factors
        self.set_damping_factors()
        
    def set_damping_factors(self, rot_factor=1.05, trans_factor=1.1):
        if not hasattr(self, "action_damping_factor"):
            self.action_damping_factor = {}
        
        # update the damping factors
        self.action_damping_factor["rotation"] = rot_factor
        self.action_damping_factor["translation"] = trans_factor
        
    # def apply_action_damping(self, action, pred_rewards):
    #     # apply a damping factor
    #     # rotation in axis-angle representation
    #     rot_mat = py3d_transforms.euler_angles_to_matrix(action[:3], convention="XYZ")
    #     rot_axang = py3d_transforms.matrix_to_axis_angle(rot_mat)
    #     action[:3] = (1 / torch.exp(self.action_damping_factor["rotation"] * pred_rewards)) * rot_axang
        
    #     # convert to rotation matrix
    #     rot_mat = py3d_transforms.axis_angle_to_matrix(action[:3])
        
    #     # convert to Euler Angles
    #     action[:3] = py3d_transforms.matrix_to_euler_angles(rot_mat, convention="XYZ")
        
    #     # translation
    #     action[3:] = 1 / torch.exp(self.action_damping_factor["translation"] * pred_rewards) * action[3:]
                
    #     return action
        
    def compute_initial_action(self, current_pose, steps: int = 5):
        """
        Initialize the action.
        Only supports a bird's-eye-view initilization (for now).
        """
        
        # current pose
        c_pose = current_pose
        current_pose = torch.eye(4, device=self.device).float()
        current_pose[:3] = torch.tensor(np.asarray(c_pose))
        
        # TODO: Refactor
        des_waypoint_rot = torch.eye(3)
        
        des_waypoint_pose = torch.eye(4, device=self.device).float()
        des_waypoint_pose[:3, :3] = des_waypoint_rot
        des_waypoint_pose[:3, -1] = current_pose[:3, -1]
        des_waypoint_pose[2, -1] = 2
        des_waypoint_pose[0, -1] = 0.55
        des_waypoint_pose[1, -1] = -0.25
        
        # compute the actions
        actions = torch.zeros(steps, 1, 6, dtype=torch.float32, device=self.device)
        
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
            act_step = torch.linalg.inv(next_pose_st) @ curr_pose_st
            
            # extract the rotation and translation components
            act_step_rot = roma.rotmat_to_euler(convention='XYZ', rotmat=act_step[:3, :3])
            act_step_trans = act_step[:3, -1]
            
            # cache the action
            actions[st] = torch.cat((act_step_rot, act_step_trans), dim=-1).unsqueeze(0)
            
        return actions
        
    def compute_total_distance(self, pose_history):
        """
        Compute the total Euclidean distance traveled by the agent.
        """
        total_distance = 0.0
        for i in range(1, len(pose_history)):
            prev_pose = np.array(pose_history[i - 1])
            curr_pose = np.array(pose_history[i])
            distance = np.linalg.norm(curr_pose[:3] - prev_pose[:3])  # TODO: check that pose[:3] is x, y, z
            total_distance += distance
        return total_distance
    
    def compute_pose_difference(self, pose_history):
        """
        Compute the difference between the pose and target object.
        """
        target_object_pose = torch.tensor(self.scene.object_poses[self.target_object]) # (x, y, z)
        # target_object_ori = torch.tensor(self.scene.object_positions[self.target_object]["orientation"]) # Euler angles
        # target_object_rot_mat = py3d_transforms.euler_angles_to_matrix(target_object_ori, convention="XYZ")
        
        # pose_history = torch.tensor(np.array(pose_history))
        pose_history = torch.stack(pose_history) 
        T = pose_history.shape[0]

        camera_positions = pose_history[:, :, 3] # (T, 3)
        cam_to_obj = target_object_pose[None, :] - camera_positions # (T, 3)
        distance_to_target_history = torch.norm(cam_to_obj, dim=1) # (T,)

        camera_rot_mats = pose_history[:, :, :3]  # (T, 3, 3)
        camera_forward_dirs = -camera_rot_mats[:, :, 2]  # (T, 3), negative Z column

        cam_to_obj_normalized = normalize(cam_to_obj, dim=1)
        camera_forward_dirs_normalized = normalize(camera_forward_dirs, dim=1)

        cos_angles = torch.sum(cam_to_obj_normalized * camera_forward_dirs_normalized, dim=1).clamp(-1.0, 1.0)
        viewing_angles_rad = torch.acos(cos_angles)  # (T,)
        # viewing_angles_deg = torch.rad2deg(viewing_angles_rad) # optional
        
        return distance_to_target_history, viewing_angles_rad

    def compute_weighted_step(self, bbox_history):
        """
        Compute the weighted bounding box for the object.
        """
        # TODO: is this what we want?
        bbox_history = np.array(bbox_history)
        bbox_areas = (bbox_history[:, 2] - bbox_history[:, 0]) * (bbox_history[:, 3] - bbox_history[:, 1]) # (x_min, y_min, x_max, y_max)
        steps = np.arange(1, len(bbox_areas) + 1)
        weighted_step = steps / (bbox_areas + EPS)
        return weighted_step
    
    def display_and_save_metrics(self, metrics):
        """
        Display and save the metrics.
        """
        # display the metrics
        avg_distance = np.mean([v["total_distance"] for v in metrics.values()])
        avg_weighted_step = np.mean([v["weighted_step"][-1] for v in metrics.values()])

        print(f"Average Total Distance: {avg_distance}")
        print(f"Average Weighted Step: {avg_weighted_step}")
        
        class NumpyTorchEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, torch.Tensor):
                    return obj.item() if obj.ndim == 0 else obj.tolist()
                elif isinstance(obj, np.ndarray):
                    return obj.item() if obj.ndim == 0 else obj.tolist()
                elif isinstance(obj, (np.float32, np.float64, np.float16)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64, np.int16)):
                    return int(obj)
                return super().default(obj)
        
        # save the metrics
        with open(f"{self.output_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, cls=NumpyTorchEncoder, indent=4)
            print(f"*** Metrics saved as {self.output_dir}/metrics.json ***")

        # plot the metrics
        for pose_index, metric in metrics.items():
            fig, ax = plt.subplots(3, 1, figsize=(10, 10))
            fig.suptitle(f"Metrics for Pose {pose_index}")
            ax[0].plot(metric["true_reward_history"], label="True Reward")
            ax[0].plot(metric["pred_reward_history"], label="Predicted Reward")
            ax[0].set_title("Reward History")
            ax[0].legend()

            ax[1].plot(metric["distance_to_target_hitstory"], label="Distance to Target")
            ax[1].set_title("Distance to Target")

            ax[2].plot(metric["angle_to_target_history"], label="Angle to Target")
            ax[2].set_title("Angle to Target")
            
            for axis in ax:
                axis.grid()

            plt.savefig(f"{self.output_dir}/metrics_{pose_index}.png")
            print(f"*** Metrics saved as {self.output_dir}/metrics_{pose_index}.png ***")
        
    def eval_planner(
        self,
        initial_pose_list: List,
        run_name="eval",
        num_steps=200,
        enable_action_damping: bool = False,
    ):
        """
        Evaluate the Planners for the World Model.
        """
        if enable_action_damping and not hasattr(self, "action_damping_factor"):
            self.set_damping_factors()
        
        metrics = {}

        for pose_index, pose in enumerate(tqdm(initial_pose_list, desc="Planning")):
            # current pose
            curr_pose = pose

            # trajectory evaluation
            pose_history = [torch.tensor(curr_pose)]
            true_reward_history = []
            pred_reward_history = []
            bbox_history = []

            # set up planning output
            video_name = Path(f"{self.output_dir}/{run_name}_{pose_index}.mp4")

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
            
            # # set up initial latent
            # current_obs = scene.get_obs_from_pose(curr_pose, type="img")
            # init_obs = self.world_model.transform(current_obs).unsqueeze(0).to(device)
            # z = self.world_model.encoder(init_obs)
            
            # initialize the predicted reward
            next_reward_predicted = "N/A"
            next_reward_pred_std = "N/A"

            for step in tqdm(range(num_steps), desc=f"Planning with initialization {pose_index}"):
                # get current image
                current_obs_img = self.scene.get_obs_from_pose(curr_pose, type="img")
                # current_obs_img.save(f"{img_output_dir}/{step:06d}.png")
                
                reward_true, bbox_true = compute_reward(
                    current_obs_img,
                    query_text=f"{self.target_object}. table. random. object. background.",
                ) # only gdino is supported

                true_reward_history.append(reward_true)
                bbox_history.append(bbox_true)

                # model
                current_obs = self.world_model.transform(current_obs_img).unsqueeze(0).to(self.device)
                
                # encode the image
                z = self.world_model.encoder(current_obs)
                
                if step == 0:
                    # initialize the predicted next latent state
                    pred_next_z = z.detach()
                
                # decode the predicted latent state at the current timestep
                with torch.no_grad():
                    z_decode, _ = self.world_model.decoder(pred_next_z.unsqueeze(1))
                    
                    # apply the inverse image transfomrs
                    z_decode = self.world_model.inv_transform(z_decode)
                    z_decode = torch.clamp(
                        z_decode,
                        min=0,
                        max=1,
                    )
                    
                    # apply the inverse image transform to the observed RGB
                    obs_img = self.world_model.inv_transform(current_obs).to(self.device)
                    obs_img = torch.clamp(
                        obs_img,
                        min=0,
                        max=1,
                    )
                    
                # latent state from the observed image
                obs_latent = self.world_model.encoder(current_obs)
                
                # compute the error in the predicted and observed latent states
                latent_diff = torch.mean(torch.abs(obs_latent - pred_next_z))
                print(f"Step={step}, Latent Difference={latent_diff.item()}")

                # predicted rewards for the current image
                pred_reward, pred_bboxes = self.world_model.rewards_predictor(
                    z,
                    cond=self.target_obj_embed,
                )
                
                # sample from the distribution of the rewards
                pred_reward_mean, pred_reward_logvar = pred_reward[..., 0:1], pred_reward[..., 1:2]
                pred_reward_std = torch.exp(pred_reward_logvar / 2.0)
                pred_reward = pred_reward_mean + torch.randn_like(pred_reward_std) * pred_reward_std
                        
                reward_predicted = self.world_model.conf_pred_output_act(pred_reward)
                pred_reward_history.append(reward_predicted.item())
                
                if pred_bboxes is not None:
                    # apply the sigmoid function to the predicted bounding boxes
                    pred_bboxes = self.world_model.conf_pred_output_act(pred_bboxes)
                    pred_bboxes = pred_bboxes.squeeze()
                    
                if step == 0:
                    next_predicted_bbox = pred_bboxes
        
                # reward_predicted = self.world_model.conf_pred_output_act(reward_raw)

                # annotate the images
                # ground-truth observation
                obs_img = (
                    obs_img.squeeze()
                    .moveaxis(-3, -1)
                    .detach()
                    .cpu()
                    .numpy() * 255  
                ).astype(np.uint8)
                
                # RGB-to-BGR
                cv2_obs_img = cv2.cvtColor(obs_img, cv2.COLOR_RGB2BGR)
                
                # annotate with bounding box
                cv2.rectangle(
                    cv2_obs_img,
                    (int(bbox_true[0]),
                    int(bbox_true[1])),
                    (int(bbox_true[2]),
                    int(bbox_true[3])),
                    (255, 0, 0),
                    2
                )
                
                if pred_bboxes is not None:
                    # rescale
                    pred_bboxes = pred_bboxes * self.img_size
                    
                    # predicted at the current frame
                    cv2.rectangle(
                        cv2_obs_img,
                        (int(pred_bboxes[0]),
                        int(pred_bboxes[1])),
                        (int(pred_bboxes[2]),
                        int(pred_bboxes[3])),
                        (0, 0, 255),
                        2
                    )
                
                # predicted observation
                z_decode = (
                    z_decode.squeeze()
                    .moveaxis(-3, -1)
                    .detach()
                    .cpu()
                    .numpy() * 255  
                ).astype(np.uint8)
                
                # RGB-to-BGR
                cv2_decode_img = cv2.cvtColor(z_decode, cv2.COLOR_RGB2BGR)
                
                if next_predicted_bbox is not None:
                    # rescale
                    next_predicted_bbox = next_predicted_bbox * self.img_size
                    
                    # annotate with bounding box
                    cv2.rectangle(
                        cv2_decode_img,
                        (int(next_predicted_bbox[0]),
                        int(next_predicted_bbox[1])),
                        (int(next_predicted_bbox[2]),
                        int(next_predicted_bbox[3])),
                        (255, 0, 0),
                        2
                    )
                  
                # combined images (observed image, predicted image in the Width axis)
                frame = (
                    np.concatenate(
                        (
                            cv2_obs_img,
                            cv2_decode_img,
                        ),
                        axis=-2,
                    )
                )
                
                # # annotate image
                # frame = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
                # frame = combined_img

                cv2.putText(
                    frame, 
                    f"Step={step}", 
                    (x_offset, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), text_thickness
                    )
                next_reward_pred = f"{next_reward_predicted.item():.3f}" if step > 0 else next_reward_predicted
                next_reward_pred_std = f"{next_reward_pred_std.item():.3f}" if step > 0 else next_reward_pred_std
                cv2.putText(
                    frame,
                    f"Pred Rewards at ({step - 1}) = {next_reward_pred}", 
                    (x_offset, y_offset + y_increment), 
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), text_thickness
                    )
                cv2.putText(
                    frame, 
                    f"Pred Rewards at ({step}) = {reward_predicted.item():.3f}", 
                    (x_offset, y_offset + 2 * y_increment), 
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), text_thickness
                    )
                cv2.putText(
                    frame, 
                    f"GT Rewards at ({step})={reward_true:.3f}", 
                    (x_offset, y_offset + 3 * y_increment), 
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), 1
                    )
                
                # convert to unit8
                # frame = frame.astype(np.uint8)
                video_writer.write(frame)
                
                # initialize the actions
                if self.initialize_action and reward_predicted < self.initialize_action_reward_threshold:
                    init_action = self.compute_initial_action(current_pose=curr_pose)
                else:
                    init_action = None
                
                # compute the action
                pred_action, pred_next_z, next_reward_predicted, next_predicted_bbox, next_reward_pred_std = self.world_model.joint_gradient_planner(
                    z,
                    target_obj_embed=self.target_obj_embed,
                    verbose_print=False,
                    init_action=init_action,
                )
                # pred_action, pred_next_z, next_reward_predicted = self.world_model.simple_gradient_planner2(z)
                # pred_action, pred_next_z, next_reward_predicted = self.world_model.mpc_gradient_planner(z)
        
                # if enable_action_damping:
                #     # apply a damping factor
                #     pred_action = self.apply_action_damping(
                #         action=pred_action,
                #         pred_rewards=reward_predicted
                #     )
                    
                # update pose
                curr_pose = self.scene.apply_action(curr_pose, pred_action)
                pose_history.append(curr_pose)
                
            video_writer.release()
            cv2.destroyAllWindows()
            # print(f"*** Video saved as {output_video} ***")

            # TODO: Remove (just local video playback issues)
            # output path
            output_video = f"{video_name}_temp.mp4"
            
            # encode with H264 codec
            os.system(f"ffmpeg -hide_banner -loglevel error -y -i {video_name} -vcodec libx264 {output_video}")

            # delete the temporary file
            Path(video_name).unlink()
            os.rename(f"{output_video}", f"{video_name}")
        
            print(f"*** Video saved as {video_name} ***")

            # compute the stats
        #     distance_to_target_history, angle_to_target_history = self.compute_pose_difference(pose_history)

        #     metrics[pose_index] = {
        #         "pose_history": pose_history,
        #         "bbox_history": bbox_history,
        #         "true_reward_history": true_reward_history,
        #         "pred_reward_history": pred_reward_history,
        #         "distance_to_target_hitstory": distance_to_target_history,
        #         "angle_to_target_history": angle_to_target_history,
        #         "total_distance": self.compute_total_distance(pose_history),
        #         "weighted_step": self.compute_weighted_step(bbox_history),
        #     }
          
        # # display and save the metrics
        # self.display_and_save_metrics(metrics)
