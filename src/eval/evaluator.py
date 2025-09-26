import json
import cv2
import os
import sys
from pathlib import Path
import numpy as np
import torch

from tqdm.auto import tqdm
from torch.nn.functional import normalize
import matplotlib.pyplot as plt

from torch.nn.functional import normalize
from src.scenes.pybullet_scene import PyBulletScene
from src.scenes.gsplat_scene import GSScene
from src.utils.gdino import compute_reward, compute_text_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def is_sequence(x):
    return isinstance(x, (list, tuple)) and not isinstance(x, np.ndarray)


class Evaluator:
    """
    Test a single planner on all provided test cases in the given experiment.
    """
    def __init__(self, 
                 planner,
                 boundaries,
                 experiment_file_path, 
                 video_outupt_dir, 
                 video_name_prefix,
                 video_name_suffix,
                 output_video=True,
                 time_limit=30,
                 speed=0.05,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        
        self.planner = planner
        self.tests = self._load_experiment(experiment_file_path)
        self.device = device
        
        # === FILE PATHS === #
        self.output_video = output_video
        self.video_output_dir = Path(video_outupt_dir)
        self.video_output_dir.mkdir(parents=True, exist_ok=True)
        self.video_name_prefix = video_name_prefix
        self.video_name_suffix = video_name_suffix
        
        # === OTHER CONSTANTS === #
        self.time_limit = time_limit
        self.speed = speed
        self.boundaries = boundaries


    def _load_experiment(self, config_file):
        """Loads test configurations from a JSON file."""
        with open(config_file, "r") as f:
            test_cases = json.load(f)
        for test in test_cases:
            test["init_pose"] = torch.tensor(test["init_pose"], dtype=torch.float32)
        return test_cases
    
    
    def load_scene(self, scene_name, scene_idx):
        """
        Load the scene and return the scene object.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    
    def in_boundary_check(self, camera_pose):
        if self.boundaries is None:
            return True
        x, y, z = camera_pose[0, -1], camera_pose[1, -1], camera_pose[2, -1]
        if self.boundaries['xmin'] <= x <= self.boundaries['xmax'] and \
           self.boundaries['ymin'] <= y <= self.boundaries['ymax'] and \
           self.boundaries['zmin'] <= z <= self.boundaries['zmax']:
            return True
        else:
            print(f"Camera pose {camera_pose} is out of the defined boundaries.")
            return False
        
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
        
        # reinitialize planner with new target for VLM Action Generator
        if type(self.planner.action_generator).__name__ == "VLMActionGenerator":
            if type(self.planner).__name__ == "SimplePlanner":
                self.planner.action_generator.initialize_conversation(self.target_object, num_proposals=1)
            else:
                self.planner.action_generator.initialize_conversation(self.target_object, num_proposals=None)
                
    
    def compute_pose_difference(self, pose_history, target_object_pose):
        """
        Compute the difference between the pose and target object.
        Args:
            pose_history: List of torch.Tensor.
        """
        # target_object_pose = torch.tensor(self.scene.object_poses[self.target_object]) # (x, y, z)

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

        return distance_to_target_history, viewing_angles_rad
    
    def evaluate(self):
        """"
        Main evaluation loop, return a list of dictionaries (len = num_tests)
        - 
        """
        all_results = []
        for test_id, test in enumerate(tqdm(self.tests, desc='Running the test configs')):
            # load test case information
            scene_name = test["environment_name"]
            scene_idx = test["scene_idx"]
            initial_pose = test["init_pose"]
            query = test["target_query"]
            target_object = test["target_object"]
            starting_confidence = test["start_confidence"]
            
            scene = self.load_scene(scene_name, scene_idx)
            
            self.set_target_object(target_object)
            
            # initialize saved info
            timestamp_history=[]
            true_reward_history = []
            reward_predicted_history = []
            next_reward_predicted_history = []
            next_reward_pred_std_history = []
            total_distance_travelled_history = []
            
            curr_pose = initial_pose
            pose_history = [curr_pose.clone().detach()]
            
            timestamp_history=[]
            bbox_history = []
            
            if self.output_video:
                self.video_shape = (scene.img_resize_shape, scene.img_resize_shape)
                video_name = f"{self.video_output_dir}/{self.video_name_prefix}_{test_id}_{self.video_name_suffix}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_name, fourcc, 10, self.video_shape)

            curr_time = 0
            total_distance_travelled = 0
            
            while curr_time < self.time_limit:
                # save time stamp
                # timestamp_history.append(curr_time)
                
                # get observation
                current_obs_img = scene.get_obs_from_pose(curr_pose, type="img")
                
                # get action list
                action_results = self.planner.get_actions(
                    current_obs_img,
                    target_obj_embed=self.target_obj_embed,
                    curr_pose=curr_pose
                )
                
                if action_results is None:
                    print("planning has finished")
                    break
                pred_action, next_reward_predicted, next_reward_pred_std = action_results
                
                if isinstance(pred_action, torch.Tensor):
                    pred_action = pred_action.tolist()
                    if not is_sequence(pred_action[0]):
                        pred_action = [pred_action]
                    
                if (not is_sequence(pred_action)):
                    pred_action = [pred_action]
                    next_reward_predicted = [next_reward_predicted]
                    next_reward_pred_std = [next_reward_pred_std]
            

                # execute the actions
                for i in range(len(pred_action)):
                    
                    in_boundary = self.in_boundary_check(curr_pose)
                    
                    # check if curr_pose is out of provided boundary
                    if not in_boundary:
                        curr_time += self.time_limit
                        break
                    
                    # get predicted action
                    a = pred_action[i]
                    
                    # get predicted reward from latent
                    if type(self.planner).__name__ == "SimplePlanner":
                        reward_predicted = -1
                    else:
                        curr_z = self.planner.compute_latent_from_observation(current_obs_img)
                        reward_predicted = self.planner.compute_reward_from_latent(curr_z, self.target_obj_embed)
                        if isinstance(reward_predicted, torch.Tensor):
                            reward_predicted = reward_predicted.item()
                    # get true reward from gdino
                    reward_true, _ = compute_reward(
                        current_obs_img,
                        query_text=f"{query}. table. random. object. background.",
                    )
                    
                    # update history
                    timestamp_history.append(curr_time)
                    true_reward_history.append(reward_true) # TODO data type?
                    reward_predicted_history.append(reward_predicted)
                    total_distance_travelled_history.append(total_distance_travelled)
                    
                    # update distance and time
                    a_np = np.array(a).flatten()    
                    distance = np.linalg.norm(a_np[:3])
                    total_distance_travelled += distance
                    time_taken = distance / self.speed
                    curr_time += time_taken
                    
                    # save video
                    if self.output_video:
                        video_frame = self.create_video_frame(current_obs_img, target_object, curr_time, reward_true, reward_predicted)
                        video_writer.write(video_frame)
                    
                    # apply the action
                    try:
                        curr_pose = scene.apply_action(curr_pose, torch.tensor(a))
                        
                    except Exception as e:
                        print(f"Error applying action: {e}")
                        print(f"Action: {a}")
                        print(f"Current Pose: {curr_pose}")
                        breakpoint()
                        
                    pose_history.append(curr_pose)
                    current_obs_img = scene.get_obs_from_pose(curr_pose, type="img")
                    
            video_writer.release()
            cv2.destroyAllWindows()
            
            # convert video
            output_video = f"{video_name}_temp.mp4"
            os.system(f"ffmpeg -hide_banner -loglevel error -y -i {video_name} -vcodec libx264 {output_video}")
            Path(video_name).unlink()
            os.rename(f"{output_video}", f"{video_name}")
        
            print(f"*** Video saved as {video_name} ***")
            
            target_object_pose = torch.tensor(scene.object_poses[target_object])
            distance_to_target_history, angle_to_target_history = self.compute_pose_difference(pose_history, target_object_pose)
            
            test_outcome = {
                "test_id": test_id,
                # "target_object": target_object,
                "timestamp_history": timestamp_history,
                "true_reward_history": true_reward_history,
                "reward_predicted_history": reward_predicted_history,
                "total_distance_travelled_history": total_distance_travelled_history,
                "distance_to_target_history": distance_to_target_history,
                "angle_to_target_history": angle_to_target_history,
                "video_name": video_name
            }
            if type(self.planner.action_generator).__name__ == "VLMActionGenerator":
                conv_history = self.planner.action_generator.response_history
                test_outcome['conversation_history'] =  conv_history
                # save conversation history
                conversation_history_path = f"{self.video_output_dir}/{self.video_name_prefix}_{test_id}_conversation_history.json"
                with open(conversation_history_path, "w") as f:
                    json.dump(conv_history, f)
                
            
            all_results.append(test_outcome)
            
            # delete the scene
            scene.delete_scene()
            del scene
            
        return all_results
            
    def create_video_frame(self, obs_img, target_object, curr_time, reward_true, reward_predicted):
        
        text_scale =  224 / 1280 * 3.5
        x_offset = int(10 * text_scale)
        y_offset = int(30 * text_scale)
        y_increment = int(50 * text_scale)
        
        frame = np.array(obs_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        cv2.putText(
            frame, 
            f"Tgt={target_object}", 
            (x_offset, y_offset), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            text_scale, (0, 255, 0), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, 
            f"t={curr_time:.2f}, GTR={reward_true:.3f}", 
            (x_offset, y_offset + y_increment), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            text_scale, (0, 255, 0), 1, cv2.LINE_AA,
        )
        cv2.putText(
            frame, 
            f"PR: {reward_predicted:.3f}", 
            (x_offset, y_offset + 2 * y_increment), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            text_scale, (0, 255, 0), 1, cv2.LINE_AA,
                        )
        return cv2.resize(frame, (self.video_shape[1], self.video_shape[0]))
        

class PybulletEvaluator(Evaluator):
    
    def load_scene(self, scene_name, scene_idx):
        """
        Load the scene and return the scene object.
        """
        # load pybullet scene
        scene_cfg_path = f"/n/fs/ap-project/GSDataGen/configs/environments/pybullet/{scene_name}.yaml"
        scene = PyBulletScene(
            device=self.device,
            img_resize_shape=224,
            pb_config_path=scene_cfg_path,
            scene_idx=scene_idx
        )
        
        return scene
    
class GaussianSplatEvaluator(Evaluator):
    
    def load_scene(self, scene_name, scene_idx):
        """
        Load the scene and return the scene object.
        """
        # load pybullet scene
        scene_cfg_path = f"/n/fs/ap-project/GSDataGen/configs/environments/gsplat/{scene_name}.yaml"
        scene = GSScene(
            device=self.device,
            img_resize_shape=224,
            gs_config_path=scene_cfg_path,
            scene_idx=scene_idx
        )
        
        return scene