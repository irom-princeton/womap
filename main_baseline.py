# visualize planning with vlm

import argparse
import torch.multiprocessing as mp

import pprint
import os
import sys
import yaml
from pathlib import Path
from PIL import Image
import cv2

import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate

import json

from glob import glob

import numpy as np
import torch

from src.models.worldmodel import WorldModel
from src.models.gsplatmodel import GSScene # TODO
from src.scenes.pybullet_scene import PyBulletScene
from src.tests.test_planner import WMTestPlanner
from src.tests.vlm_planner import VLMPlanner
from src.utils.gdino import compute_reward
from src.utils.wm_utils import load_wm_for_inference

sys.path.append(f"{Path(__file__).parent.parent}")

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(
    scene_name,
    scene_idx,
    wm,
    initial_poses,
    num_steps,
    target_object,
    output_dir,
    visualize_rollout
):

    scene_cfg_path = f"/n/fs/ap-project/GSDataGen/configs/environments/pybullet/{scene_name}.yaml"

    scene = PyBulletScene(
        device=device,
        img_resize_shape=224,
        pb_config_path=scene_cfg_path,
        scene_idx=scene_idx
    )

    for i, initial_pose in enumerate(initial_poses):

        print(f"Planning for initial pose {i}")
        video_name = Path(f"{output_dir}/test_{i}.mp4")
        video_name.parent.mkdir(parents=True, exist_ok=True)
        video_name = str(video_name)
        
        video_shape = (224 * 2, 224) # TODO
        
        # set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_name, fourcc, 10, video_shape)

        planner = VLMPlanner(device, target_object, k=3, step_size_limit=[0.01, 0.05]) # hard code

        curr_pose = initial_pose

        for step in range(num_steps):
            # get current observation
            current_obs_img = scene.get_obs_from_pose(curr_pose, type="img")

            # write to video
            frame = np.array(current_obs_img)  # Convert to NumPy array
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR

            frame_resized = cv2.resize(frame, (224, 224))

            # Create a black frame of size (448, 224)
            black_frame = np.zeros((224, 448, 3), dtype=np.uint8)

            # Place the resized frame on the right half
            black_frame[:, 224:] = frame_resized

            # Write index to frame
            cv2.putText(black_frame, f"i={step}", (234, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

            # Write frame to video
            video_writer.write(black_frame)
            
            pred_actions = [
                [0, 0, 0, 0.2, 0, 0],
                [0, 0, 0, 0, 0.2, 0],
                [0, 0, 0, 0, 0, 0.2],
            ]

            if pred_actions is None or len(pred_actions) == 1:
                print("Finished at step", step)
                break
            
            # test roll-out in world model
            max_idx = max_reward = 0
            best_actions = None

            for i, pred_action in enumerate(pred_actions):
                start_pose = curr_pose

                # interpolate wm actions
                interpolated_actions = planner.interpolate_action(pred_action)
                
                # initial obs from wm
                current_obs = wm.transform(current_obs_img).unsqueeze(0).to(device)
                z = wm.encoder(current_obs)
                cumulative_reward = 0
                

                print(len(interpolated_actions)) # TODO cumulative or last?
                # roll out
                for j, action in enumerate(interpolated_actions): # TODO side-by-side?

                    # predicted reward from current latent
                    raw_reward = wm.rewards_predictor(z)
                    predicted_reward = wm.conf_pred_output_act(raw_reward).item()

                    cumulative_reward += predicted_reward

                    # visualize decoded rollout obs for debugging purposes
                    if visualize_rollout:
                        # ground truth observation
                        obs_img = scene.get_obs_from_pose(start_pose, type="numpy")
                        obs_img = cv2.cvtColor(obs_img, cv2.COLOR_RGB2BGR)
                        # obs_img = wm.inv_transform(wm.transform(obs).to(device))
                        # obs_img = np.clamp(obs_img, min=0, max=1)

                        # world model prediction
                        z_decode, _ = wm.decoder(z.unsqueeze(1))
                        z_decode = wm.inv_transform(z_decode)
                        z_decode = torch.clamp(z_decode, min=0, max=1)

                        # write decoded frame to video
                        frame = (z_decode.squeeze().detach().cpu().numpy()* 255).astype(np.uint8)
                        frame = np.moveaxis(frame, 0, -1)  # Move channel dimension to the end

                        # assert frame shape matches video_shape
                        # assert frame.shape[0] == video_shape[0] and frame.shape[1] == video_shape[1], f"Frame shape {frame.shape} does not match video shape {video_shape}"

                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
                        cv2.putText(frame, f"action={i}", (10, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                        cv2.putText(frame, f"j={j}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

                        reward_text = f"P={predicted_reward:.2f}, C={cumulative_reward:.2f}"
                        cv2.putText(frame, reward_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

                        resized_reconstruction = cv2.resize(frame, (224, 224))
                        resized_obs = cv2.resize((obs_img * 255).astype(np.uint8), (224, 224))
                        # concatenate the two images
                        combined_frame = np.hstack((resized_reconstruction, resized_obs))
                        # write to video
                        video_writer.write(cv2.resize(combined_frame, video_shape))

                    # apply action to latent
                    action_tensor = torch.tensor(action).to(device)
                    # switch the order of the first 3 element with the last 3
                    # action_tensor = torch.cat((action_tensor[3:], action_tensor[:3]))
                    print(action_tensor)
                    z = wm.dynamics_predictor(z, action_tensor.unsqueeze(0).unsqueeze(0))
                    start_pose = scene.apply_action(start_pose, np.array(action))

                
                if cumulative_reward > max_reward:
                    max_reward = cumulative_reward
                    max_idx = i
                    best_actions = interpolated_actions
                print(f"Step {step}, Action {i}, Cumulative Reward: {cumulative_reward:.2f}, Max Reward: {max_reward:.2f}")
            
            print("executing action", max_idx)

            for j in range(len(best_actions)):
                print(best_actions[j])
                # get current observation
                current_obs_img = scene.get_obs_from_pose(curr_pose, type="img")

                # write to video
                frame = np.array(current_obs_img)  # Convert to NumPy array
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR

                # write index to frame
                cv2.putText(frame, f"i={step}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                frame_resized = cv2.resize(frame, (224, 224))

                black_frame = np.zeros((224, 448, 3), dtype=np.uint8)
                black_frame[:, 224:] = frame_resized
                
                video_writer.write(cv2.resize(black_frame, video_shape))

                # video_writer.write(cv2.resize(frame, video_shape))
                curr_pose = scene.apply_action(curr_pose, np.array(best_actions[j]))

            # curr_pose = scene.apply_action(curr_pose, np.array(pred_actions[max_idx]))
        
        video_writer.release()
        cv2.destroyAllWindows()
        
        # output path
        output_video = f"{video_name}_temp.mp4"
        
        # encode with H264 codec
        os.system(f"ffmpeg -hide_banner -loglevel error -y -i {video_name} -vcodec libx264 {output_video}")

        # delete the temporary file
        Path(video_name).unlink()
        os.rename(f"{output_video}", f"{video_name}")
    
        print(f"*** Video saved as {video_name} ***")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, required=False, default="eight_objects", help="Scene name")
    parser.add_argument("--scene_idx", type=int, required=False, default=0, help="Scene index")
    parser.add_argument("--init_pose_file_name", type=str, required=False, default="eight_scene0",
                        help="Name of the file to use for the initial poses.")
    parser.add_argument("--init_file_dir", type=str, required=False, default="/n/fs/ap-project/active_perception_world_models/test_inputs/initial_poses",
                        help="Parent directory containing the initial poses.")
    parser.add_argument("--output_dir", type=str, required=False, default="0301", help="Output directory")
    parser.add_argument("--num_steps", type=int, required=False, default="200", help="Number of steps to plan for")
    parser.add_argument("--target_object", type=str, required=False, default="mug", help="Target object to optimize rewards for")
    parser.add_argument("--wm_cfg", type=str, required=True, help="world model config file")
    parser.add_argument("--wm_ckpt_path", type=str, required=True, help="world model checkpoint path")
    parser.add_argument("--visualize_rollout", action="store_true", help="Visualize the planning process")

    # parse the arguments
    args = parser.parse_args()
    # initialization file path for poses
    init_file_path = Path(f"{args.init_file_dir}/pybullet/{args.init_pose_file_name}.json")
    all_initial_poses = dict(json.load(open(init_file_path)))
    initial_pose_list = all_initial_poses["poses"]
    
    # make directory, if necessary
    os.makedirs(args.output_dir, exist_ok=True)

    # load world model config
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path="configs", version_base="1.1")
    cfg = hydra.compose(config_name=args.wm_cfg)
    
    wm = load_wm_for_inference(cfg, args.wm_ckpt_path)
    
    main(
        scene_name=args.scene_name,
        scene_idx=args.scene_idx,
        wm=wm,
        initial_poses=initial_pose_list,
        num_steps=args.num_steps,
        target_object=args.target_object,
        output_dir=args.output_dir,
        visualize_rollout=args.visualize_rollout
    )
