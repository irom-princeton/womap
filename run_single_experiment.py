#  the new experiment file

from typing import Literal
import argparse
import os
import sys
from pathlib import Path
from PIL import Image

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import json
import itertools
import pickle
import numpy as np
import torch
from src.utils.wm_utils import load_wm_for_inference

# planner
from src.tests.simple_planner import SimplePlanner
from src.tests.wm_planner import WorldModelPlanner

# action generator
from src.tests.action_generator import *

# evaluator
from src.eval.evaluator import PybulletEvaluator


sys.path.append(f"{Path(__file__).parent.parent}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_planner_from_config(planner_name, planner_config_dict, wm_model_dict):
    
    # === WORLD MODEL === #
    
    wm = None

    if "wm" in planner_name:
        if wm_dict is None:
            raise ValueError("World model is required for planner")
        
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(config_path="configs", version_base="1.1")
        cfg = hydra.compose(config_name=wm_model_dict["config_name"])

        # create world model object
        wm = load_wm_for_inference(cfg, wm_model_dict["ckpt_path"])
    
    
    # === ACTION GENERATOR === #
    
    action_generator = None

    if "all" in planner_name:
        action_generator = StepActionGenerator(query_mode=planner_config_dict["query_mode"])
    elif "grid" in planner_name:
        action_generator = GridActionGenerator()
    elif "cem" in planner_name:
        action_generator = CEMActionGenerator() # you can pass in additional parameters, e.g., the mean and standard dev.
    elif "vlm" in planner_name:
        query_mode = planner_config_dict["query_mode"]
        num_proposals = planner_config_dict["n_prop"]
        action_generator = VLMActionGenerator(query_mode=query_mode, num_proposals=num_proposals)
    else:
        raise NotImplementedError("Not implemented: ", planner_name)
    
    
    # === PLANNER == #
    
    planner = None
    if "wm" in planner_name:
        
        if "default_action_init" not in planner_config_dict.keys():
            planner_config_dict["default_action_init"] = "random"
        
        reintialization_threshold = planner_config_dict["reint_thresh"]
        planner = WorldModelPlanner(
            action_generator=action_generator,
            world_model=wm,
            device=device,
            enable_action_damping=False,
            default_action_initialization=planner_config_dict["default_action_init"],
            reinitialize_action_reward_threshold=reintialization_threshold,
            num_proposals=planner_config_dict["n_prop"]
        )

    else:
        # no world model for action evaluation
        planner = SimplePlanner(
            action_generator=action_generator,
            device=device
        )
    
    return planner


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--exp_config",
        type=str,
        required=True,
        help="path to your experiment config file"
    )
    
    argparser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="output toplevel directory"
    )
    
    argparser.add_argument(
        "--time_limit",
        type=int,
        required=True,
        help="number of seconds to run the experiment"
    )
    
    argparser.add_argument(
        "--model_id",
        type=int,
        required=True,
        help="model id to run"
    )
    
    argparser.add_argument(
        "--exp_file_path",
        type=str,
        required=True,
        help="path to the experiment file"
    )
    
    args = argparser.parse_args()
    exp_config_path = args.exp_config
    result_dir = args.output_dir
    time_limit = args.time_limit
    model_id = args.model_id
    exp_file_path = args.exp_file_path
    
    
    # load experiment json config
    with open(exp_config_path, "r") as f:
        exp_config = json.load(f)
        
    environment_type = exp_config["environment_type"]
    environment = exp_config["environment"]
    boundaries = exp_config["boundaries"]
    top_output_dir = f"{result_dir}/{environment}"
    
    # ===== LOADING PATHS ===== #
    
    experiment_dir = f"/n/fs/robot-data/womap/experiments/{environment}"
    
    # world model dict specify config and checkpoint for each available world model
    wm_dict_path = f"/n/fs/ap-project/active_perception_world_models/world_model_configs/{environment}.json"
    
    with open(wm_dict_path, "r") as f:
        wm_dict = json.load(f)
    
    # ===== LOADING MODELS AND EXPERIMENTS ===== #
    
    all_model_dictss = exp_config["models"]
    
    model_dict = all_model_dictss[model_id]
    
    def dict_to_str(d):
        return "_".join([f"{k}={v}" for k, v in d.items() if v is not None])
    
    
    planner_name = model_dict["planner"]
    planner_config_dict = model_dict["planner_config"]
    world_model_key = model_dict["world_model"]
    
    world_model_dict = None
    if world_model_key is not None:
        world_model_dict = wm_dict[world_model_key]

    # load the model config
    planner = load_planner_from_config(planner_name, planner_config_dict, world_model_dict)
    
    model_prefix = f"{planner_name}-{dict_to_str(planner_config_dict)}"
    
    # set output directory
    model_output_dir = os.path.join(top_output_dir, model_prefix)
    if world_model_key is not None:
        model_output_dir = os.path.join(model_output_dir, world_model_key)
        
    test_name = os.path.splitext(os.path.basename(exp_file_path))[0] 
    result_output_dir = os.path.join(model_output_dir, test_name)
    
    os.makedirs(result_output_dir, exist_ok=True)
    
    
    if environment_type == "pybullet":
        evaluator = PybulletEvaluator(
            planner=planner, 
            experiment_file_path = exp_file_path,
            boundaries=boundaries,
            video_outupt_dir=result_output_dir,
            video_name_prefix="v",
            video_name_suffix="",
            output_video=True,
            time_limit=time_limit,
            speed=0.05
        )
    else:
        raise NotImplementedError("Not implemented: ", environment_type)
        evaluator = GaussianSplatEvaluator(
            planner=planner, 
            experiment_file_path = exp_file_path,
            boundaries=boundaries,
            video_outupt_dir=result_output_dir,
            video_name_prefix="v",
            video_name_suffix="",
            output_video=True,
            time_limit=time_limit,
            speed=0.05 # TODO decide speed in gs
        )
        
    # run the experiment
    metrics = evaluator.evaluate()
    
    # dump the results
    result_output_path = os.path.join(result_output_dir, f"result.pkl")
    with open(result_output_path, 'wb') as f:
        pickle.dump(metrics, f)
        
