from typing import Literal
import argparse
import os
import sys
import yaml
import random
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

from src.tests.wm_planner import WMPlanner
from src.tests.vlm_planner import VLMPlanner

from src.eval.evaluator import PybulletEvaluator

sys.path.append(f"{Path(__file__).parent.parent}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO refactor to another file
wm_model_dict = {
    "kitchen_7_50scenes": {
        "config_name": '0405',
        "env_type": 'pybullet',
        "proj_name": "2025-04-05-world_model_multiscene",
        "run_name": "dino-frozen-decoder-vqvae-02:15:31-0405"
    },
}

def get_ckpt_path_from_wm_model(wm_model, ckpt):
    """
    Get the checkpoint path for a given world model and checkpoint number.
    """
    wm_model_info = wm_model_dict[wm_model]
    env_type = wm_model_info["env_type"]
    proj_name = wm_model_info["proj_name"]
    run_name = wm_model_info["run_name"]
    model_weights_path = f"/n/fs/ap-project/active_perception_world_models/logs/{env_type}/{proj_name}/{run_name}/{run_name}-ep{ckpt}.pth.tar"
    return model_weights_path

# TODO: Will refactor this code
# planner_names = ["wm-only", 'vlm-action', 'vlm-action-v2']
planner_names = ["wm-only"]
wm_model_names = ["kitchen_7_50scenes"]
# wm_checkpoints = [50, 70, 100]
wm_checkpoints = [70] # 70, 100

# target object
# target_object = "banana"
target_object = "green_bowl"

# test scene
scene_id = 60

# test configs
test_configs = [f"scene={scene_id}-init=random-tar={target_object.replace('_', '')}-occ-n=5"]

# other params
num_experiments_per_config = 1
num_steps = 200

# planner options
enable_action_damping: bool = False
enable_action_initialization: bool = True
wm_action_init_mode: Literal["topdown", "grid"] = "grid"

# TODO: Remove this option
# option to use random experiment IDs
use_random_experiment_id: bool = False

# experiment ID tag
exp_id_tag = f"_{wm_action_init_mode}" if planner_names[0].lower() == "wm-only" else ""

# setup output path
experiment_id = random.randint(100000, 999999) if use_random_experiment_id else "debug"
experiment_id = f"{experiment_id}{exp_id_tag}"
   
print(f"Experiment ID: {experiment_id}")
results = {}

# output_dir = f"/n/fs/ap-project/active_perception_world_models/experiments/{experiment_id}"
output_dir = f"/n/fs/wmdev/projects/active_perception_temp_master/multi_target_a_perc_cost_to_go/experiments/{experiment_id}"
os.makedirs(output_dir, exist_ok=True)

# dump experiment config
experiment_config = {
    "planner_names": planner_names,
    "wm_model_names": wm_model_names,
    "wm_checkpoints": wm_checkpoints,
    "test_configs": test_configs,
    "num_experiments_per_config": num_experiments_per_config,
    "num_steps": num_steps,
}

with open(os.path.join(output_dir, f"experiment_config_{experiment_id}.yaml"), 'w') as f:
    yaml.dump(experiment_config, f)

for planner_name, wm_model_name, wm_ckpt, test_config in itertools.product(planner_names, wm_model_names, wm_checkpoints, test_configs):
    print(f"planner_name: {planner_name}, WM wm_model_name: {wm_model_name}, wm_ckpt: {wm_ckpt}, test_config: {test_config}")

    wm_model = wm_model_dict[wm_model_name]
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path="configs", version_base="1.1")
    cfg = hydra.compose(config_name=wm_model["config_name"])
    model_weights_path = get_ckpt_path_from_wm_model(wm_model_name, wm_ckpt)

    # create world model object
    wm = load_wm_for_inference(cfg, model_weights_path)

    # create planner
    if planner_name == "wm-only":
        planner = WMPlanner(wm, 
                            enable_action_damping=enable_action_damping,
                            enable_action_initialization=enable_action_initialization,
                            action_init_mode=wm_action_init_mode)
    elif planner_name == "vlm":
        raise NotImplementedError("VLM planner not implemented") # TODO
    
    # set up evaluation 
    evaluator = PybulletEvaluator(
        planner=planner, 
        experiment_name=test_config,
        output_dir=output_dir,
        output_video=True,
        num_steps=num_steps,
    )
    
    # prefix for the video name
    video_name_prefix = f"{planner_name}_{wm_model_name}_{wm_ckpt}_{test_config}"

    for i in range(num_experiments_per_config):
        # create tuple dict key
        experiment_key = (planner_name, wm_model_name, wm_ckpt, test_config, i)
        print(f"Experiment {i+1}/{num_experiments_per_config}")
        metrics = evaluator.evaluate(save=False, video_name_prefix=f"{video_name_prefix}_run{i}")
        
        # save results
        results[experiment_key] = metrics

# dump results to json
results_path = os.path.join(output_dir, f"results_{experiment_id}.pkl")

with open(results_path, 'wb') as f:
    pickle.dump(results, f)