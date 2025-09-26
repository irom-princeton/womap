import argparse
import os
import sys
import yaml
from pathlib import Path
from PIL import Image

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import json


import numpy as np
import torch
from src.utils.wm_utils import load_wm_for_inference

from src.tests.wm_planner import WMPlanner
from src.tests.vlm_planner import VLMPlanner

from src.eval.evaluator import PybulletEvaluator

sys.path.append(f"{Path(__file__).parent.parent}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(
    wm,
    planner_type,
    output_dir,
    num_steps: int = 200,
    test_config_name: str = "0331_unseen",
):
    # TODO: Refactor to support sepecification of the planner externally, e.g., via a config.
    # planner 
    if planner_type == "default":
        planner = WMPlanner(wm, 
                            enable_action_damping=False,
                            enable_random_initialization=False)
    elif planner_type == "vlm":
        planner = VLMPlanner(wm, 
                             enable_action_damping=False,
                             enable_random_initialization=False)
    
    # set up evaluation 
    evaluator = PybulletEvaluator( # TODO create evaluator class
        planner=planner, #TODO need to set up planner from wm
        experiment_name=test_config_name, #TODO refactor to specify the experiment name through a config parameter.
        output_dir=output_dir,
        output_video=True,
        num_steps=num_steps,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # test parameters
    parser.add_argument("--test_config_name", type=str, required=False, default="0331_unseen", help="Name of the test config file (without extension)")
    parser.add_argument("--num_steps", type=int, required=False, default="200", help="Number of steps to plan for")
    parser.add_argument("--output_dir", type=str, required=False, default="0301", help="Output directory")
    # environment config
    parser.add_argument("--config_path", type=str, required=False, default=None, help="Path for the config file")
    parser.add_argument("--config_name", type=str, required=False, default="config", help="Name of the config file (without extension)")
    # planner
    parser.add_argument("--planner", type=str, required=False, default="default", help="Type of planner")
    # world model 
    parser.add_argument("--proj_name", type=str, required=False, default="2025-03-01-world_model_multiscene", help="Name of the project")
    parser.add_argument("--run_name", type=str, required=False, default="dino-frozen-decoder-vqvae-19:06:59-tbanana_perm_s2", help="Name of the run")
    parser.add_argument("--model_root_path", type=str, required=False, default=None, help="root path for the model")
    parser.add_argument("--model_weights_path", type=str, required=False, default=None, help="{Path for the saved models}")
    parser.add_argument("--ckpt", type=int, required=False, default=10, help="Checkpoint")

    # parse the arguments
    args = parser.parse_args()
    
    # make directory, if necessary
    os.makedirs(args.output_dir, exist_ok=True)

    # initialize the config path
    if args.config_path is None:
        args.config_path = f"{args.model_root_path}/logs/gsplat/{args.proj_name}/{args.run_name}"
        
    if args.model_weights_path is None:
        model_weights_path = f"{args.config_path}/{args.run_name}-ep{args.ckpt}.pth.tar"
    else:
        model_weights_path = f"{args.model_root_path}/{args.model_weights_path}"
    
    # load config
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=args.config_path, version_base="1.1")
    cfg = hydra.compose(config_name=args.config_name)
    
    wm = load_wm_for_inference(cfg, model_weights_path)
    
    main(
        wm=wm,
        planner_type=args.planner,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        test_config_name=args.test_config_name,
    )
