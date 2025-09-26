import argparse
import torch.multiprocessing as mp

import pprint
import os
import sys
import yaml
from pathlib import Path
from PIL import Image

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import json

from glob import glob

import numpy as np
import torch

from src.models.worldmodel import WorldModel
from src.scenes.gsplat_scene import GSScene # TODO
from src.scenes.pybullet_scene import PyBulletScene
from src.tests.test_planner import WMTestPlanner
from src.utils.gdino import compute_reward

sys.path.append(f"{Path(__file__).parent.parent}")

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(
    cfg,
    model_weights_path,
    run_name,
    scene_type,
    output_dir,
    init_file_path,
    target_object,
    num_steps: int = 200,
    enable_action_damping: bool = False,
    scene_idx: int = 0,
    initialize_action: bool = False,
):
    # instantiate the encoder
    encoder = instantiate(cfg.encoder)

    if encoder.__class__.__name__ == "VisionTransformer":
        num_patches = encoder.patch_embed.num_patches
    else:
        num_patches = (cfg.shared.img_resize_shape // encoder.patch_size) ** 2

    # instantiate the dynamics predictor
    dynamics_predictor = instantiate(cfg.dynamics_predictor,
                                     num_patches=num_patches,
                                     embed_dim=encoder.embed_dim,
                                     num_heads=encoder.num_heads,
                                     history_length=cfg.training.latent_state_history_length)
    
    # instantiate the rewards predictor
    rewards_predictor = instantiate(cfg.rewards_predictor,
                                    num_patches=num_patches,
                                    embed_dim=encoder.embed_dim,
                                    cond_dim=cfg.shared.cond_dim)
    
    # raise an error if there is no decoder
    if cfg.decoder is None:
        raise ValueError("Decoder is required for this test script")
    
    decoder = instantiate(cfg.decoder, emb_dim=encoder.embed_dim)
    
    #  instantiate the world model
    wm = WorldModel(cfg=cfg,
                    device=device,
                    encoder=encoder,
                    dynamics_predictor=dynamics_predictor,
                    rewards_predictor=rewards_predictor,
                    decoder=decoder,
                    mode="inference",
                    ablation=None,
                    encoder_frozen=True)
    
    # load weights
    wm.load_rewards_predictor_weights(model_weights_path)
    wm.load_dynamics_predictor_weights(model_weights_path)
    wm.load_decoder_weights(model_weights_path)

    if scene_type == "pybullet":
        # extract last substring after "\" # TODO
        scene_name = cfg.dataset.root_path.split("/")[-1]
        print(f"scene_name: {scene_name}")
        
        scene_cfg_path = f"/n/fs/ap-project/GSDataGen/configs/environments/pybullet/{scene_name}.yaml"
        scene = PyBulletScene(
            device=device,
            img_resize_shape=cfg.shared.img_resize_shape,
            pb_config_path=scene_cfg_path,
            scene_idx=scene_idx
        )
    else:
        scene_name = cfg.dataset.root_path.split("/")[-1]
        print(f"scene_name: {scene_name}")
        
        scene_cfg_path = f"/n/fs/ap-project/GSDataGen/configs/environments/gsplat/{scene_name}.yaml"
        scene = GSScene(
            device=device, 
            img_resize_shape=wm.img_resize_shape,
            gs_config_path=scene_cfg_path,
            scene_idx=scene_idx
        )
        print('Not implemented')

    # Set up test
    test = WMTestPlanner(
        scene, 
        wm, 
        output_dir,
        target_object=target_object,
        initialize_action=initialize_action,
    )
    
    # load initial position
    all_initial_poses = dict(json.load(open(init_file_path)))
    
    # TODO fix
    initial_pose_list = all_initial_poses["poses"] # list of cam_to_world_mat's
    
    # Test dynamics consistency
    test.eval_planner(
        initial_pose_list=initial_pose_list,
        run_name=run_name,
        num_steps=num_steps,
        enable_action_damping=enable_action_damping,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj_name", type=str, required=False, default="2025-03-01-world_model_multiscene", help="Name of the project")
    parser.add_argument("--run_name", type=str, required=False, default="dino-frozen-decoder-vqvae-19:06:59-tbanana_perm_s2", help="Name of the run")
    parser.add_argument("--config_path", type=str, required=False, default=None, help="Path for the config file")
    parser.add_argument("--config_name", type=str, required=False, default="config", help="Name of the config file (without extension)")
    parser.add_argument("--ckpt", type=int, required=False, default=10, help="Checkpoint")
    parser.add_argument("--model_root_path", type=str, required=False, default=None, help="root path for the model")
    parser.add_argument("--model_weights_path", type=str, required=False, default=None, help="Path for the saved models")
    parser.add_argument("--init_pose_file_name", type=str, required=False, default="test1",
                        help="Name of the file to use for the initial poses.")
    parser.add_argument("--init_file_dir", type=str, required=False, default="/n/fs/ap-project/active_perception_world_models/initial_poses",
                        help="Parent directory containing the initial poses.")
    parser.add_argument("--output_dir", type=str, required=False, default="0301", help="Output directory")
    parser.add_argument("--num_steps", type=int, required=False, default="200", help="Number of steps to plan for")
    parser.add_argument("--target_object", type=str, required=False, default="mug", help="Target object to optimize rewards for")
    parser.add_argument("--enable_action_damping", action="store_true", help="Whether to enable action damping")
    parser.add_argument("--enable_birds_eye_view_init", action="store_true", help="Whether to initialize the action biased to give a top-down view")
    parser.add_argument("--scene_idx", type=int, required=False, default=0, help="Scene index")


    # parse the arguments
    args = parser.parse_args()

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

    # initialization file path for poses
    init_file_path = Path(f"{args.init_file_dir}/{cfg.dataset.scene_type}/{args.init_pose_file_name}.json")
    
    # make directory, if necessary
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(
        cfg, 
        model_weights_path=model_weights_path,
        run_name=args.run_name,
        scene_type=cfg.dataset.scene_type, 
        output_dir=args.output_dir,
        target_object=args.target_object,
        init_file_path=init_file_path,
        num_steps=args.num_steps,
        enable_action_damping=args.enable_action_damping,
        scene_idx=args.scene_idx,
        initialize_action=args.enable_birds_eye_view_init,
    )
