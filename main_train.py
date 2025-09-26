import argparse
import os
import sys
import yaml
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

import wandb
import torch

from src.models.worldmodel import WorldModel

sys.path.append(f"{Path(__file__).parent.parent}")

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(cfg, ablation, encoder_frozen, enable_decoder, verbose_print, wandbproj, wandbexp):
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
    
    if enable_decoder:
        # instantiate the decoder
        decoder = instantiate(cfg.decoder,
                              emb_dim=encoder.embed_dim)
    else:
        decoder = None
        
    # instantiate the world model
    wm = WorldModel(cfg=cfg,
                    device=device,
                    encoder=encoder,
                    dynamics_predictor=dynamics_predictor,
                    rewards_predictor=rewards_predictor,
                    decoder=decoder,
                    mode="train",
                    ablation=ablation,
                    encoder_frozen=encoder_frozen,
                    verbose_print=verbose_print)
    
    wandb.init(project=wandbproj, name=wandbexp)
    wm.train(wandb_projname=wandbproj, wandb_expname=wandbexp)

if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, required=False, default="main", help="Name of the config file (without extension)")
    parser.add_argument("--ablation", type=str, required=False, default=None, help="Ablation study to run")
    parser.add_argument("--unfreeze_encoder", action="store_true", help="Whether to unfreeze the encoder")
    parser.add_argument("--enable_decoder", action="store_true", help="Whether to enable a decoder")
    parser.add_argument("--verbose", "-v", action="store_true", help="Whether to enable the verbose print mode.")
    parser.add_argument("--wandbproj", type=str, required=True, help="Name of the project")
    parser.add_argument("--wandbexp", type=str, required=True, help="Name of the experiment in the project")

    args = parser.parse_args()
    
    # load the config
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path="configs", version_base="1.1")
    cfg = hydra.compose(config_name=args.config_name)

    # save the config
    folder = f"{Path(__file__).parent.parent}/{cfg.logging.folder}/{args.wandbproj}/{args.wandbexp}"
    os.makedirs(folder, exist_ok=True)
    config_path = os.path.join(folder, "config.yaml")
    OmegaConf.save(cfg, config_path)
    
    # default: no ablations, encoder_frozen=True
    main(cfg, 
         ablation=args.ablation, 
         encoder_frozen=not args.unfreeze_encoder, 
         enable_decoder=args.enable_decoder,
         verbose_print=args.verbose,
         wandbproj=args.wandbproj, 
         wandbexp=args.wandbexp)