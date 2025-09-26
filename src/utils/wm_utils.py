import os
import sys
from pathlib import Path
import hydra
from hydra.utils import instantiate
import torch

from prettytable import PrettyTable

sys.path.append(f"{Path(__file__).parent.parent}")

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_wm_for_inference(
    cfg,
    model_weights_path
):
    from src.models.worldmodel import WorldModel

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
                                     num_heads=encoder.num_heads)
    
    # instantiate the rewards predictor
    rewards_predictor = instantiate(cfg.rewards_predictor,
                                    num_patches=num_patches,
                                    embed_dim=encoder.embed_dim)
    
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

    return wm

def print_number_of_model_parameters(model, title=""):
    # table of results
    table = PrettyTable()
    table.title = title
    table.field_names = ["Type", "Parameters"]
    
    # total number of trainable and non-trainable parameters
    num_train_params = 0
    num_non_train_params = 0
    
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            # increment the counter
            num_train_params += parameter.numel()
        else:
            # increment the counter
            num_non_train_params += parameter.numel()
            
    # update the table
    table.add_row(["Trainable", num_train_params])
    table.add_row(["Non-Trainable", num_non_train_params])
    
    # print
    print(table)