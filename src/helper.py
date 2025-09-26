# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys

import torch
import torch.nn as nn

import src.models.vision_transformer as vit
from src.models.encoder.dino import DinoV2Encoder
from src.models.encoder.clip import CLIPEncoder
from src.utils.schedulers import WarmupCosineSchedule, CosineWDSchedule
from src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_checkpoint(
    device,
    r_path,
    encoder,
    predictor,
    target_encoder,
    detection_confidence_predictor,
    opt,
    scaler,
    train=True
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device("cpu"))
        epoch = checkpoint["epoch"]

        # -- loading encoder
        if encoder is not None:
            pretrained_dict = checkpoint["encoder"]
            if not train:
                pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
            msg = encoder.load_state_dict(pretrained_dict)
            logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")
        else:
            raise ValueError("The encoder is None.")
            
        # -- loading predictor
        if predictor is not None:
            pretrained_dict = checkpoint["predictor"]
            if not train:
                pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
            msg = predictor.load_state_dict(pretrained_dict)
            logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

        # -- loading target_encoder
        if target_encoder is not None:
            # print(list(checkpoint.keys()))
            pretrained_dict = checkpoint["target_encoder"]
            if not train:
                pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")
        
        # -- loading detection confidence predictor
        if detection_confidence_predictor is not None:
            # print(list(checkpoint.keys()))
            pretrained_dict = checkpoint["detection_confidence_predictor"]
            if not train:
                pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
            msg = detection_confidence_predictor.load_state_dict(pretrained_dict)
            logger.info(
                f"loaded pretrained detection-confidence-predictor from epoch {epoch} with msg: {msg}"
            )
        # -- loading optimizer
        if opt is not None:
            opt.load_state_dict(checkpoint["opt"])
        if scaler is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        logger.info(f"loaded optimizers from epoch {epoch}")
        logger.info(f"read-path: {r_path}")
        del checkpoint

    except Exception as e:
        logger.info(f"Encountered exception when loading checkpoint {e}")
        epoch = 0

    return encoder, predictor, target_encoder, detection_confidence_predictor, opt, scaler, epoch

def load_predictor_checkpoint(
    device,
    r_path,
    predictor,
    train=True
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device("cpu"))
        epoch = checkpoint["epoch"]

        pretrained_dict = checkpoint["predictor"]
        if not train:
            pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(f"loaded predictor from epoch {epoch} with msg: {msg}")

        del checkpoint

    except Exception as e:
        logger.info(f"pred - Encountered exception when loading checkpoint {e}")
        epoch = 0

    return predictor


def init_model(
    device,
    encoder_name="vit",
    encoder_arch_name=None,
    patch_size=16,
    crop_size=224,
    pred_depth=6,
    pred_emb_dim=384,
    conf_pred_depth=2,
    conf_pred_num_heads=3,
    conf_pred_output_act=nn.Sigmoid(),
    conf_pred_output_dim=1,
    action_dim=7,
):
    # encoder's name
    encoder_name = encoder_name.lower()
    
    # initialize the default model names
    if encoder_arch_name is None and encoder_name == "vit":
        encoder_arch_name = "vit_small"
    elif encoder_arch_name is None and encoder_name == "dino":
        encoder_arch_name = "dinov2_vits14"
        # only one feature key supported for now.
        dinov2_feature_key = "x_norm_patchtokens"
    elif encoder_arch_name is None and encoder_name == "clip":
        encoder_arch_name = "ViT-L/14" # "ViT-B/32", "ViT-L/14", "ViT-L/14@336px", or "RN50x64"
    
    # initialize the encoder
    if encoder_name == "vit":
        encoder = vit.__dict__[encoder_arch_name](img_size=[crop_size], patch_size=patch_size)
    elif encoder_name == "dino":
        encoder = DinoV2Encoder(name=encoder_arch_name, feature_key=dinov2_feature_key)
    elif encoder_name == "clip":
        encoder = CLIPEncoder(name=encoder_arch_name)
        
    # number of patches in the encoder
    if encoder_name == "dino" or encoder_name == "clip":
        num_patches = (crop_size // encoder.patch_size) ** 2
    else:
        num_patches = encoder.patch_embed.num_patches
    
    # initialize the predictor
    predictor = vit.__dict__["vit_predictor"](
        num_patches=num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads,
        action_dim=action_dim,
    )

    # predictor for the detection confidence
    detection_confidence_predictor = vit.__dict__["vit_output_predictor"](
        num_patches=num_patches,
        embed_dim=encoder.embed_dim,
        conf_pred_output_dim=conf_pred_output_dim,
        predictor_embed_dim=pred_emb_dim,
        output_activation=conf_pred_output_act,
        depth=conf_pred_depth,
        num_heads=conf_pred_num_heads,
    )

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    # Initialize the encoder's weight only for ViTs
    if encoder_name == "vit":
        for m in encoder.modules():
            init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    for m in detection_confidence_predictor.modules():
        init_weights(m)

    # move to device
    encoder.to(device)
    predictor.to(device)
    detection_confidence_predictor.to(device)
    
    logger.info(encoder)
    return encoder, predictor, detection_confidence_predictor


def init_opt(
    encoder,
    predictor,
    decoder,
    detection_confidence_predictor,
    freeze_encoder: bool,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25,
):
    param_groups = []
    
    # encoder
    if not freeze_encoder:
        param_groups.extend([
            {
                "params": (
                    p
                    for n, p in encoder.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1)
                )
            },
            {
                "params": (
                    p
                    for n, p in encoder.named_parameters()
                    if ("bias" in n) or (len(p.shape) == 1)
                ),
                "WD_exclude": True,
                "weight_decay": 0,
            },
        ]
        )

    # decoder
    if decoder is not None:
        param_groups.extend(
            [
                {
                    "params": (
                        p
                        for n, p in decoder.named_parameters()
                        if ("bias" not in n) and (len(p.shape) != 1)
                    )
                },
                {
                    "params": (
                        p
                        for n, p in decoder.named_parameters()
                        if ("bias" in n) or (len(p.shape) == 1)
                    ),
                    "WD_exclude": True,
                    "weight_decay": 0,
                },
            ]
        )
    
    # predictor
    if predictor is not None:
        param_groups.extend([
            {
                "params": (
                    p
                    for n, p in predictor.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1)
                )
            },
            {
                "params": (
                    p
                    for n, p in predictor.named_parameters()
                    if ("bias" in n) or (len(p.shape) == 1)
                ),
                "WD_exclude": True,
                "weight_decay": 0,
            },
        ]
        )
    
    # rewards model
    if detection_confidence_predictor is not None:
        param_groups.extend([
            {
                "params": (
                    p
                    for n, p in detection_confidence_predictor.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1)
                )
            },
            {
                "params": (
                    p
                    for n, p in detection_confidence_predictor.named_parameters()
                    if ("bias" in n) or (len(p.shape) == 1)
                ),
                "WD_exclude": True,
                "weight_decay": 0,
            },
        ]
        )
    
    logger.info("Using AdamW")
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler